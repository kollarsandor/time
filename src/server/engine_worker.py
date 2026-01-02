import asyncio
import ctypes
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, AsyncIterator
from threading import Lock, Condition, Thread
from queue import Queue, Empty, Full
from src.server.config import ServerConfig
from src.server.engine_wrapper import NativeEngine, MockEngine, EngineInfo
from src.server.metrics import metrics


@dataclass
class InferenceRequest:
    request_id: str
    token_ids: List[int]
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    stream: bool
    created_at: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None
    token_queue: Optional[asyncio.Queue] = None
    prompt_tokens: int = 0

    def __post_init__(self):
        self.prompt_tokens = len(self.token_ids)


@dataclass
class InferenceResult:
    request_id: str
    tokens: List[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float


class EngineWorker:
    def __init__(self, config: ServerConfig, use_mock: bool = False):
        self.config = config
        self.use_mock = use_mock
        if use_mock:
            self.engine: Union[NativeEngine, MockEngine] = MockEngine()
        else:
            self.engine = NativeEngine(
                engine_path=config.engine_path,
                model_dir=config.model_dir,
                max_batch=config.max_concurrency,
                max_seq=config.max_seq_len,
                num_gpus=config.num_gpus,
            )
        self._request_queue: Queue = Queue(maxsize=config.queue_size)
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._stop_event = asyncio.Event()
        self._worker_running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        if not self.engine.load():
            raise RuntimeError("Failed to load engine")
        self._worker_running = True
        self._worker_task = loop.create_task(self._worker_loop())

    async def stop(self):
        self._stop_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        with self._lock:
            for req_id, req_data in list(self._active_requests.items()):
                state_ptr = req_data.get("state_ptr")
                if state_ptr:
                    self.engine.free_request(state_ptr)
            self._active_requests.clear()
        self.engine.shutdown()
        self._worker_running = False

    def is_running(self) -> bool:
        return self._worker_running and self.engine.is_loaded()

    def get_engine_info(self) -> Optional[EngineInfo]:
        return self.engine.get_info()

    def get_queue_depth(self) -> int:
        return self._request_queue.qsize()

    def get_active_count(self) -> int:
        with self._lock:
            return len(self._active_requests)

    async def submit_request(self, request: InferenceRequest) -> Union[InferenceResult, AsyncIterator[int]]:
        if not self._worker_running:
            raise RuntimeError("Worker not running")
        with self._lock:
            if len(self._active_requests) >= self.config.max_concurrency:
                raise RuntimeError("Max concurrency reached")
        if request.stream:
            request.token_queue = asyncio.Queue()
            try:
                self._request_queue.put_nowait(request)
            except Full:
                raise RuntimeError("Request queue full")
            return self._stream_tokens(request)
        else:
            request.future = self._loop.create_future()
            try:
                self._request_queue.put_nowait(request)
            except Full:
                raise RuntimeError("Request queue full")
            return await request.future

    async def _stream_tokens(self, request: InferenceRequest) -> AsyncIterator[int]:
        while True:
            item = await request.token_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                request = self._request_queue.get_nowait()
                await self._process_request(request)
            except Empty:
                pass
            batch_results = await self._process_batch_step()
            if not batch_results:
                await asyncio.sleep(self.config.worker_poll_interval)
            metrics.set_queue_depth(self._request_queue.qsize())
            metrics.set_active_requests(self.get_active_count())
            info = self.engine.get_info()
            if info:
                metrics.set_kv_cache(info.kv_pages_used, info.kv_pages_max)

    async def _process_request(self, request: InferenceRequest):
        start_time = time.time()
        metrics.inc_request()
        metrics.add_prompt_tokens(request.prompt_tokens)
        request_id_int = hash(request.request_id) & 0x7FFFFFFFFFFFFFFF
        state_ptr = self.engine.prefill(
            request_id=request_id_int,
            token_ids=request.token_ids,
            temperature=request.temperature,
            top_p=request.top_p,
            rep_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens,
        )
        if state_ptr is None:
            metrics.inc_error()
            error = RuntimeError("Prefill failed")
            if request.stream and request.token_queue:
                await request.token_queue.put(error)
                await request.token_queue.put(None)
            elif request.future:
                request.future.set_exception(error)
            return
        with self._lock:
            self._active_requests[request.request_id] = {
                "state_ptr": state_ptr,
                "request": request,
                "tokens": [],
                "start_time": start_time,
                "first_token_time": None,
            }

    async def _process_batch_step(self) -> bool:
        with self._lock:
            active_list = list(self._active_requests.items())
        if not active_list:
            return False
        state_ptrs = [data["state_ptr"] for _, data in active_list]
        request_ids = [req_id for req_id, _ in active_list]
        if self.use_mock:
            results = self.engine.batch_decode(state_ptrs)
        else:
            results = self.engine.batch_decode(state_ptrs)
        metrics.record_batch(len(active_list))
        completed = []
        for i, (req_id, (token, done)) in enumerate(zip(request_ids, results)):
            with self._lock:
                if req_id not in self._active_requests:
                    continue
                data = self._active_requests[req_id]
            request = data["request"]
            if token is not None:
                data["tokens"].append(token)
                if data["first_token_time"] is None:
                    data["first_token_time"] = time.time()
                    ttft_ms = (data["first_token_time"] - data["start_time"]) * 1000
                    metrics.record_ttft(ttft_ms)
                metrics.add_tokens(1)
                if request.stream and request.token_queue:
                    await request.token_queue.put(token)
            if done:
                completed.append(req_id)
        for req_id in completed:
            with self._lock:
                if req_id not in self._active_requests:
                    continue
                data = self._active_requests.pop(req_id)
            request = data["request"]
            tokens = data["tokens"]
            start_time = data["start_time"]
            first_token_time = data["first_token_time"]
            end_time = time.time()
            total_ms = (end_time - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
            metrics.record_latency(total_ms)
            finish_reason = "length" if len(tokens) >= request.max_tokens else "stop"
            result = InferenceResult(
                request_id=req_id,
                tokens=tokens,
                finish_reason=finish_reason,
                prompt_tokens=request.prompt_tokens,
                completion_tokens=len(tokens),
                ttft_ms=ttft_ms,
                total_ms=total_ms,
            )
            if not self.use_mock:
                self.engine.free_request(data["state_ptr"])
            if request.stream and request.token_queue:
                await request.token_queue.put(None)
            elif request.future:
                request.future.set_result(result)
        return len(completed) > 0 or len(active_list) > 0
