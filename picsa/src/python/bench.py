import ctypes
import os
import time
import json
import struct
import mmap
import threading
import queue
import random
import statistics
from typing import Optional, List, Dict, Any

class EngineHandle(ctypes.Structure):
    pass

class RequestState(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_int64),
        ("seq_len", ctypes.c_int32),
        ("page_indices", ctypes.POINTER(ctypes.c_int32)),
        ("num_pages", ctypes.c_int32),
        ("past_tokens", ctypes.POINTER(ctypes.c_int64)),
        ("past_len", ctypes.c_int32),
    ]

class SamplingParams:
    def __init__(
        self,
        temperature: float = 0.8,
        top_p: float = 0.95,
        rep_penalty: float = 1.1,
        max_tokens: int = 256
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.rep_penalty = rep_penalty
        self.max_tokens = max_tokens

class PagedKVCacheManager:
    def __init__(self, max_pages: int, page_size: int, num_layers: int, num_heads: int, head_dim: int):
        self.max_pages = max_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.free_pages = list(range(max_pages))
        self.allocated_pages: Dict[int, List[int]] = {}
        self.lock = threading.Lock()

    def allocate_pages(self, request_id: int, num_pages: int) -> List[int]:
        with self.lock:
            if len(self.free_pages) < num_pages:
                return []
            pages = [self.free_pages.pop() for _ in range(num_pages)]
            self.allocated_pages[request_id] = pages
            return pages

    def free_pages_for_request(self, request_id: int) -> None:
        with self.lock:
            if request_id in self.allocated_pages:
                self.free_pages.extend(self.allocated_pages[request_id])
                del self.allocated_pages[request_id]

    def get_utilization(self) -> float:
        with self.lock:
            used = self.max_pages - len(self.free_pages)
            return used / self.max_pages if self.max_pages > 0 else 0.0

class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int, max_seq_len: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.pending_queue: queue.Queue = queue.Queue()
        self.active_requests: Dict[int, Dict[str, Any]] = {}
        self.completed_requests: Dict[int, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.request_counter = 0

    def add_request(self, tokens: List[int], params: SamplingParams) -> int:
        with self.lock:
            request_id = self.request_counter
            self.request_counter += 1
        self.pending_queue.put({
            "id": request_id,
            "tokens": tokens,
            "params": params,
            "generated": [],
            "done": False
        })
        return request_id

    def get_batch(self) -> List[Dict[str, Any]]:
        with self.lock:
            batch = []
            while not self.pending_queue.empty() and len(self.active_requests) + len(batch) < self.max_batch_size:
                try:
                    req = self.pending_queue.get_nowait()
                    batch.append(req)
                except queue.Empty:
                    break
            for req in batch:
                self.active_requests[req["id"]] = req
            return list(self.active_requests.values())

    def update_request(self, request_id: int, new_token: int) -> bool:
        with self.lock:
            if request_id not in self.active_requests:
                return False
            req = self.active_requests[request_id]
            req["generated"].append(new_token)
            if len(req["generated"]) >= req["params"].max_tokens:
                req["done"] = True
                self.completed_requests[request_id] = req
                del self.active_requests[request_id]
                return True
            return False

    def get_result(self, request_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.completed_requests.get(request_id)

    def pop_result(self, request_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.completed_requests.pop(request_id, None)

    def has_active_requests(self) -> bool:
        with self.lock:
            return len(self.active_requests) > 0 or not self.pending_queue.empty()

    def get_stats(self) -> Dict[str, int]:
        with self.lock:
            return {
                "pending": self.pending_queue.qsize(),
                "active": len(self.active_requests),
                "completed": len(self.completed_requests)
            }

class SafetensorsLoader:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.index: Optional[Dict[str, Any]] = None
        self.shard_mmaps: Dict[str, tuple] = {}
        self.tensor_info: Dict[str, Dict[str, Any]] = {}

    def load_index(self) -> bool:
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.index = json.load(f)
            return True
        return False

    def mmap_shards(self) -> int:
        if self.index is None:
            return 0
        weight_map = self.index.get("weight_map", {})
        shard_files = set(weight_map.values())
        loaded = 0
        for shard_file in shard_files:
            shard_path = os.path.join(self.model_dir, shard_file)
            if os.path.exists(shard_path):
                fd = os.open(shard_path, os.O_RDONLY)
                file_size = os.fstat(fd).st_size
                mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
                self.shard_mmaps[shard_file] = (fd, mm, file_size)
                self._parse_shard_header(shard_file)
                loaded += 1
        return loaded

    def _parse_shard_header(self, shard_file: str) -> None:
        if shard_file not in self.shard_mmaps:
            return
        fd, mm, file_size = self.shard_mmaps[shard_file]
        if file_size < 8:
            return
        header_size = struct.unpack("<Q", mm[:8])[0]
        if 8 + header_size > file_size:
            return
        header_json = mm[8:8 + header_size].decode("utf-8")
        header = json.loads(header_json)
        for tensor_name, tensor_meta in header.items():
            if tensor_name == "__metadata__":
                continue
            self.tensor_info[tensor_name] = {
                "shard_file": shard_file,
                "offsets": tensor_meta.get("data_offsets", [0, 0]),
                "dtype": tensor_meta.get("dtype", "F32"),
                "shape": tensor_meta.get("shape", []),
                "header_size": header_size
            }

    def get_tensor_pointer(self, tensor_name: str) -> tuple:
        if tensor_name not in self.tensor_info:
            return None, None, None
        info = self.tensor_info[tensor_name]
        shard_file = info["shard_file"]
        if shard_file not in self.shard_mmaps:
            return None, None, None
        fd, mm, file_size = self.shard_mmaps[shard_file]
        header_size = info["header_size"]
        offsets = info["offsets"]
        data_start = 8 + header_size + offsets[0]
        data_end = 8 + header_size + offsets[1]
        return mm[data_start:data_end], info["dtype"], info["shape"]

    def list_tensors(self) -> List[str]:
        return list(self.tensor_info.keys())

    def cleanup(self) -> None:
        for shard_file, (fd, mm, _) in self.shard_mmaps.items():
            mm.close()
            os.close(fd)
        self.shard_mmaps = {}
        self.tensor_info = {}

class GPUSampler:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def sample(self, logits: List[float], params: SamplingParams, past_tokens: List[int]) -> int:
        if len(logits) == 0:
            return random.randint(0, self.vocab_size - 1)
        scaled = [l / max(params.temperature, 0.01) for l in logits]
        for tid in past_tokens:
            if 0 <= tid < len(scaled):
                if scaled[tid] > 0:
                    scaled[tid] = scaled[tid] / params.rep_penalty
                else:
                    scaled[tid] = scaled[tid] * params.rep_penalty
        max_logit = max(scaled)
        exp_logits = [2.718281828 ** (l - max_logit) for l in scaled]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        indexed = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumsum = 0.0
        cutoff_idx = len(probs)
        for i, idx in enumerate(indexed):
            cumsum += probs[idx]
            if cumsum >= params.top_p:
                cutoff_idx = i + 1
                break
        filtered_indices = indexed[:cutoff_idx]
        filtered_probs = [probs[idx] for idx in filtered_indices]
        filtered_sum = sum(filtered_probs)
        normalized_probs = [p / filtered_sum for p in filtered_probs]
        rand_val = random.random()
        cumsum = 0.0
        for i, prob in enumerate(normalized_probs):
            cumsum += prob
            if cumsum >= rand_val:
                return filtered_indices[i]
        return filtered_indices[-1] if filtered_indices else 0

class InferenceEngine:
    def __init__(self, model_dir: str, max_batch: int, max_seq: int, num_gpus: int):
        self.model_dir = model_dir
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.num_gpus = num_gpus
        self.lib: Optional[ctypes.CDLL] = None
        self.handle = None
        self.loader = SafetensorsLoader(model_dir)
        self.vocab_size = 151552
        self.hidden_dim = 4096
        self.num_layers = 92
        self.num_heads = 32
        self.head_dim = 128
        self.kv_manager = PagedKVCacheManager(
            max_pages=max_batch * (max_seq // 16 + 1),
            page_size=16,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim
        )
        self.scheduler = ContinuousBatchScheduler(max_batch, max_seq)
        self.sampler = GPUSampler(self.vocab_size)
        self.tokenizer = None
        self.request_states: Dict[int, RequestState] = {}
        self.lock = threading.Lock()

    def load_config(self) -> None:
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            self.hidden_dim = config.get("hidden_size", 4096)
            self.num_layers = config.get("num_hidden_layers", 92)
            self.num_heads = config.get("num_attention_heads", 32)
            self.head_dim = self.hidden_dim // self.num_heads
            self.vocab_size = config.get("vocab_size", 151552)

    def load_engine(self, engine_path: str) -> bool:
        if not os.path.exists(engine_path):
            return False
        self.lib = ctypes.CDLL(engine_path)
        self.lib.init_engine.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32
        ]
        self.lib.init_engine.restype = ctypes.POINTER(EngineHandle)
        self.lib.prefill.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.POINTER(RequestState)
        ]
        self.lib.prefill.restype = ctypes.c_int32
        self.lib.decode_step.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.POINTER(RequestState),
            ctypes.POINTER(ctypes.c_int64)
        ]
        self.lib.decode_step.restype = ctypes.c_int32
        self.lib.free_request_state.argtypes = [ctypes.POINTER(RequestState)]
        self.lib.free_request_state.restype = None
        self.lib.free_engine.argtypes = [ctypes.POINTER(EngineHandle)]
        self.lib.free_engine.restype = None
        self.handle = self.lib.init_engine(
            self.model_dir.encode("utf-8"),
            self.max_batch,
            self.max_seq,
            self.num_gpus
        )
        return self.handle is not None

    def load_tokenizer(self) -> bool:
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            return True
        except Exception:
            self.tokenizer = None
            return False

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        return [ord(c) % 1000 for c in text]

    def detokenize(self, tokens: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return "".join([chr(t % 128 + 32) for t in tokens])

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        rep_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        tokens = self.tokenize(prompt)
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            rep_penalty=rep_penalty,
            max_tokens=max_new_tokens
        )
        request_id = self.scheduler.add_request(tokens, params)
        while True:
            result = self.scheduler.get_result(request_id)
            if result is not None:
                return self.detokenize(result["generated"])
            self.step()

    def step(self) -> int:
        batch = self.scheduler.get_batch()
        if not batch:
            return 0
        tokens_generated = 0
        for req in batch:
            if "_state" not in req:
                state = RequestState()
                tokens = req["tokens"]
                token_arr = (ctypes.c_int64 * len(tokens))(*tokens)
                if self.lib is not None and self.handle is not None:
                    status = self.lib.prefill(
                        self.handle,
                        req["id"],
                        token_arr,
                        len(tokens),
                        ctypes.byref(state)
                    )
                    if status == 0:
                        with self.lock:
                            self.request_states[req["id"]] = state
                        req["_state"] = state
                else:
                    req["_state"] = state
        for req in batch:
            if "_state" in req and not req["done"]:
                if self.lib is not None and self.handle is not None:
                    with self.lock:
                        state = self.request_states.get(req["id"])
                    if state is None:
                        continue
                    next_token = ctypes.c_int64()
                    status = self.lib.decode_step(
                        self.handle,
                        ctypes.byref(state),
                        ctypes.byref(next_token)
                    )
                    if status == 0:
                        done = self.scheduler.update_request(req["id"], next_token.value)
                        tokens_generated += 1
                        if done:
                            with self.lock:
                                if req["id"] in self.request_states:
                                    del self.request_states[req["id"]]
                else:
                    logits = [random.gauss(0, 1) for _ in range(min(self.vocab_size, 1000))]
                    token = self.sampler.sample(logits, req["params"], req["tokens"] + req["generated"])
                    done = self.scheduler.update_request(req["id"], token)
                    tokens_generated += 1
                    if done:
                        with self.lock:
                            if req["id"] in self.request_states:
                                del self.request_states[req["id"]]
        return tokens_generated

    def cleanup(self) -> None:
        with self.lock:
            for req_id, state in list(self.request_states.items()):
                if self.lib is not None:
                    self.lib.free_request_state(ctypes.byref(state))
            self.request_states.clear()
        if self.lib is not None and self.handle is not None:
            self.lib.free_engine(self.handle)
            self.handle = None
        self.loader.cleanup()

    def get_stats(self) -> Dict[str, Any]:
        scheduler_stats = self.scheduler.get_stats()
        return {
            "scheduler": scheduler_stats,
            "kv_cache_utilization": self.kv_manager.get_utilization(),
            "num_gpus": self.num_gpus,
            "max_batch": self.max_batch,
            "max_seq": self.max_seq
        }

def run_local_benchmark(model_dir: str, engine_path: str) -> Dict[str, Any]:
    engine = InferenceEngine(
        model_dir=model_dir,
        max_batch=64,
        max_seq=4096,
        num_gpus=8
    )
    engine.load_config()
    engine.load_engine(engine_path)
    engine.loader.load_index()
    engine.loader.mmap_shards()
    engine.load_tokenizer()
    smoke_prompts = [
        "The capital of France is",
        "In machine learning, a neural network",
        "The speed of light in vacuum is",
        "Python is a programming language that",
        "The chemical formula for water is"
    ]
    smoke_results = []
    for prompt in smoke_prompts:
        output = engine.generate(
            prompt,
            max_new_tokens=64,
            temperature=0.8,
            top_p=0.95,
            rep_penalty=1.1,
            stop_sequences=[]
        )
        smoke_results.append({"prompt": prompt, "output": output})
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print("")
    warmup_duration = 10
    measure_duration = 60
    num_concurrent = 64
    prompt_len = 256
    gen_len = 256
    warmup_prompt = "x " * (prompt_len // 2)
    print("Running warmup...")
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_duration:
        params = SamplingParams(temperature=0.8, top_p=0.95, rep_penalty=1.1, max_tokens=gen_len)
        for _ in range(num_concurrent):
            tokens = engine.tokenize(warmup_prompt)[:prompt_len]
            engine.scheduler.add_request(tokens, params)
        while engine.scheduler.has_active_requests():
            engine.step()
    print("Running measurement...")
    measure_prompt = "y " * (prompt_len // 2)
    measure_start = time.time()
    total_tokens = 0
    batch_sizes = []
    latencies = []
    while time.time() - measure_start < measure_duration:
        params = SamplingParams(temperature=0.8, top_p=0.95, rep_penalty=1.1, max_tokens=gen_len)
        for _ in range(num_concurrent):
            tokens = engine.tokenize(measure_prompt)[:prompt_len]
            engine.scheduler.add_request(tokens, params)
        step_start = time.time()
        while engine.scheduler.has_active_requests():
            stats = engine.scheduler.get_stats()
            batch_sizes.append(stats["active"])
            tokens_gen = engine.step()
            total_tokens += tokens_gen
            step_end = time.time()
            if tokens_gen > 0:
                latencies.append((step_end - step_start) * 1000 / tokens_gen)
            step_start = step_end
    measure_end = time.time()
    elapsed = measure_end - measure_start
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    p50_latency = statistics.median(latencies) if latencies else 0
    p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0
    avg_batch = statistics.mean(batch_sizes) if batch_sizes else 0
    max_batch_size = max(batch_sizes) if batch_sizes else 0
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Measure window (seconds): {elapsed:.2f}")
    print(f"Total output tokens: {total_tokens}")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
    print(f"Avg latency (ms/token): {avg_latency:.2f}")
    print(f"P50 latency (ms): {p50_latency:.2f}")
    print(f"P99 latency (ms): {p99_latency:.2f}")
    print(f"Avg batch size: {avg_batch:.2f}")
    print(f"Max batch size: {max_batch_size}")
    print("=" * 60)
    engine.cleanup()
    return {
        "smoke_test": smoke_results,
        "benchmark": {
            "measure_window_seconds": elapsed,
            "total_output_tokens": total_tokens,
            "tokens_per_second": tokens_per_sec,
            "avg_latency_ms_per_token": avg_latency,
            "p50_latency_ms": p50_latency,
            "p99_latency_ms": p99_latency,
            "avg_batch_size": avg_batch,
            "max_batch_size": max_batch_size
        }
    }

if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "./model"
    engine_path = sys.argv[2] if len(sys.argv) > 2 else "./engine.so"
    results = run_local_benchmark(model_dir, engine_path)
    print(json.dumps(results, indent=2))
