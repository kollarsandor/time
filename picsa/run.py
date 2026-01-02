#!/usr/bin/env python3
import sys
import os
import time
import json
import argparse
import ctypes
import struct
import mmap
import threading
import queue
import statistics
from typing import Optional, List, Dict, Any, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler

class EngineHandle(ctypes.Structure):
    pass

class RequestStateC(ctypes.Structure):
    _fields_ = [
        ("request_id", ctypes.c_int64),
        ("seq_len", ctypes.c_int32),
        ("page_indices", ctypes.POINTER(ctypes.c_int32)),
        ("num_pages", ctypes.c_int32),
        ("past_tokens", ctypes.POINTER(ctypes.c_int64)),
        ("past_len", ctypes.c_int32),
        ("max_gen_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("rep_penalty", ctypes.c_float),
        ("stop_tokens", ctypes.POINTER(ctypes.c_int64)),
        ("num_stop_tokens", ctypes.c_int32),
        ("is_prefill_done", ctypes.c_int32),
        ("is_finished", ctypes.c_int32),
    ]

class SamplingParams:
    def __init__(
        self,
        temperature: float = 0.8,
        top_p: float = 0.95,
        rep_penalty: float = 1.1,
        max_tokens: int = 256,
        stop_token_ids: Optional[List[int]] = None
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.rep_penalty = rep_penalty
        self.max_tokens = max_tokens
        self.stop_token_ids = stop_token_ids if stop_token_ids else []

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

    def extend_pages(self, request_id: int, num_additional: int) -> List[int]:
        with self.lock:
            if len(self.free_pages) < num_additional:
                return []
            pages = [self.free_pages.pop() for _ in range(num_additional)]
            if request_id in self.allocated_pages:
                self.allocated_pages[request_id].extend(pages)
            else:
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

    def add_request(
        self,
        tokens: List[int],
        params: SamplingParams,
        state_ptr: Optional[ctypes.c_void_p] = None
    ) -> int:
        with self.lock:
            request_id = self.request_counter
            self.request_counter += 1
        self.pending_queue.put({
            "id": request_id,
            "tokens": tokens,
            "params": params,
            "generated": [],
            "done": False,
            "prefill_done": False,
            "state_ptr": state_ptr
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
            if new_token in req["params"].stop_token_ids:
                req["done"] = True
                self.completed_requests[request_id] = req
                del self.active_requests[request_id]
                return True
            return False

    def mark_prefill_done(self, request_id: int, state_ptr: ctypes.c_void_p) -> None:
        with self.lock:
            if request_id in self.active_requests:
                self.active_requests[request_id]["prefill_done"] = True
                self.active_requests[request_id]["state_ptr"] = state_ptr

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

class InferenceEngine:
    def __init__(self, model_dir: str, engine_path: str, max_batch: int, max_seq: int, num_gpus: int):
        self.model_dir = model_dir
        self.engine_path = engine_path
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.num_gpus = num_gpus
        self.lib: Optional[ctypes.CDLL] = None
        self.handle: Any = None
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
        self.tokenizer = None
        self.eos_token_id: Optional[int] = None
        self.request_state_ptrs: Dict[int, ctypes.c_void_p] = {}
        self.lock = threading.Lock()

    def load_config(self) -> bool:
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            return False
        with open(config_path, "r") as f:
            config = json.load(f)
        self.hidden_dim = config.get("hidden_size", 4096)
        self.num_layers = config.get("num_hidden_layers", 92)
        self.num_heads = config.get("num_attention_heads", 32)
        self.head_dim = self.hidden_dim // self.num_heads
        self.vocab_size = config.get("vocab_size", 151552)
        return True

    def load_engine(self) -> bool:
        if not os.path.exists(self.engine_path):
            print(f"Engine not found: {self.engine_path}")
            return False
        try:
            self.lib = ctypes.CDLL(self.engine_path)
        except OSError as e:
            print(f"Failed to load engine: {e}")
            return False
        self.lib.init_engine.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32
        ]
        self.lib.init_engine.restype = ctypes.POINTER(EngineHandle)
        self.lib.prefill_opaque.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32
        ]
        self.lib.prefill_opaque.restype = ctypes.c_void_p
        self.lib.decode_step_opaque.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32)
        ]
        self.lib.decode_step_opaque.restype = ctypes.c_int32
        self.lib.free_request_opaque.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.c_void_p
        ]
        self.lib.free_request_opaque.restype = None
        self.lib.run_batch_decode_ext.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32)
        ]
        self.lib.run_batch_decode_ext.restype = ctypes.c_int32
        self.lib.free_engine.argtypes = [ctypes.POINTER(EngineHandle)]
        self.lib.free_engine.restype = None
        self.lib.get_engine_info.argtypes = [ctypes.POINTER(EngineHandle), ctypes.c_int32]
        self.lib.get_engine_info.restype = ctypes.c_int64
        self.handle = self.lib.init_engine(
            self.model_dir.encode("utf-8"),
            self.max_batch,
            self.max_seq,
            self.num_gpus
        )
        if self.handle is None:
            print("Failed to initialize engine")
            return False
        return True

    def load_tokenizer(self) -> bool:
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                self.eos_token_id = self.tokenizer.eos_token_id
            return True
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            return False

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.encode(text)

    def detokenize(self, tokens: List[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

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
        stop_token_ids = [self.eos_token_id] if self.eos_token_id is not None else []
        if stop_sequences:
            for seq in stop_sequences:
                seq_tokens = self.tokenize(seq)
                if seq_tokens:
                    stop_token_ids.extend(seq_tokens)
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            rep_penalty=rep_penalty,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids
        )
        request_id = self.scheduler.add_request(tokens, params)
        while True:
            result = self.scheduler.get_result(request_id)
            if result is not None:
                return self.detokenize(result["generated"])
            self.step()
            time.sleep(0.0001)

    def step(self) -> int:
        batch = self.scheduler.get_batch()
        if not batch:
            return 0
        tokens_generated = 0
        prefill_requests = [req for req in batch if not req["prefill_done"]]
        failed_prefills = []
        for req in prefill_requests:
            tokens = req["tokens"]
            token_arr = (ctypes.c_int64 * len(tokens))(*tokens)
            params = req["params"]
            state_ptr = self.lib.prefill_opaque(
                self.handle,
                req["id"],
                token_arr,
                len(tokens),
                ctypes.c_float(params.temperature),
                ctypes.c_float(params.top_p),
                ctypes.c_float(params.rep_penalty),
                params.max_tokens
            )
            if state_ptr:
                with self.lock:
                    self.request_state_ptrs[req["id"]] = state_ptr
                self.scheduler.mark_prefill_done(req["id"], state_ptr)
            else:
                failed_prefills.append(req["id"])
        for req_id in failed_prefills:
            self.scheduler.update_request(req_id, 0)
            self.kv_manager.free_pages_for_request(req_id)
        decode_requests = [req for req in batch if req["prefill_done"] and not req["done"]]
        if decode_requests:
            num_decode = len(decode_requests)
            state_ptrs = (ctypes.c_void_p * num_decode)()
            out_tokens = (ctypes.c_int64 * num_decode)()
            out_done = (ctypes.c_int32 * num_decode)()
            for i, req in enumerate(decode_requests):
                with self.lock:
                    state_ptrs[i] = self.request_state_ptrs.get(req["id"], None)
            status = self.lib.run_batch_decode_ext(
                self.handle,
                state_ptrs,
                num_decode,
                out_tokens,
                out_done
            )
            if status == 0:
                for i, req in enumerate(decode_requests):
                    new_token = out_tokens[i]
                    is_done = out_done[i] != 0
                    done = self.scheduler.update_request(req["id"], new_token)
                    tokens_generated += 1
                    if done or is_done:
                        with self.lock:
                            state_ptr = self.request_state_ptrs.pop(req["id"], None)
                        if state_ptr:
                            self.lib.free_request_opaque(self.handle, state_ptr)
                        self.kv_manager.free_pages_for_request(req["id"])
        return tokens_generated

    def cleanup(self) -> None:
        with self.lock:
            for req_id, state_ptr in list(self.request_state_ptrs.items()):
                if state_ptr and self.lib and self.handle:
                    self.lib.free_request_opaque(self.handle, state_ptr)
            self.request_state_ptrs.clear()
        if self.lib is not None and self.handle is not None:
            self.lib.free_engine(self.handle)
            self.handle = None

    def get_stats(self) -> Dict[str, Any]:
        scheduler_stats = self.scheduler.get_stats()
        return {
            "scheduler": scheduler_stats,
            "kv_cache_utilization": self.kv_manager.get_utilization(),
            "num_gpus": self.num_gpus,
            "max_batch": self.max_batch,
            "max_seq": self.max_seq,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size
        }

class APIHandler(BaseHTTPRequestHandler):
    engine: Optional[InferenceEngine] = None

    def log_message(self, format: str, *args) -> None:
        pass

    def do_POST(self) -> None:
        if self.path == "/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                request = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 256)
            temperature = request.get("temperature", 0.8)
            top_p = request.get("top_p", 0.95)
            rep_penalty = request.get("rep_penalty", 1.1)
            stop_sequences = request.get("stop_sequences", [])
            if not prompt:
                self.send_error(400, "Missing prompt")
                return
            if self.engine is None:
                self.send_error(500, "Engine not initialized")
                return
            try:
                output = self.engine.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    rep_penalty=rep_penalty,
                    stop_sequences=stop_sequences
                )
                response = {"output": output, "prompt": prompt}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
            except Exception as e:
                self.send_error(500, str(e))
        elif self.path == "/stats":
            if self.engine is None:
                self.send_error(500, "Engine not initialized")
                return
            stats = self.engine.get_stats()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
        else:
            self.send_error(404, "Not found")

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        elif self.path == "/stats":
            if self.engine is None:
                self.send_error(500, "Engine not initialized")
                return
            stats = self.engine.get_stats()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode("utf-8"))
        else:
            self.send_error(404, "Not found")

def run_smoke_test(engine: InferenceEngine) -> List[Dict[str, str]]:
    prompts = [
        "The capital of France is",
        "In machine learning, a neural network",
        "The speed of light in vacuum is",
        "Python is a programming language that",
        "The chemical formula for water is"
    ]
    results = []
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    for prompt in prompts:
        try:
            start = time.time()
            output = engine.generate(
                prompt=prompt,
                max_new_tokens=64,
                temperature=0.8,
                top_p=0.95,
                rep_penalty=1.1
            )
            elapsed = time.time() - start
            results.append({"prompt": prompt, "output": output, "time_s": elapsed})
            print(f"Prompt: {prompt}")
            print(f"Output: {output}")
            print(f"Time: {elapsed:.2f}s")
            print("-" * 40)
        except Exception as e:
            results.append({"prompt": prompt, "error": str(e)})
            print(f"Prompt: {prompt}")
            print(f"Error: {e}")
            print("-" * 40)
    return results

def run_benchmark(
    engine: InferenceEngine,
    duration: int,
    concurrency: int,
    prompt_len: int,
    gen_len: int
) -> Dict[str, Any]:
    print("=" * 60)
    print("BENCHMARK")
    print("=" * 60)
    print(f"Duration: {duration}s")
    print(f"Concurrency: {concurrency}")
    print(f"Prompt length: {prompt_len} tokens")
    print(f"Generation length: {gen_len} tokens")
    print("-" * 40)
    warmup_duration = min(10, duration // 6)
    warmup_prompt = "This is a warmup prompt for the inference engine. " * (prompt_len // 10 + 1)
    warmup_tokens = engine.tokenize(warmup_prompt)[:prompt_len]
    print(f"Running warmup ({warmup_duration}s)...")
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_duration:
        params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            rep_penalty=1.1,
            max_tokens=gen_len,
            stop_token_ids=[engine.eos_token_id] if engine.eos_token_id else []
        )
        for _ in range(min(concurrency, 8)):
            engine.scheduler.add_request(warmup_tokens, params)
        while engine.scheduler.has_active_requests():
            engine.step()
    print("Warmup complete")
    print("-" * 40)
    measure_prompt = "This is the measurement prompt for benchmarking performance. " * (prompt_len // 10 + 1)
    measure_tokens = engine.tokenize(measure_prompt)[:prompt_len]
    print(f"Running measurement ({duration}s)...")
    measure_start = time.time()
    total_tokens = 0
    total_requests = 0
    batch_sizes = []
    step_latencies = []
    while time.time() - measure_start < duration:
        params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            rep_penalty=1.1,
            max_tokens=gen_len,
            stop_token_ids=[engine.eos_token_id] if engine.eos_token_id else []
        )
        for _ in range(concurrency):
            engine.scheduler.add_request(measure_tokens, params)
            total_requests += 1
        while engine.scheduler.has_active_requests():
            stats = engine.scheduler.get_stats()
            batch_sizes.append(stats["active"])
            step_start = time.time()
            tokens_gen = engine.step()
            step_end = time.time()
            total_tokens += tokens_gen
            if tokens_gen > 0:
                step_latencies.append((step_end - step_start) * 1000 / tokens_gen)
    measure_end = time.time()
    elapsed = measure_end - measure_start
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    avg_latency = statistics.mean(step_latencies) if step_latencies else 0
    p50_latency = statistics.median(step_latencies) if step_latencies else 0
    sorted_latencies = sorted(step_latencies) if step_latencies else [0]
    p99_idx = min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)
    p99_latency = sorted_latencies[p99_idx]
    avg_batch = statistics.mean(batch_sizes) if batch_sizes else 0
    max_batch_size = max(batch_sizes) if batch_sizes else 0
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Measurement window: {elapsed:.2f}s")
    print(f"Total output tokens: {total_tokens}")
    print(f"Total requests: {total_requests}")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
    print(f"Avg latency (ms/token): {avg_latency:.3f}")
    print(f"P50 latency (ms/token): {p50_latency:.3f}")
    print(f"P99 latency (ms/token): {p99_latency:.3f}")
    print(f"Avg batch size: {avg_batch:.2f}")
    print(f"Max batch size: {max_batch_size}")
    print("=" * 60)
    target_met = tokens_per_sec >= 3000
    print(f"Target (3000 tok/s): {'MET' if target_met else 'NOT MET'}")
    return {
        "measure_window_seconds": elapsed,
        "total_output_tokens": total_tokens,
        "total_requests": total_requests,
        "tokens_per_second": tokens_per_sec,
        "avg_latency_ms_per_token": avg_latency,
        "p50_latency_ms": p50_latency,
        "p99_latency_ms": p99_latency,
        "avg_batch_size": avg_batch,
        "max_batch_size": max_batch_size,
        "target_met": target_met
    }

def run_serve(engine: InferenceEngine, host: str, port: int) -> None:
    APIHandler.engine = engine
    server = HTTPServer((host, port), APIHandler)
    print(f"Serving on http://{host}:{port}")
    print("Endpoints:")
    print("  POST /generate - Generate text")
    print("  GET  /health   - Health check")
    print("  GET  /stats    - Engine statistics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-4.7-FP8 Inference Engine")
    parser.add_argument("--model", type=str, default="./model", help="Model directory")
    parser.add_argument("--engine", type=str, default="./build/engine.so", help="Engine shared library path")
    parser.add_argument("--max-batch", type=int, default=64, help="Maximum batch size")
    parser.add_argument("--max-seq", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    smoke_parser = subparsers.add_parser("smoke", help="Run smoke test")
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds")
    bench_parser.add_argument("--concurrency", type=int, default=64, help="Number of concurrent requests")
    bench_parser.add_argument("--prompt-len", type=int, default=256, help="Prompt length in tokens")
    bench_parser.add_argument("--gen-len", type=int, default=256, help="Generation length in tokens")
    serve_parser = subparsers.add_parser("serve", help="Run HTTP server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        print("\nExample usage:")
        print("  python run.py --model ./model --engine ./build/engine.so smoke")
        print("  python run.py --model ./model --engine ./build/engine.so bench --duration 60 --concurrency 64")
        print("  python run.py --model ./model --engine ./build/engine.so serve --port 8080")
        return
    print("=" * 60)
    print("GLM-4.7-FP8 Inference Engine")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Engine: {args.engine}")
    print(f"Max batch: {args.max_batch}")
    print(f"Max seq: {args.max_seq}")
    print(f"GPUs: {args.num_gpus}")
    print("-" * 60)
    engine = InferenceEngine(
        model_dir=args.model,
        engine_path=args.engine,
        max_batch=args.max_batch,
        max_seq=args.max_seq,
        num_gpus=args.num_gpus
    )
    print("Loading config...")
    if not engine.load_config():
        print("Warning: Could not load config.json, using defaults")
    print("Loading tokenizer...")
    if not engine.load_tokenizer():
        print("Error: Could not load tokenizer")
        return
    print("Loading engine...")
    if not engine.load_engine():
        print("Error: Could not load engine")
        return
    print("Engine loaded successfully")
    print("-" * 60)
    try:
        if args.command == "smoke":
            results = run_smoke_test(engine)
            print("\nSmoke test results:")
            print(json.dumps(results, indent=2))
        elif args.command == "bench":
            results = run_benchmark(
                engine=engine,
                duration=args.duration,
                concurrency=args.concurrency,
                prompt_len=args.prompt_len,
                gen_len=args.gen_len
            )
            print("\nBenchmark results:")
            print(json.dumps(results, indent=2))
        elif args.command == "serve":
            run_serve(engine, args.host, args.port)
    finally:
        print("Cleaning up...")
        engine.cleanup()
        print("Done")

if __name__ == "__main__":
    main()
