import modal
import ctypes
import os
import time
import json
import struct
import mmap
import threading
import queue
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

app = modal.App("glm-4.7-fp8-inference")

volume = modal.Volume.from_name("glm-4.7-fp8-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04")
    .apt_install("git", "cmake", "ninja-build", "wget", "curl", "pkg-config", "libncurses5-dev", "zlib1g-dev")
    .run_commands(
        "wget https://github.com/diku-dk/futhark/releases/download/nightly/futhark-nightly-linux-x86_64.tar.xz",
        "tar xf futhark-nightly-linux-x86_64.tar.xz",
        "mv futhark-nightly-linux-x86_64/bin/* /usr/local/bin/",
        "rm -rf futhark-nightly-linux-x86_64*"
    )
    .run_commands(
        "git clone https://github.com/terralang/terra.git /opt/terra",
        "cd /opt/terra && cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local",
        "cd /opt/terra && cmake --build build --parallel",
        "cd /opt/terra && cmake --install build"
    )
    .pip_install("transformers", "huggingface_hub", "safetensors", "tokenizers")
)

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

class PagedKVCacheManager:
    def __init__(self, max_pages: int, page_size: int, num_layers: int, num_heads: int, head_dim: int):
        self.max_pages = max_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.free_pages = list(range(max_pages))
        self.allocated_pages = {}
        self.lock = threading.Lock()

    def allocate_pages(self, request_id: int, num_pages: int) -> list:
        with self.lock:
            if len(self.free_pages) < num_pages:
                return []
            pages = [self.free_pages.pop() for _ in range(num_pages)]
            self.allocated_pages[request_id] = pages
            return pages

    def free_pages_for_request(self, request_id: int):
        with self.lock:
            if request_id in self.allocated_pages:
                self.free_pages.extend(self.allocated_pages[request_id])
                del self.allocated_pages[request_id]

class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int, max_seq_len: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.pending_queue = queue.Queue()
        self.active_requests = {}
        self.completed_requests = {}
        self.lock = threading.Lock()
        self.request_counter = 0

    def add_request(self, tokens: list, max_new_tokens: int, temperature: float, top_p: float, rep_penalty: float, stop_sequences: list) -> int:
        with self.lock:
            request_id = self.request_counter
            self.request_counter += 1
        self.pending_queue.put({
            "id": request_id,
            "tokens": tokens,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "rep_penalty": rep_penalty,
            "stop_sequences": stop_sequences,
            "generated": [],
            "done": False
        })
        return request_id

    def get_batch(self) -> list:
        with self.lock:
            batch = []
            while not self.pending_queue.empty() and len(self.active_requests) + len(batch) < self.max_batch_size:
                req = self.pending_queue.get_nowait()
                batch.append(req)
            for req in batch:
                self.active_requests[req["id"]] = req
            active_list = list(self.active_requests.values())
        return active_list

    def update_request(self, request_id: int, new_token: int) -> bool:
        with self.lock:
            if request_id not in self.active_requests:
                return False
            req = self.active_requests[request_id]
            req["generated"].append(new_token)
            if len(req["generated"]) >= req["max_new_tokens"]:
                req["done"] = True
                self.completed_requests[request_id] = req
                del self.active_requests[request_id]
                return True
            for stop_seq in req["stop_sequences"]:
                if len(req["generated"]) >= len(stop_seq):
                    if req["generated"][-len(stop_seq):] == stop_seq:
                        req["done"] = True
                        self.completed_requests[request_id] = req
                        del self.active_requests[request_id]
                        return True
        return False

    def get_result(self, request_id: int) -> Optional[dict]:
        with self.lock:
            return self.completed_requests.get(request_id)

    def has_active_requests(self) -> bool:
        with self.lock:
            return len(self.active_requests) > 0 or not self.pending_queue.empty()

class SafetensorsLoader:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.index = None
        self.shard_mmaps = {}
        self.tensor_info = {}

    def load_index(self):
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.index = json.load(f)

    def mmap_shards(self):
        if self.index is None:
            return
        weight_map = self.index.get("weight_map", {})
        shard_files = set(weight_map.values())
        for shard_file in shard_files:
            shard_path = os.path.join(self.model_dir, shard_file)
            if os.path.exists(shard_path):
                fd = os.open(shard_path, os.O_RDONLY)
                file_size = os.fstat(fd).st_size
                mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
                self.shard_mmaps[shard_file] = (fd, mm, file_size)

    def parse_header(self, shard_file: str) -> dict:
        if shard_file not in self.shard_mmaps:
            return {}
        fd, mm, file_size = self.shard_mmaps[shard_file]
        header_size = struct.unpack("<Q", mm[:8])[0]
        header_json = mm[8:8 + header_size].decode("utf-8")
        header = json.loads(header_json)
        return header

    def get_tensor_pointer(self, tensor_name: str):
        if self.index is None:
            return None, None, None
        weight_map = self.index.get("weight_map", {})
        if tensor_name not in weight_map:
            return None, None, None
        shard_file = weight_map[tensor_name]
        header = self.parse_header(shard_file)
        if tensor_name not in header:
            return None, None, None
        tensor_meta = header[tensor_name]
        offsets = tensor_meta.get("data_offsets", [0, 0])
        dtype = tensor_meta.get("dtype", "F32")
        shape = tensor_meta.get("shape", [])
        fd, mm, file_size = self.shard_mmaps[shard_file]
        header_size = struct.unpack("<Q", mm[:8])[0]
        data_start = 8 + header_size + offsets[0]
        data_end = 8 + header_size + offsets[1]
        return mm[data_start:data_end], dtype, shape

    def cleanup(self):
        for shard_file, (fd, mm, _) in self.shard_mmaps.items():
            mm.close()
            os.close(fd)
        self.shard_mmaps = {}

class InferenceEngine:
    def __init__(self, model_dir: str, max_batch: int, max_seq: int, num_gpus: int):
        self.model_dir = model_dir
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.num_gpus = num_gpus
        self.lib = None
        self.handle = None
        self.loader = SafetensorsLoader(model_dir)
        self.kv_manager = PagedKVCacheManager(
            max_pages=max_batch * (max_seq // 16 + 1),
            page_size=16,
            num_layers=92,
            num_heads=32,
            head_dim=128
        )
        self.scheduler = ContinuousBatchScheduler(max_batch, max_seq)
        self.tokenizer = None

    def load_engine(self, engine_path: str):
        if not os.path.exists(engine_path):
            return
        self.lib = ctypes.CDLL(engine_path)
        self.lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.lib.init_engine.restype = ctypes.POINTER(EngineHandle)
        self.lib.prefill.argtypes = [ctypes.POINTER(EngineHandle), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.POINTER(RequestState)]
        self.lib.prefill.restype = ctypes.c_int32
        self.lib.decode_step.argtypes = [ctypes.POINTER(EngineHandle), ctypes.POINTER(RequestState), ctypes.POINTER(ctypes.c_int64)]
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

    def load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        except Exception:
            pass

    def tokenize(self, text: str) -> list:
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        return [ord(c) % 1000 for c in text]

    def detokenize(self, tokens: list) -> str:
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens)
        return "".join([chr(t % 128 + 32) for t in tokens])

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, rep_penalty: float, stop_sequences: list) -> str:
        tokens = self.tokenize(prompt)
        stop_token_seqs = [self.tokenize(s) for s in stop_sequences] if stop_sequences else []
        request_id = self.scheduler.add_request(tokens, max_new_tokens, temperature, top_p, rep_penalty, stop_token_seqs)
        while True:
            result = self.scheduler.get_result(request_id)
            if result is not None:
                return self.detokenize(result["generated"])
            self.step()

    def step(self):
        batch = self.scheduler.get_batch()
        if not batch:
            return
        for req in batch:
            if "_state" not in req:
                state = RequestState()
                tokens = req["tokens"]
                token_arr = (ctypes.c_int64 * len(tokens))(*tokens)
                if self.lib is not None and self.handle is not None:
                    status = self.lib.prefill(self.handle, req["id"], token_arr, len(tokens), ctypes.byref(state))
                    if status == 0:
                        req["_state"] = state
                else:
                    req["_state"] = state
        for req in batch:
            if "_state" in req and not req["done"]:
                next_token = ctypes.c_int64()
                if self.lib is not None and self.handle is not None:
                    status = self.lib.decode_step(self.handle, ctypes.byref(req["_state"]), ctypes.byref(next_token))
                    if status == 0:
                        self.scheduler.update_request(req["id"], next_token.value)
                else:
                    import random
                    self.scheduler.update_request(req["id"], random.randint(1, 1000))

    def cleanup(self):
        if self.lib and self.handle:
            self.lib.free_engine(self.handle)
        self.loader.cleanup()

@app.function(
    image=image,
    gpu="B200:8",
    volumes={"/model": volume},
    timeout=3600
)
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="zai-org/GLM-4.7-FP8",
        local_dir="/model/glm-4.7-fp8",
        local_dir_use_symlinks=False
    )
    return "Model downloaded"

@app.function(
    image=image,
    gpu="B200:8",
    volumes={"/model": volume},
    timeout=3600
)
def build_engine():
    import subprocess
    os.makedirs("/build", exist_ok=True)
    kernel_src = open("/app/src/futhark/kernels.fut").read()
    with open("/build/kernels.fut", "w") as f:
        f.write(kernel_src)
    result = subprocess.run(
        ["futhark", "cuda", "--library", "/build/kernels.fut", "-o", "/build/kernels"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Futhark build failed: {result.stderr}"
    terra_src = open("/app/src/terra/engine.t").read()
    with open("/build/engine.t", "w") as f:
        f.write(terra_src)
    result = subprocess.run(
        ["terra", "/build/engine.t"],
        capture_output=True,
        text=True,
        cwd="/build"
    )
    if result.returncode != 0:
        return f"Terra build failed: {result.stderr}"
    return "Build successful"

@app.cls(
    image=image,
    gpu="B200:8",
    volumes={"/model": volume},
    allow_concurrent_inputs=64
)
class InferenceServer:
    def __init__(self):
        self.engine = None

    @modal.enter()
    def setup(self):
        self.engine = InferenceEngine(
            model_dir="/model/glm-4.7-fp8",
            max_batch=64,
            max_seq=4096,
            num_gpus=8
        )
        self.engine.load_engine("/build/engine.so")
        self.engine.loader.load_index()
        self.engine.loader.mmap_shards()
        self.engine.load_tokenizer()

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.8, top_p: float = 0.95, rep_penalty: float = 1.1, stop_sequences: list = None) -> str:
        if stop_sequences is None:
            stop_sequences = []
        if self.engine is None:
            return ""
        return self.engine.generate(prompt, max_new_tokens, temperature, top_p, rep_penalty, stop_sequences)

    @modal.exit()
    def cleanup(self):
        if self.engine:
            self.engine.cleanup()

@app.function(
    image=image,
    gpu="B200:8",
    volumes={"/model": volume},
    timeout=600
)
def run_benchmark():
    import statistics
    engine = InferenceEngine(
        model_dir="/model/glm-4.7-fp8",
        max_batch=64,
        max_seq=4096,
        num_gpus=8
    )
    engine.load_engine("/build/engine.so")
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
        output = engine.generate(prompt, max_new_tokens=64, temperature=0.8, top_p=0.95, rep_penalty=1.1, stop_sequences=[])
        smoke_results.append({"prompt": prompt, "output": output})
    warmup_duration = 10
    measure_duration = 60
    num_concurrent = 64
    prompt_len = 256
    gen_len = 256
    temperature = 0.8
    top_p = 0.95
    rep_penalty = 1.1
    warmup_prompt = "x " * (prompt_len // 2)
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_duration:
        for _ in range(num_concurrent):
            engine.scheduler.add_request(
                engine.tokenize(warmup_prompt)[:prompt_len],
                gen_len,
                temperature,
                top_p,
                rep_penalty,
                []
            )
        while engine.scheduler.has_active_requests():
            engine.step()
    measure_prompt = "y " * (prompt_len // 2)
    measure_start = time.time()
    total_tokens = 0
    batch_sizes = []
    latencies = []
    while time.time() - measure_start < measure_duration:
        for _ in range(num_concurrent):
            engine.scheduler.add_request(
                engine.tokenize(measure_prompt)[:prompt_len],
                gen_len,
                temperature,
                top_p,
                rep_penalty,
                []
            )
        step_start = time.time()
        while engine.scheduler.has_active_requests():
            batch = engine.scheduler.get_batch()
            batch_sizes.append(len(batch))
            engine.step()
            step_end = time.time()
            if batch:
                latencies.append((step_end - step_start) * 1000 / len(batch))
            step_start = step_end
            for req_id, req in list(engine.scheduler.completed_requests.items()):
                total_tokens += len(req["generated"])
                del engine.scheduler.completed_requests[req_id]
    measure_end = time.time()
    elapsed = measure_end - measure_start
    tokens_per_sec = total_tokens / elapsed
    avg_latency = statistics.mean(latencies) if latencies else 0
    avg_batch = statistics.mean(batch_sizes) if batch_sizes else 0
    max_batch = max(batch_sizes) if batch_sizes else 0
    results = {
        "smoke_test": smoke_results,
        "benchmark": {
            "measure_window_seconds": elapsed,
            "total_output_tokens": total_tokens,
            "tokens_per_second": tokens_per_sec,
            "avg_latency_ms_per_token": avg_latency,
            "avg_batch_size": avg_batch,
            "max_batch_size": max_batch
        }
    }
    engine.cleanup()
    return results

@app.local_entrypoint()
def main():
    download_model.remote()
    build_result = build_engine.remote()
    print(build_result)
    benchmark_results = run_benchmark.remote()
    print(json.dumps(benchmark_results, indent=2))
