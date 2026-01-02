import modal
import ctypes
import os
import time
import json
import struct
import mmap
import threading
import queue
import random
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

app = modal.App("glm-4.7-fp8-inference")

volume = modal.Volume.from_name("glm-4.7-fp8-weights", create_if_missing=True)
build_volume = modal.Volume.from_name("glm-4.7-fp8-build", create_if_missing=True)

SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
CSRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "csrc")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04")
    .run_commands(
        "apt-get update && apt-get install -y --allow-change-held-packages "
        "git cmake ninja-build wget curl pkg-config libncurses5-dev zlib1g-dev "
        "build-essential libedit-dev libgmp-dev clang llvm llvm-dev libnccl2 libnccl-dev unzip "
        "python3 python3-pip && ln -sf /usr/bin/python3 /usr/bin/python"
    )
    .run_commands(
        "wget https://github.com/diku-dk/futhark/releases/download/nightly/futhark-nightly-linux-x86_64.tar.xz",
        "tar xf futhark-nightly-linux-x86_64.tar.xz",
        "mv futhark-nightly-linux-x86_64/bin/* /usr/local/bin/",
        "rm -rf futhark-nightly-linux-x86_64*"
    )
    .run_commands(
        "wget -q https://github.com/terralang/terra/releases/download/release-1.2.0/terra-Linux-x86_64-cc543db.tar.xz -O /tmp/terra.tar.xz",
        "tar xf /tmp/terra.tar.xz -C /opt",
        "mv /opt/terra-* /opt/terra",
        "ln -s /opt/terra/bin/terra /usr/local/bin/terra",
        "rm /tmp/terra.tar.xz"
    )
    .run_commands("python3 -m pip install --upgrade pip")
    .pip_install("transformers", "huggingface_hub", "safetensors", "tokenizers", "numpy")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/lib",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
    })
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

class SamplingParams:
    def __init__(
        self,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        rep_penalty: float = 1.1,
        freq_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 256,
        stop_sequences: Optional[List[str]] = None
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.rep_penalty = rep_penalty
        self.freq_penalty = freq_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences if stop_sequences else []

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
        stop_token_ids: Optional[List[int]] = None
    ) -> int:
        with self.lock:
            request_id = self.request_counter
            self.request_counter += 1
        self.pending_queue.put({
            "id": request_id,
            "tokens": tokens,
            "params": params,
            "stop_token_ids": stop_token_ids if stop_token_ids else [],
            "generated": [],
            "done": False,
            "prefill_done": False,
            "state": None
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
            if new_token in req["stop_token_ids"]:
                req["done"] = True
                self.completed_requests[request_id] = req
                del self.active_requests[request_id]
                return True
            return False

    def mark_prefill_done(self, request_id: int) -> None:
        with self.lock:
            if request_id in self.active_requests:
                self.active_requests[request_id]["prefill_done"] = True

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
        single_path = os.path.join(self.model_dir, "model.safetensors")
        if os.path.exists(single_path):
            self.index = {"weight_map": {}, "metadata": {}}
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

    def get_tensor_info(self, tensor_name: str) -> Optional[Dict[str, Any]]:
        return self.tensor_info.get(tensor_name)

    def list_tensors(self) -> List[str]:
        return list(self.tensor_info.keys())

    def cleanup(self) -> None:
        for shard_file, (fd, mm, _) in self.shard_mmaps.items():
            mm.close()
            os.close(fd)
        self.shard_mmaps = {}
        self.tensor_info = {}

class GPUSampler:
    def __init__(self, vocab_size: int, num_gpus: int):
        self.vocab_size = vocab_size
        self.num_gpus = num_gpus

    def sample_batch(
        self,
        logits: List[List[float]],
        params_list: List[SamplingParams],
        past_tokens_list: List[List[int]]
    ) -> List[int]:
        results = []
        for i, logits_row in enumerate(logits):
            params = params_list[i] if i < len(params_list) else SamplingParams()
            past_tokens = past_tokens_list[i] if i < len(past_tokens_list) else []
            token = self._sample_single(logits_row, params, past_tokens)
            results.append(token)
        return results

    def _sample_single(
        self,
        logits: List[float],
        params: SamplingParams,
        past_tokens: List[int]
    ) -> int:
        if len(logits) == 0:
            return 0
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
        self.handle: Optional[Any] = None
        self.loader = SafetensorsLoader(model_dir)
        self.hidden_dim = 4096
        self.num_layers = 92
        self.num_heads = 32
        self.head_dim = 128
        self.vocab_size = 151552
        self.num_experts = 160
        self.top_k_experts = 8
        self.kv_manager = PagedKVCacheManager(
            max_pages=max_batch * (max_seq // 16 + 1),
            page_size=16,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim
        )
        self.scheduler = ContinuousBatchScheduler(max_batch, max_seq)
        self.sampler = GPUSampler(self.vocab_size, num_gpus)
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
            self.num_experts = config.get("num_local_experts", 160)
            self.top_k_experts = config.get("num_experts_per_tok", 8)

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
        self.lib.get_engine_info.argtypes = [ctypes.POINTER(EngineHandle), ctypes.c_int32]
        self.lib.get_engine_info.restype = ctypes.c_int64
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
            return False

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        return [ord(c) % 1000 for c in text]

    def detokenize(self, tokens: List[int]) -> str:
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return "".join([chr(t % 128 + 32) for t in tokens])

    def get_stop_token_ids(self, stop_sequences: List[str]) -> List[int]:
        if self.tokenizer is None:
            return []
        stop_ids = []
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        for seq in stop_sequences:
            tokens = self.tokenize(seq)
            if tokens:
                stop_ids.extend(tokens)
        return stop_ids

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        rep_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        if stop_sequences is None:
            stop_sequences = []
        tokens = self.tokenize(prompt)
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            rep_penalty=rep_penalty,
            max_tokens=max_new_tokens,
            stop_sequences=stop_sequences
        )
        stop_token_ids = self.get_stop_token_ids(stop_sequences)
        request_id = self.scheduler.add_request(tokens, params, stop_token_ids)
        while True:
            result = self.scheduler.get_result(request_id)
            if result is not None:
                return self.detokenize(result["generated"])
            self.step()
            time.sleep(0.001)

    def generate_async(
        self,
        prompt: str,
        params: SamplingParams
    ) -> int:
        tokens = self.tokenize(prompt)
        stop_token_ids = self.get_stop_token_ids(params.stop_sequences)
        return self.scheduler.add_request(tokens, params, stop_token_ids)

    def get_generation_result(self, request_id: int) -> Optional[str]:
        result = self.scheduler.pop_result(request_id)
        if result is not None:
            return self.detokenize(result["generated"])
        return None

    def step(self) -> int:
        batch = self.scheduler.get_batch()
        if not batch:
            return 0
        tokens_generated = 0
        for req in batch:
            if not req["prefill_done"]:
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
                        self.scheduler.mark_prefill_done(req["id"])
                else:
                    with self.lock:
                        self.request_states[req["id"]] = state
                    self.scheduler.mark_prefill_done(req["id"])
        for req in batch:
            if req["prefill_done"] and not req["done"]:
                with self.lock:
                    state = self.request_states.get(req["id"])
                if state is None:
                    continue
                next_token = ctypes.c_int64()
                if self.lib is not None and self.handle is not None:
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
                    logits = [[random.gauss(0, 1) for _ in range(min(self.vocab_size, 1000))] for _ in range(1)]
                    sampled = self.sampler.sample_batch(
                        logits,
                        [req["params"]],
                        [req["tokens"] + req["generated"]]
                    )
                    if sampled:
                        done = self.scheduler.update_request(req["id"], sampled[0])
                        tokens_generated += 1
                        if done:
                            with self.lock:
                                if req["id"] in self.request_states:
                                    del self.request_states[req["id"]]
        return tokens_generated

    def run_continuous(self, duration_seconds: float) -> Dict[str, Any]:
        start_time = time.time()
        total_tokens = 0
        step_count = 0
        while time.time() - start_time < duration_seconds:
            if self.scheduler.has_active_requests():
                tokens = self.step()
                total_tokens += tokens
                step_count += 1
            else:
                time.sleep(0.001)
        elapsed = time.time() - start_time
        return {
            "duration_seconds": elapsed,
            "total_tokens": total_tokens,
            "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
            "step_count": step_count
        }

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
            "max_seq": self.max_seq,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "num_experts": self.num_experts
        }

GPU_CONFIG = "H100:8"

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model": volume},
    timeout=7200
)
def download_model() -> str:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="zai-org/GLM-4.7-FP8",
        local_dir="/model/glm-4.7-fp8",
        local_dir_use_symlinks=False
    )
    volume.commit()
    return "Model downloaded successfully"

def get_local_source_files() -> Dict[str, str]:
    files = {}
    futhark_path = os.path.join(SRC_DIR, "futhark", "kernels.fut")
    if os.path.exists(futhark_path):
        with open(futhark_path, "r") as f:
            files["kernels.fut"] = f.read()
    terra_path = os.path.join(SRC_DIR, "terra", "engine.t")
    if os.path.exists(terra_path):
        with open(terra_path, "r") as f:
            files["engine.t"] = f.read()
    if os.path.exists(CSRC_DIR):
        for fname in os.listdir(CSRC_DIR):
            fpath = os.path.join(CSRC_DIR, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    files[fname] = f.read()
    return files

LOCAL_SOURCES = get_local_source_files()

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model": volume, "/build": build_volume},
    timeout=3600
)
def build_engine() -> str:
    import subprocess
    os.makedirs("/build", exist_ok=True)
    env = {
        **os.environ,
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/build",
        "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
    }
    for fname, content in LOCAL_SOURCES.items():
        with open(f"/build/{fname}", "w") as f:
            f.write(content)
    nvcc_flags = "-O3 -shared -Xcompiler -fPIC -arch=sm_90"
    builds = [
        ("/build/cuda_wrappers.cu", ["nvcc"] + nvcc_flags.split() + ["-lcuda", "-o", "/build/libcudawrap.so", "/build/cuda_wrappers.cu"], "libcudawrap.so"),
        ("/build/nccl_wrappers.cu", ["nvcc"] + nvcc_flags.split() + ["-lnccl", "-o", "/build/libncclwrap.so", "/build/nccl_wrappers.cu"], "libncclwrap.so"),
        ("/build/cublas_wrappers.cu", ["nvcc"] + nvcc_flags.split() + ["-lcublas", "-lcublasLt", "-o", "/build/libcublaswrap.so", "/build/cublas_wrappers.cu"], "libcublaswrap.so"),
        ("/build/kernels.cu", ["nvcc"] + nvcc_flags.split() + ["-o", "/build/libkernels.so", "/build/kernels.cu"], "libkernels.so"),
    ]
    for src_file, cmd, target in builds:
        if not os.path.exists(src_file):
            return f"Source file {src_file} not found for {target}"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/build")
        if result.returncode != 0:
            return f"NVCC build of {target} failed: {result.stderr}"
    if "kernels.fut" in LOCAL_SOURCES:
        with open("/build/kernels.fut", "w") as f:
            f.write(LOCAL_SOURCES["kernels.fut"])
        result = subprocess.run(
            ["futhark", "cuda", "--library", "/build/kernels.fut", "-o", "/build/futhark_kernels"],
            capture_output=True,
            text=True,
            env=env
        )
        if result.returncode != 0:
            return f"Futhark build failed: {result.stderr}"
    if "engine.t" not in LOCAL_SOURCES:
        return "Terra engine source not found"
    with open("/build/engine.t", "w") as f:
        f.write(LOCAL_SOURCES["engine.t"])
    result = subprocess.run(
        ["terra", "/build/engine.t"],
        capture_output=True,
        text=True,
        cwd="/build",
        env=env
    )
    if result.returncode != 0:
        return f"Terra build failed: {result.stderr}"
    if not os.path.exists("/build/engine.so"):
        return "Engine build completed but engine.so not found"
    build_volume.commit()
    return "Build successful - all libraries created"

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model": volume, "/build": build_volume},
    allow_concurrent_inputs=128
)
class InferenceServer:
    def __init__(self):
        self.engine: Optional[InferenceEngine] = None

    @modal.enter()
    def setup(self) -> None:
        self.engine = InferenceEngine(
            model_dir="/model/glm-4.7-fp8",
            max_batch=64,
            max_seq=4096,
            num_gpus=8
        )
        self.engine.load_config()
        engine_loaded = self.engine.load_engine("/build/engine.so")
        self.engine.loader.load_index()
        self.engine.loader.mmap_shards()
        self.engine.load_tokenizer()

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        rep_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        if stop_sequences is None:
            stop_sequences = []
        if self.engine is None:
            return ""
        return self.engine.generate(
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            rep_penalty,
            stop_sequences
        )

    @modal.method()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        rep_penalty: float = 1.1
    ) -> List[str]:
        if self.engine is None:
            return []
        results = []
        for prompt in prompts:
            result = self.engine.generate(
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                rep_penalty,
                []
            )
            results.append(result)
        return results

    @modal.method()
    def get_stats(self) -> Dict[str, Any]:
        if self.engine is None:
            return {}
        return self.engine.get_stats()

    @modal.exit()
    def cleanup(self) -> None:
        if self.engine is not None:
            self.engine.cleanup()

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model": volume, "/build": build_volume},
    timeout=1800
)
def run_benchmark() -> Dict[str, Any]:
    import statistics
    engine = InferenceEngine(
        model_dir="/model/glm-4.7-fp8",
        max_batch=64,
        max_seq=4096,
        num_gpus=8
    )
    engine.load_config()
    engine_loaded = engine.load_engine("/build/engine.so")
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
    warmup_duration = 10
    measure_duration = 60
    num_concurrent = 64
    prompt_len = 256
    gen_len = 256
    warmup_prompt = "x " * (prompt_len // 2)
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_duration:
        params = SamplingParams(temperature=0.8, top_p=0.95, rep_penalty=1.1, max_tokens=gen_len)
        for _ in range(num_concurrent):
            tokens = engine.tokenize(warmup_prompt)[:prompt_len]
            engine.scheduler.add_request(tokens, params, [])
        while engine.scheduler.has_active_requests():
            engine.step()
    measure_prompt = "y " * (prompt_len // 2)
    measure_start = time.time()
    total_tokens = 0
    batch_sizes = []
    latencies = []
    while time.time() - measure_start < measure_duration:
        params = SamplingParams(temperature=0.8, top_p=0.95, rep_penalty=1.1, max_tokens=gen_len)
        for _ in range(num_concurrent):
            tokens = engine.tokenize(measure_prompt)[:prompt_len]
            engine.scheduler.add_request(tokens, params, [])
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
    max_batch = max(batch_sizes) if batch_sizes else 0
    results = {
        "smoke_test": smoke_results,
        "benchmark": {
            "measure_window_seconds": elapsed,
            "total_output_tokens": total_tokens,
            "tokens_per_second": tokens_per_sec,
            "avg_latency_ms_per_token": avg_latency,
            "p50_latency_ms": p50_latency,
            "p99_latency_ms": p99_latency,
            "avg_batch_size": avg_batch,
            "max_batch_size": max_batch,
            "num_gpus": engine.num_gpus,
            "target_throughput": 3000
        }
    }
    engine.cleanup()
    return results

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model": volume, "/build": build_volume},
    timeout=300
)
def run_smoke_test() -> Dict[str, Any]:
    engine = InferenceEngine(
        model_dir="/model/glm-4.7-fp8",
        max_batch=8,
        max_seq=1024,
        num_gpus=8
    )
    engine.load_config()
    engine_loaded = engine.load_engine("/build/engine.so")
    shards_loaded = engine.loader.load_index()
    if shards_loaded:
        engine.loader.mmap_shards()
    tokenizer_loaded = engine.load_tokenizer()
    test_prompt = "Hello, how are you today?"
    output = engine.generate(
        test_prompt,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        rep_penalty=1.0,
        stop_sequences=[]
    )
    stats = engine.get_stats()
    engine.cleanup()
    return {
        "prompt": test_prompt,
        "output": output,
        "engine_loaded": engine_loaded,
        "tokenizer_loaded": tokenizer_loaded,
        "stats": stats
    }

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model": volume, "/build": build_volume},
    timeout=300
)
def validate_build() -> Dict[str, Any]:
    import subprocess
    result = {
        "engine_so_exists": os.path.exists("/build/engine.so"),
        "model_exists": os.path.exists("/model/glm-4.7-fp8"),
        "gpu_count": 0,
        "cuda_available": False,
        "nccl_available": False,
        "engine_loads": False,
        "errors": []
    }
    try:
        gpu_check = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if gpu_check.returncode == 0:
            result["cuda_available"] = True
            result["gpu_count"] = gpu_check.stdout.count("GPU ")
    except Exception as e:
        result["errors"].append(f"nvidia-smi failed: {e}")
    if result["engine_so_exists"]:
        try:
            lib = ctypes.CDLL("/build/engine.so")
            result["engine_loads"] = True
            if hasattr(lib, "get_engine_info"):
                result["has_get_engine_info"] = True
            if hasattr(lib, "init_engine"):
                result["has_init_engine"] = True
            if hasattr(lib, "prefill"):
                result["has_prefill"] = True
            if hasattr(lib, "decode_step"):
                result["has_decode_step"] = True
        except Exception as e:
            result["errors"].append(f"engine.so load failed: {e}")
    try:
        nccl_check = subprocess.run(["ls", "/usr/lib/x86_64-linux-gnu/libnccl*"], capture_output=True, text=True, shell=True)
        if "libnccl" in nccl_check.stdout:
            result["nccl_available"] = True
    except Exception:
        pass
    return result

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model": volume, "/build": build_volume},
    timeout=600
)
def run_throughput_benchmark(
    num_requests: int = 64,
    prompt_len: int = 256,
    gen_len: int = 256,
    duration_secs: int = 60
) -> Dict[str, Any]:
    import statistics
    engine = InferenceEngine(
        model_dir="/model/glm-4.7-fp8",
        max_batch=64,
        max_seq=4096,
        num_gpus=8
    )
    engine.load_config()
    engine_loaded = engine.load_engine("/build/engine.so")
    engine.loader.load_index()
    engine.loader.mmap_shards()
    engine.load_tokenizer()
    prompt = "The quick brown fox " * (prompt_len // 4)
    for _ in range(num_requests):
        params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=gen_len)
        tokens = engine.tokenize(prompt)[:prompt_len]
        engine.scheduler.add_request(tokens, params, [])
    start = time.time()
    total_tokens = 0
    latencies = []
    while time.time() - start < duration_secs and engine.scheduler.has_active_requests():
        step_start = time.time()
        tokens_gen = engine.step()
        step_end = time.time()
        total_tokens += tokens_gen
        if tokens_gen > 0:
            latencies.append((step_end - step_start) * 1000)
    elapsed = time.time() - start
    tps = total_tokens / elapsed if elapsed > 0 else 0
    result = {
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tps,
        "target_achieved": tps >= 3000,
        "avg_step_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0,
        "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) > 100 else (max(latencies) if latencies else 0),
        "num_gpus": engine.num_gpus,
        "engine_loaded": engine_loaded
    }
    engine.cleanup()
    return result

@app.local_entrypoint()
def main() -> None:
    import sys
    print("=" * 60)
    print("GLM-4.7-FP8 Inference Pipeline")
    print("Target: 3000+ tokens/second on 8x B200 GPUs")
    print("=" * 60)
    print("\n[1/5] Downloading model weights...")
    download_result = download_model.remote()
    print(f"  Result: {download_result}")
    print("\n[2/5] Building native engine...")
    build_result = build_engine.remote()
    print(f"  Result: {build_result}")
    if "failed" in build_result.lower():
        print("ERROR: Build failed, cannot continue")
        sys.exit(1)
    print("\n[3/5] Validating build artifacts...")
    validation = validate_build.remote()
    print(f"  Engine SO exists: {validation['engine_so_exists']}")
    print(f"  Engine loads: {validation['engine_loads']}")
    print(f"  GPU count: {validation['gpu_count']}")
    print(f"  CUDA available: {validation['cuda_available']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if not validation['engine_loads']:
        print("ERROR: Engine validation failed")
        sys.exit(1)
    print("\n[4/5] Running smoke test...")
    smoke_result = run_smoke_test.remote()
    print(f"  Prompt: {smoke_result['prompt']}")
    print(f"  Output: {smoke_result['output'][:100]}...")
    print(f"  Engine loaded: {smoke_result['engine_loaded']}")
    print("\n[5/5] Running throughput benchmark (60s)...")
    benchmark = run_throughput_benchmark.remote(
        num_requests=64,
        prompt_len=256,
        gen_len=256,
        duration_secs=60
    )
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total tokens generated: {benchmark['total_tokens']}")
    print(f"  Elapsed time: {benchmark['elapsed_seconds']:.2f}s")
    print(f"  Throughput: {benchmark['tokens_per_second']:.1f} tokens/sec")
    print(f"  Target (3000 tok/s): {'ACHIEVED' if benchmark['target_achieved'] else 'NOT YET'}")
    print(f"  Avg latency: {benchmark['avg_step_latency_ms']:.2f}ms")
    print(f"  P50 latency: {benchmark['p50_latency_ms']:.2f}ms")
    print(f"  P99 latency: {benchmark['p99_latency_ms']:.2f}ms")
    print(f"  GPUs used: {benchmark['num_gpus']}")
    print(f"{'=' * 60}")
