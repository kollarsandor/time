import ctypes
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass


class EngineHandle(ctypes.Structure):
    pass


class RequestState(ctypes.Structure):
    pass


@dataclass
class EngineInfo:
    hidden_dim: int
    num_layers: int
    num_heads: int
    vocab_size: int
    num_experts: int
    top_k_experts: int
    num_gpus: int
    max_batch: int
    max_seq: int
    kv_pages_used: int
    kv_pages_max: int
    active_requests: int
    num_tensors: int
    use_gpu: int


class NativeEngine:
    def __init__(self, engine_path: str, model_dir: str, max_batch: int, max_seq: int, num_gpus: int):
        self.engine_path = engine_path
        self.model_dir = model_dir
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.num_gpus = num_gpus
        self.lib: Optional[ctypes.CDLL] = None
        self.handle: Optional[ctypes.POINTER(EngineHandle)] = None
        self._loaded = False

    def load(self) -> bool:
        if not os.path.exists(self.engine_path):
            return False
        try:
            self.lib = ctypes.CDLL(self.engine_path)
            self._setup_functions()
            model_dir_bytes = self.model_dir.encode("utf-8")
            self.handle = self.lib.init_engine(
                model_dir_bytes,
                ctypes.c_int32(self.max_batch),
                ctypes.c_int32(self.max_seq),
                ctypes.c_int32(self.num_gpus),
            )
            if self.handle:
                self._loaded = True
                return True
            return False
        except Exception:
            return False

    def _setup_functions(self):
        self.lib.init_engine.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.lib.init_engine.restype = ctypes.POINTER(EngineHandle)
        self.lib.prefill_opaque.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
        ]
        self.lib.prefill_opaque.restype = ctypes.c_void_p
        self.lib.decode_step_opaque.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.decode_step_opaque.restype = ctypes.c_int32
        self.lib.free_request_opaque.argtypes = [ctypes.POINTER(EngineHandle), ctypes.c_void_p]
        self.lib.free_request_opaque.restype = None
        self.lib.free_engine.argtypes = [ctypes.POINTER(EngineHandle)]
        self.lib.free_engine.restype = None
        self.lib.get_engine_info.argtypes = [ctypes.POINTER(EngineHandle), ctypes.c_int32]
        self.lib.get_engine_info.restype = ctypes.c_int64
        self.lib.run_batch_decode_ext.argtypes = [
            ctypes.POINTER(EngineHandle),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.run_batch_decode_ext.restype = ctypes.c_int32

    def is_loaded(self) -> bool:
        return self._loaded and self.handle is not None

    def get_info(self) -> Optional[EngineInfo]:
        if not self.is_loaded():
            return None
        return EngineInfo(
            hidden_dim=int(self.lib.get_engine_info(self.handle, 0)),
            num_layers=int(self.lib.get_engine_info(self.handle, 1)),
            num_heads=int(self.lib.get_engine_info(self.handle, 2)),
            vocab_size=int(self.lib.get_engine_info(self.handle, 3)),
            num_experts=int(self.lib.get_engine_info(self.handle, 4)),
            top_k_experts=int(self.lib.get_engine_info(self.handle, 5)),
            num_gpus=int(self.lib.get_engine_info(self.handle, 6)),
            max_batch=int(self.lib.get_engine_info(self.handle, 7)),
            max_seq=int(self.lib.get_engine_info(self.handle, 8)),
            kv_pages_used=int(self.lib.get_engine_info(self.handle, 9)),
            kv_pages_max=int(self.lib.get_engine_info(self.handle, 10)),
            active_requests=int(self.lib.get_engine_info(self.handle, 11)),
            num_tensors=int(self.lib.get_engine_info(self.handle, 12)),
            use_gpu=int(self.lib.get_engine_info(self.handle, 13)),
        )

    def prefill(
        self,
        request_id: int,
        token_ids: List[int],
        temperature: float,
        top_p: float,
        rep_penalty: float,
        max_tokens: int,
    ) -> Optional[ctypes.c_void_p]:
        if not self.is_loaded():
            return None
        token_array = (ctypes.c_int64 * len(token_ids))(*token_ids)
        state_ptr = self.lib.prefill_opaque(
            self.handle,
            ctypes.c_int64(request_id),
            token_array,
            ctypes.c_int32(len(token_ids)),
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
            ctypes.c_float(rep_penalty),
            ctypes.c_int32(max_tokens),
        )
        return state_ptr if state_ptr else None

    def decode_step(self, state_ptr: ctypes.c_void_p) -> Tuple[Optional[int], bool]:
        if not self.is_loaded() or not state_ptr:
            return None, True
        next_token = ctypes.c_int64(0)
        is_done = ctypes.c_int32(0)
        status = self.lib.decode_step_opaque(
            self.handle,
            state_ptr,
            ctypes.byref(next_token),
            ctypes.byref(is_done),
        )
        if status < 0:
            return None, True
        return int(next_token.value), bool(is_done.value)

    def batch_decode(
        self, state_ptrs: List[ctypes.c_void_p]
    ) -> List[Tuple[Optional[int], bool]]:
        if not self.is_loaded() or not state_ptrs:
            return [(None, True) for _ in state_ptrs]
        num_states = len(state_ptrs)
        ptr_array = (ctypes.c_void_p * num_states)(*state_ptrs)
        out_tokens = (ctypes.c_int64 * num_states)()
        out_done = (ctypes.c_int32 * num_states)()
        status = self.lib.run_batch_decode_ext(
            self.handle,
            ptr_array,
            ctypes.c_int32(num_states),
            out_tokens,
            out_done,
        )
        if status < 0:
            return [(None, True) for _ in state_ptrs]
        results = []
        for i in range(num_states):
            token = int(out_tokens[i]) if not out_done[i] else None
            done = bool(out_done[i])
            results.append((token, done))
        return results

    def free_request(self, state_ptr: ctypes.c_void_p):
        if self.is_loaded() and state_ptr:
            self.lib.free_request_opaque(self.handle, state_ptr)

    def shutdown(self):
        if self.is_loaded():
            self.lib.free_engine(self.handle)
            self.handle = None
            self._loaded = False


class MockEngine:
    def __init__(self, vocab_size: int = 151552, max_tokens: int = 256):
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self._loaded = False
        self._request_counter = 0
        self._active_requests: dict = {}

    def load(self) -> bool:
        self._loaded = True
        return True

    def is_loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            hidden_dim=4096,
            num_layers=92,
            num_heads=32,
            vocab_size=self.vocab_size,
            num_experts=160,
            top_k_experts=8,
            num_gpus=8,
            max_batch=64,
            max_seq=4096,
            kv_pages_used=len(self._active_requests) * 16,
            kv_pages_max=4096,
            active_requests=len(self._active_requests),
            num_tensors=1000,
            use_gpu=0,
        )

    def prefill(
        self,
        request_id: int,
        token_ids: List[int],
        temperature: float,
        top_p: float,
        rep_penalty: float,
        max_tokens: int,
    ) -> Optional[int]:
        if not self._loaded:
            return None
        self._request_counter += 1
        state_id = self._request_counter
        self._active_requests[state_id] = {
            "request_id": request_id,
            "tokens_generated": 0,
            "max_tokens": max_tokens,
            "prompt_len": len(token_ids),
        }
        return state_id

    def decode_step(self, state_id: int) -> Tuple[Optional[int], bool]:
        if state_id not in self._active_requests:
            return None, True
        state = self._active_requests[state_id]
        state["tokens_generated"] += 1
        import random
        token = random.randint(1, self.vocab_size - 1)
        done = state["tokens_generated"] >= state["max_tokens"]
        if done:
            del self._active_requests[state_id]
        return token, done

    def batch_decode(self, state_ids: List[int]) -> List[Tuple[Optional[int], bool]]:
        return [self.decode_step(sid) for sid in state_ids]

    def free_request(self, state_id: int):
        if state_id in self._active_requests:
            del self._active_requests[state_id]

    def shutdown(self):
        self._active_requests.clear()
        self._loaded = False
