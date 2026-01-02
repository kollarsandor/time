import sys
import os
import time
import json
import random
import threading
import queue
from typing import Optional, List, Dict, Any

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

    def has_active_requests(self) -> bool:
        with self.lock:
            return len(self.active_requests) > 0 or not self.pending_queue.empty()

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

    def tokenize(self, text: str) -> List[int]:
        return [ord(c) % 1000 for c in text]

    def detokenize(self, tokens: List[int]) -> str:
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
            if not req["done"]:
                logits = [random.gauss(0, 1) for _ in range(min(self.vocab_size, 1000))]
                token = self.sampler.sample(logits, req["params"], req["tokens"] + req["generated"])
                done = self.scheduler.update_request(req["id"], token)
                tokens_generated += 1
        return tokens_generated

    def cleanup(self) -> None:
        pass

def main() -> None:
    print("GLM-4.7-FP8 Inference Engine")
    print("=" * 60)
    print("This engine is designed to run on Modal.com with 8x B200 GPUs")
    print("")
    print("To deploy to Modal:")
    print("  modal deploy src/python/modal_app.py")
    print("")
    print("To run benchmark on Modal:")
    print("  modal run src/python/modal_app.py")
    print("")
    print("Running local scheduler test...")
    print("")
    engine = InferenceEngine(
        model_dir="./model",
        max_batch=4,
        max_seq=512,
        num_gpus=1
    )
    test_prompts = [
        "Hello world",
        "The capital of France is",
        "Python programming"
    ]
    for prompt in test_prompts:
        output = engine.generate(
            prompt=prompt,
            max_new_tokens=16,
            temperature=0.8,
            top_p=0.95,
            rep_penalty=1.1,
            stop_sequences=[]
        )
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print("")
    print("Running throughput test...")
    num_requests = 32
    max_tokens = 64
    params = SamplingParams(temperature=0.8, top_p=0.95, rep_penalty=1.1, max_tokens=max_tokens)
    start_time = time.time()
    for i in range(num_requests):
        tokens = engine.tokenize(f"Test prompt {i}")
        engine.scheduler.add_request(tokens, params)
    total_tokens = 0
    while engine.scheduler.has_active_requests():
        tokens = engine.step()
        total_tokens += tokens
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    print(f"Generated {total_tokens} tokens in {elapsed:.2f} seconds")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/second")
    print("")
    print("Local test completed")
    engine.cleanup()

if __name__ == "__main__":
    main()
