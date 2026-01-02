import time
from typing import Optional
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricsCollector:
    request_count: int = 0
    error_count: int = 0
    tokens_generated: int = 0
    prompt_tokens_total: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    ttft_sum: float = 0.0
    ttft_count: int = 0
    queue_depth: int = 0
    active_requests: int = 0
    batch_size_sum: int = 0
    batch_count: int = 0
    kv_pages_used: int = 0
    kv_pages_max: int = 0
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
    _lock: Lock = field(default_factory=Lock)

    def inc_request(self):
        with self._lock:
            self.request_count += 1

    def inc_error(self):
        with self._lock:
            self.error_count += 1

    def add_tokens(self, count: int):
        with self._lock:
            self.tokens_generated += count

    def add_prompt_tokens(self, count: int):
        with self._lock:
            self.prompt_tokens_total += count

    def record_latency(self, latency_ms: float):
        with self._lock:
            self.latency_sum += latency_ms
            self.latency_count += 1

    def record_ttft(self, ttft_ms: float):
        with self._lock:
            self.ttft_sum += ttft_ms
            self.ttft_count += 1

    def set_queue_depth(self, depth: int):
        with self._lock:
            self.queue_depth = depth

    def set_active_requests(self, count: int):
        with self._lock:
            self.active_requests = count

    def record_batch(self, size: int):
        with self._lock:
            self.batch_size_sum += size
            self.batch_count += 1

    def set_kv_cache(self, used: int, max_pages: int):
        with self._lock:
            self.kv_pages_used = used
            self.kv_pages_max = max_pages

    def set_gpu_memory(self, used: int, total: int):
        with self._lock:
            self.gpu_memory_used = used
            self.gpu_memory_total = total

    def to_prometheus(self) -> str:
        with self._lock:
            lines = []
            lines.append("# HELP glm_request_total Total number of requests")
            lines.append("# TYPE glm_request_total counter")
            lines.append(f"glm_request_total {self.request_count}")
            lines.append("# HELP glm_error_total Total number of errors")
            lines.append("# TYPE glm_error_total counter")
            lines.append(f"glm_error_total {self.error_count}")
            lines.append("# HELP glm_tokens_generated_total Total tokens generated")
            lines.append("# TYPE glm_tokens_generated_total counter")
            lines.append(f"glm_tokens_generated_total {self.tokens_generated}")
            lines.append("# HELP glm_prompt_tokens_total Total prompt tokens processed")
            lines.append("# TYPE glm_prompt_tokens_total counter")
            lines.append(f"glm_prompt_tokens_total {self.prompt_tokens_total}")
            avg_latency = self.latency_sum / self.latency_count if self.latency_count > 0 else 0
            lines.append("# HELP glm_latency_avg_ms Average request latency in milliseconds")
            lines.append("# TYPE glm_latency_avg_ms gauge")
            lines.append(f"glm_latency_avg_ms {avg_latency:.2f}")
            avg_ttft = self.ttft_sum / self.ttft_count if self.ttft_count > 0 else 0
            lines.append("# HELP glm_ttft_avg_ms Average time to first token in milliseconds")
            lines.append("# TYPE glm_ttft_avg_ms gauge")
            lines.append(f"glm_ttft_avg_ms {avg_ttft:.2f}")
            lines.append("# HELP glm_queue_depth Current queue depth")
            lines.append("# TYPE glm_queue_depth gauge")
            lines.append(f"glm_queue_depth {self.queue_depth}")
            lines.append("# HELP glm_active_requests Current active requests")
            lines.append("# TYPE glm_active_requests gauge")
            lines.append(f"glm_active_requests {self.active_requests}")
            avg_batch = self.batch_size_sum / self.batch_count if self.batch_count > 0 else 0
            lines.append("# HELP glm_batch_size_avg Average batch size")
            lines.append("# TYPE glm_batch_size_avg gauge")
            lines.append(f"glm_batch_size_avg {avg_batch:.2f}")
            lines.append("# HELP glm_kv_pages_used KV cache pages used")
            lines.append("# TYPE glm_kv_pages_used gauge")
            lines.append(f"glm_kv_pages_used {self.kv_pages_used}")
            lines.append("# HELP glm_kv_pages_max KV cache pages max")
            lines.append("# TYPE glm_kv_pages_max gauge")
            lines.append(f"glm_kv_pages_max {self.kv_pages_max}")
            if self.latency_count > 0:
                tps = self.tokens_generated / (self.latency_sum / 1000.0) if self.latency_sum > 0 else 0
                lines.append("# HELP glm_tokens_per_second Current tokens per second")
                lines.append("# TYPE glm_tokens_per_second gauge")
                lines.append(f"glm_tokens_per_second {tps:.2f}")
            return "\n".join(lines) + "\n"


metrics = MetricsCollector()
