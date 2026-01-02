import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    model_dir: str = "./model"
    engine_path: str = "./build/engine.so"
    host: str = "0.0.0.0"
    port: int = 5000
    max_concurrency: int = 64
    max_batch_tokens: int = 32768
    max_seq_len: int = 4096
    max_new_tokens: int = 2048
    max_prompt_tokens: int = 2048
    max_body_size: int = 10 * 1024 * 1024
    request_timeout: float = 300.0
    num_gpus: int = 8
    log_level: str = "INFO"
    queue_size: int = 256
    worker_poll_interval: float = 0.001

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            model_dir=os.getenv("MODEL_DIR", "./model"),
            engine_path=os.getenv("ENGINE_PATH", "./build/engine.so"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "5000")),
            max_concurrency=int(os.getenv("MAX_CONCURRENCY", "64")),
            max_batch_tokens=int(os.getenv("MAX_BATCH_TOKENS", "32768")),
            max_seq_len=int(os.getenv("MAX_SEQ_LEN", "4096")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "2048")),
            max_prompt_tokens=int(os.getenv("MAX_PROMPT_TOKENS", "2048")),
            max_body_size=int(os.getenv("MAX_BODY_SIZE", str(10 * 1024 * 1024))),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "300.0")),
            num_gpus=int(os.getenv("NUM_GPUS", "8")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            queue_size=int(os.getenv("QUEUE_SIZE", "256")),
            worker_poll_interval=float(os.getenv("WORKER_POLL_INTERVAL", "0.001")),
        )
