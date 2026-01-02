import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "prompt_tokens"):
            log_obj["prompt_tokens"] = record.prompt_tokens
        if hasattr(record, "gen_tokens"):
            log_obj["gen_tokens"] = record.gen_tokens
        if hasattr(record, "ttft_ms"):
            log_obj["ttft_ms"] = record.ttft_ms
        if hasattr(record, "tps"):
            log_obj["tps"] = record.tps
        if hasattr(record, "batch_size"):
            log_obj["batch_size"] = record.batch_size
        if hasattr(record, "kv_pages"):
            log_obj["kv_pages"] = record.kv_pages
        if hasattr(record, "latency_ms"):
            log_obj["latency_ms"] = record.latency_ms
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("glm-server")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class RequestLogger:
    def __init__(self, logger: logging.Logger, request_id: str):
        self.logger = logger
        self.request_id = request_id
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.prompt_tokens = 0
        self.gen_tokens = 0

    def set_prompt_tokens(self, count: int):
        self.prompt_tokens = count

    def token_generated(self):
        if self.first_token_time is None:
            self.first_token_time = time.time()
        self.gen_tokens += 1

    def complete(self, finish_reason: str = "stop"):
        end_time = time.time()
        latency_ms = (end_time - self.start_time) * 1000
        ttft_ms = (self.first_token_time - self.start_time) * 1000 if self.first_token_time else 0
        duration_s = end_time - self.start_time
        tps = self.gen_tokens / duration_s if duration_s > 0 else 0
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            "",
            0,
            "request_complete",
            (),
            None,
        )
        record.request_id = self.request_id
        record.prompt_tokens = self.prompt_tokens
        record.gen_tokens = self.gen_tokens
        record.ttft_ms = round(ttft_ms, 2)
        record.tps = round(tps, 2)
        record.latency_ms = round(latency_ms, 2)
        self.logger.handle(record)

    def error(self, message: str):
        record = self.logger.makeRecord(
            self.logger.name,
            logging.ERROR,
            "",
            0,
            message,
            (),
            None,
        )
        record.request_id = self.request_id
        self.logger.handle(record)
