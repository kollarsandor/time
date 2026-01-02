import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


class TestServerSchemas:
    def test_completion_request_validation(self):
        from src.server.schemas import CompletionRequest
        req = CompletionRequest(prompt="Hello world")
        assert req.prompt == "Hello world"
        assert req.max_tokens == 256
        assert req.temperature == 0.7

    def test_completion_request_empty_prompt_fails(self):
        from src.server.schemas import CompletionRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="")

    def test_chat_request_validation(self):
        from src.server.schemas import ChatCompletionRequest, ChatMessage
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_chat_request_empty_messages_fails(self):
        from src.server.schemas import ChatCompletionRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=[])

    def test_temperature_bounds(self):
        from src.server.schemas import CompletionRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", temperature=3.0)
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", temperature=-0.5)

    def test_max_tokens_bounds(self):
        from src.server.schemas import CompletionRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", max_tokens=0)
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", max_tokens=10000)


class TestServerMetrics:
    def test_metrics_counter(self):
        from src.server.metrics import MetricsCollector
        m = MetricsCollector()
        m.inc_request()
        m.inc_request()
        assert m.request_count == 2

    def test_metrics_prometheus_format(self):
        from src.server.metrics import MetricsCollector
        m = MetricsCollector()
        m.inc_request()
        m.add_tokens(100)
        output = m.to_prometheus()
        assert "glm_request_total 1" in output
        assert "glm_tokens_generated_total 100" in output

    def test_metrics_latency(self):
        from src.server.metrics import MetricsCollector
        m = MetricsCollector()
        m.record_latency(100.0)
        m.record_latency(200.0)
        assert m.latency_count == 2
        assert m.latency_sum == 300.0


class TestMockEngine:
    def test_mock_engine_load(self):
        from src.server.engine_wrapper import MockEngine
        engine = MockEngine()
        assert engine.load() is True
        assert engine.is_loaded() is True

    def test_mock_engine_prefill(self):
        from src.server.engine_wrapper import MockEngine
        engine = MockEngine()
        engine.load()
        state_id = engine.prefill(
            request_id=1,
            token_ids=[1, 2, 3],
            temperature=0.7,
            top_p=0.95,
            rep_penalty=1.0,
            max_tokens=10,
        )
        assert state_id is not None

    def test_mock_engine_decode(self):
        from src.server.engine_wrapper import MockEngine
        engine = MockEngine(max_tokens=5)
        engine.load()
        state_id = engine.prefill(
            request_id=1,
            token_ids=[1],
            temperature=0.7,
            top_p=0.95,
            rep_penalty=1.0,
            max_tokens=5,
        )
        tokens = []
        done = False
        while not done:
            token, done = engine.decode_step(state_id)
            if token is not None:
                tokens.append(token)
        assert len(tokens) == 5


class TestMockTokenizer:
    def test_mock_tokenizer_encode(self):
        from src.server.tokenizer import MockTokenizer
        tok = MockTokenizer()
        tokens = tok.encode("Hello")
        assert len(tokens) == 5

    def test_mock_tokenizer_decode(self):
        from src.server.tokenizer import MockTokenizer
        tok = MockTokenizer()
        text = tok.decode([72, 101, 108, 108, 111])
        assert text == "Hello"


class TestServerEndpoints:
    @pytest.fixture
    def client(self):
        from src.server.app import create_app
        from src.server.config import ServerConfig
        config = ServerConfig(
            model_dir="./model",
            engine_path="./build/engine.so",
            max_concurrency=4,
        )
        app = create_app(config, use_mock=True)
        with TestClient(app) as c:
            yield c

    def test_healthz(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_when_ready(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["engine_loaded"] is True

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "glm_request_total" in response.text

    def test_completion_non_stream(self, client):
        response = client.post(
            "/v1/completions",
            json={
                "prompt": "Hello world",
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert "usage" in data

    def test_chat_completion_non_stream(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"


class TestBackpressure:
    def test_queue_limit_validation(self):
        from src.server.config import ServerConfig
        config = ServerConfig(queue_size=10)
        assert config.queue_size == 10

    def test_max_concurrency_validation(self):
        from src.server.config import ServerConfig
        config = ServerConfig(max_concurrency=32)
        assert config.max_concurrency == 32


class TestGracefulShutdown:
    def test_shutdown_event_exists(self):
        from src.server.app import state
        assert hasattr(state, "shutdown_event")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
