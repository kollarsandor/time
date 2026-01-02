# GLM-4.7-FP8 Inference Engine

High-performance inference engine for GLM-4.7-FP8 on 8x NVIDIA B200 GPUs via Modal.com targeting 3000+ tokens/second aggregate decode throughput.

## Quickstart

### Build

```bash
bash scripts/build.sh
```

This compiles:
1. CUDA kernels (FP8, RMSNorm, RoPE, softmax, SwiGLU, sampling)
2. Terra engine to shared library (engine.so)

### Run Server

```bash
python -m src.server.cli serve \
    --model-dir ./model \
    --engine ./build/engine.so \
    --port 5000 \
    --max-concurrency 64
```

Or use the mock engine for testing without GPU:

```bash
python -m src.server.cli serve --mock --port 5000
```

### Docker

```bash
docker build -t glm-inference .
docker run -p 5000:5000 --gpus all \
    -v /path/to/model:/app/model \
    glm-inference
```

## API Endpoints

### Health & Readiness

```bash
curl http://localhost:5000/healthz
curl http://localhost:5000/readyz
curl http://localhost:5000/metrics
```

### Completions (OpenAI-compatible)

Non-streaming:

```bash
curl -X POST http://localhost:5000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Write a poem about AI:",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

Streaming:

```bash
curl -X POST http://localhost:5000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Write a poem about AI:",
        "max_tokens": 100,
        "stream": true
    }' --no-buffer
```

### Chat Completions (OpenAI-compatible)

Non-streaming:

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 100
    }'
```

Streaming:

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100,
        "stream": true
    }' --no-buffer
```

## Benchmark

Run benchmark against a running server:

```bash
python -m src.server.cli benchmark \
    --duration 60 \
    --concurrency 32 \
    --prompt-len 128 \
    --output-len 128
```

## Configuration

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-dir` | Model directory path | `./model` |
| `--engine` | Engine .so path | `./build/engine.so` |
| `--host` | Host to bind | `0.0.0.0` |
| `--port` | Port to bind | `5000` |
| `--max-concurrency` | Max concurrent requests | `64` |
| `--max-batch-tokens` | Max tokens per batch | `32768` |
| `--max-seq-len` | Max sequence length | `4096` |
| `--max-new-tokens` | Max new tokens per request | `2048` |
| `--max-prompt-tokens` | Max prompt tokens | `2048` |
| `--num-gpus` | Number of GPUs | `8` |
| `--log-level` | Log level (INFO, DEBUG, etc) | `INFO` |
| `--queue-size` | Request queue size | `256` |
| `--mock` | Use mock engine | `false` |

### Environment Variables

All CLI options can be set via environment variables:

- `MODEL_DIR`
- `ENGINE_PATH`
- `HOST`
- `PORT`
- `MAX_CONCURRENCY`
- `MAX_BATCH_TOKENS`
- `MAX_SEQ_LEN`
- `MAX_NEW_TOKENS`
- `MAX_PROMPT_TOKENS`
- `NUM_GPUS`
- `LOG_LEVEL`
- `QUEUE_SIZE`

## Tuning Tips

### Batching

- Set `--max-concurrency` based on GPU memory
- Higher concurrency = better throughput, higher latency
- For low latency: use `--max-concurrency 8-16`
- For high throughput: use `--max-concurrency 32-64`

### Memory

- KV cache uses ~16MB per page per layer
- With 92 layers, budget ~1.5GB per concurrent request
- 8x B200 (180GB each) can handle 64+ concurrent requests

### Monitoring

Access Prometheus metrics at `/metrics`:

- `glm_request_total`: Total requests
- `glm_tokens_per_second`: Current TPS
- `glm_latency_avg_ms`: Average latency
- `glm_ttft_avg_ms`: Time to first token
- `glm_queue_depth`: Current queue depth
- `glm_active_requests`: Active requests
- `glm_kv_pages_used`: KV cache usage

## Tests

Run tests (no GPU required):

```bash
python -m pytest tests/ -v
```

Run smoke tests:

```bash
python tests/test_smoke.py
```

## Architecture

```
src/
├── server/           # Production ASGI server
│   ├── app.py       # FastAPI application
│   ├── cli.py       # CLI entrypoint
│   ├── config.py    # Configuration
│   ├── engine_worker.py  # Continuous batching worker
│   ├── engine_wrapper.py # Native engine bindings
│   ├── metrics.py   # Prometheus metrics
│   ├── schemas.py   # Pydantic models
│   └── tokenizer.py # Tokenizer wrapper
├── terra/           # Terra runtime
│   └── engine.t     # Native inference engine
├── futhark/         # Futhark GPU kernels
│   └── kernels.fut
└── python/          # Python orchestration
    └── modal_app.py # Modal deployment
```

## Modal Deployment

Deploy to Modal.com:

```bash
modal deploy src/python/modal_app.py
```

Run benchmark on Modal:

```bash
modal run src/python/modal_app.py
```
