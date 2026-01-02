FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY csrc/ csrc/
COPY src/terra/ src/terra/
COPY src/futhark/ src/futhark/
COPY scripts/ scripts/

RUN bash scripts/build.sh

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libnccl2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/build/engine.so /app/build/engine.so
COPY --from=builder /build/build/*.so /app/build/

COPY pyproject.toml /app/
COPY src/ /app/src/
COPY run.py /app/

RUN pip3 install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    transformers>=4.35.0 \
    aiohttp>=3.9.0

EXPOSE 5000

ENV MODEL_DIR=/app/model
ENV ENGINE_PATH=/app/build/engine.so
ENV HOST=0.0.0.0
ENV PORT=5000
ENV MAX_CONCURRENCY=64
ENV MAX_SEQ_LEN=4096
ENV NUM_GPUS=8
ENV LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

ENTRYPOINT ["python3", "-m", "src.server.cli", "serve"]
CMD ["--model-dir", "/app/model", "--engine", "/app/build/engine.so"]
