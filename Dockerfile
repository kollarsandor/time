FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    libnccl2 \
    libnccl-dev \
    libncurses5-dev \
    libreadline-dev \
    zlib1g-dev \
    llvm-14 \
    llvm-14-dev \
    clang-14 \
    libclang-14-dev \
    && rm -rf /var/lib/apt/lists/*

ENV TERRA_VERSION=release-1.2.0
RUN cd /tmp && \
    wget -q https://github.com/terralang/terra/releases/download/${TERRA_VERSION}/terra-Linux-x86_64-${TERRA_VERSION}.tar.xz && \
    tar -xf terra-Linux-x86_64-${TERRA_VERSION}.tar.xz && \
    mv terra-Linux-x86_64-${TERRA_VERSION} /opt/terra && \
    rm terra-Linux-x86_64-${TERRA_VERSION}.tar.xz

ENV PATH="/opt/terra/bin:${PATH}"
ENV TERRA_PATH="/opt/terra/share/terra/lib"

RUN terra --version && echo "Terra installed successfully"

ENV FUTHARK_VERSION=0.25.14
RUN cd /tmp && \
    wget -q https://futhark-lang.org/releases/futhark-${FUTHARK_VERSION}-linux-x86_64.tar.xz && \
    tar -xf futhark-${FUTHARK_VERSION}-linux-x86_64.tar.xz && \
    mv futhark-${FUTHARK_VERSION}-linux-x86_64/bin/futhark /usr/local/bin/ && \
    rm -rf futhark-${FUTHARK_VERSION}-linux-x86_64* && \
    futhark --version && echo "Futhark ${FUTHARK_VERSION} installed successfully"

WORKDIR /build

COPY csrc/ csrc/
COPY src/terra/ src/terra/
COPY src/futhark/ src/futhark/
COPY scripts/ scripts/

RUN bash scripts/build.sh

RUN test -f /build/build/engine.so || (echo "ERROR: engine.so not created" && exit 1)

RUN echo "=== Verifying engine.so ===" && \
    ls -la /build/build/*.so && \
    ldd /build/build/engine.so

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libnccl2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/build/*.so /app/build/

ENV LD_LIBRARY_PATH="/app/build:${LD_LIBRARY_PATH}"

RUN ls -la /app/build/*.so && \
    echo "Checking engine.so dependencies..." && \
    ldd /app/build/engine.so

COPY pyproject.toml /app/
COPY src/ /app/src/
COPY run.py /app/

RUN pip3 install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    transformers>=4.35.0 \
    aiohttp>=3.9.0

RUN python3 -c "import ctypes; eng = ctypes.CDLL('/app/build/engine.so'); print('engine.so loads successfully')"

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
