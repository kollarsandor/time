# GLM-4.7-FP8 Inference Engine

## Overview

High-performance inference engine for GLM-4.7-FP8 on 8x NVIDIA B200 GPUs via Modal.com targeting 3000+ tokens/second aggregate decode throughput.

## Architecture

### Futhark Kernels (src/futhark/kernels.fut)

- FP8 E4M3 quantization/dequantization with proper handling of denormals, special values, and saturation
- RMSNorm with configurable epsilon
- SwiGLU and GEGLU activations
- RoPE positional encoding with configurable base frequency
- Paged attention for both prefill and decode phases with online softmax
- FlashAttention-style tiled computation with L2-aware memory access
- MoE routing with top-k expert selection and weighted combination
- GPU-side sampling with temperature, top-p, repetition penalty, frequency penalty, presence penalty
- Multi-GPU operations: all-reduce, all-gather, all-to-all for tensor parallel and expert parallel

### Terra Runtime (src/terra/engine.t)

- Safetensors zero-copy loading with mmap and header parsing
- Model index parsing for sharded weights
- Static memory planning with pre-allocated device buffers
- Paged KV cache with page allocation, free list management, and mutex synchronization
- Multi-GPU context initialization with CUDA streams
- NCCL context setup for distributed communication
- Batch state management for continuous batching
- Request lifecycle: prefill, decode steps, completion detection
- FP8 dequantization and F32 quantization CPU fallbacks
- Sampling with temperature, top-p, repetition penalty
- C ABI exports: init_engine, prefill, decode_step, free_request_state, free_engine, get_engine_info, run_batch_decode

### Python Orchestration (src/python/)

- modal_app.py: Modal deployment with B200:8 GPU configuration
- Continuous batching scheduler with pending queue, active tracking, completion handling
- Paged KV cache manager with thread-safe allocation
- Safetensors loader with index parsing and mmap
- GPU sampler with temperature scaling, top-p filtering, repetition penalty
- Tokenizer integration with HuggingFace transformers
- Benchmark suite with warmup, measurement, latency statistics

## Model Specifications

- Model: GLM-4.7-FP8 (zai-org/GLM-4.7-FP8)
- Layers: 92
- Experts: 160 with top-k=8 routing
- Hidden dimension: 4096
- Attention heads: 32
- Head dimension: 128
- Vocabulary size: 151552
- Intermediate size: 13696
- RoPE base: 10000
- Max sequence length: 131072

## Hardware Target

- 8x NVIDIA B200 (Blackwell) GPUs
- Tensor parallel across 8 GPUs for attention
- Expert parallel across 8 GPUs for MoE (20 experts per GPU)

## Building

Run the build script:
```
bash scripts/build.sh
```

This compiles:
1. Futhark kernels to CUDA/OpenCL/multicore library
2. Terra engine to shared library (engine.so)

## Deployment

Deploy to Modal:
```
modal deploy src/python/modal_app.py
```

Run benchmark:
```
modal run src/python/modal_app.py
```

## Local Testing

Run local scheduler test:
```
python run.py
```

Run local benchmark (requires engine.so):
```
python src/python/bench.py ./model ./build/engine.so
```

## API

### Engine Initialization
```c
EngineHandle* init_engine(char* model_dir, int32 max_batch, int32 max_seq, int32 num_gpus)
```

### Prefill
```c
int32 prefill(EngineHandle* handle, int64 request_id, int64* token_ids, int32 seq_len, RequestState* state_out)
```

### Decode Step
```c
int32 decode_step(EngineHandle* handle, RequestState* state, int64* next_token_out)
```

### Cleanup
```c
void free_request_state(RequestState* state)
void free_engine(EngineHandle* handle)
```

## Performance Targets

- Aggregate decode throughput: 3000+ tokens/second
- Batch size: up to 64 concurrent requests
- Maximum sequence length: 4096 (configurable up to 131072)
- Page size: 16 tokens for KV cache

## Recent Changes (2026-01-02)

- Fixed Terra for-loop bounds (0,N-1 exclusive iteration)
- Added real CUDA wrappers (csrc/cuda_wrappers.cu/h) with actual cudaMalloc/memcpy
- Added NCCL wrappers (csrc/nccl_wrappers.cu/h) for multi-GPU communication
- Added cuBLAS wrappers (csrc/cublas_wrappers.cu/h) for GEMM operations
- Added CUDA kernels (csrc/kernels.cu/h) for FP8 dequant, RMSNorm, RoPE, softmax, SwiGLU
- Fixed softmax kernel (replaced incorrect atomicMax with shared memory reduction)
- Fixed RMSNorm kernel (proper shared memory reduction)
- Terra engine now loads real weights from safetensors to GPU
- Real forward pass: embedding lookup -> lm_head -> logits
- Supports both GPU mode (with CUDA) and CPU fallback mode
- Build script creates CPU stubs when NVCC unavailable
- Added smoke tests (tests/test_smoke.py) - all 5 pass

## CUDA/C Source Files (csrc/)

- cuda_wrappers.cu/h: CUDA memory, streams, events, synchronization
- nccl_wrappers.cu/h: NCCL collective operations (AllReduce, AllGather, etc.)
- cublas_wrappers.cu/h: cuBLAS/cuBLASLt GEMM operations
- kernels.cu/h: FP8, RMSNorm, RoPE, softmax, SwiGLU, embedding, sampling
