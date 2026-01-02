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
- 5-step deployment pipeline: download → build → validate → smoke test → benchmark

### FastAPI Server (src/server/)

- OpenAI-compatible API: /v1/completions and /v1/chat/completions
- SSE streaming for token-by-token output
- Continuous batching with async queue and backpressure handling
- Health endpoints: /healthz, /readyz, /metrics (Prometheus format)
- Structured JSON logging with request tracking (request_id, prompt_tokens, ttft_ms, tps)
- Mock engine support for CPU-only testing and CI/CD

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
```bash
bash scripts/build.sh
```

**Build Requirements:**
- NVCC (CUDA compiler) - REQUIRED for production builds
- Terra (release-1.2.0) - REQUIRED for engine.so compilation  
- Futhark (0.25.14) - REQUIRED for GPU kernels

**Build Modes:**
- **Production (default):** Requires NVCC and Terra. Fails if engine.so not created.
- **Development:** Set `ALLOW_CPU=1` to build CPU stubs for testing without GPU.

```bash
ALLOW_CPU=1 bash scripts/build.sh
```

**Build Output (in ./build/):**
1. engine.so - Main inference engine (Terra compiled)
2. libcudawrap.so - CUDA memory management
3. libncclwrap.so - NCCL multi-GPU communication
4. libcublaswrap.so - cuBLAS matrix operations
5. libkernels.so - Custom CUDA kernels

**Runtime Library Discovery:**
engine.so uses rpath=$ORIGIN to find dependencies in the same directory.

## Deployment

Deploy to Modal:
```bash
modal deploy src/python/modal_app.py
```

Run full pipeline (download → build → validate → smoke test → benchmark):
```bash
modal run src/python/modal_app.py
```

**GPU Configuration:**
Edit `GPU_CONFIG` in `src/python/modal_app.py` to change GPU type:
- `"H100:8"` - 8x H100 (default, high throughput)
- `"A100:8"` - 8x A100 (widely available)
- `"A10G:1"` - 1x A10G (cheapest for testing)

**Note:** Modal billing limits may prevent large GPU jobs. Check https://modal.com/settings/billing to adjust spending limits.

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

### Latest Session Updates (2026-01-02)

- **CRITICAL: Fixed all Terra for-loop bounds** - Terra uses inclusive bounds, so `for i = 0, N do` iterates N+1 times. All loops now use `for i = 0, N-1 do` for exclusive iteration
- Fixed 30+ for-loops across the entire codebase:
  - KV cache init/alloc/free loops
  - NCCL context initialization
  - Tensor descriptor parsing loops
  - CPU fallback loops (fp8_dequantize, rms_norm, softmax, embedding_lookup, matmul, sample_token)
  - Batch decode loops
  - Layer forward loops (run_prefill, run_decode_step)
  - MoE expert iteration loop
  - Free engine cleanup loops (shard_mappings, tensor_descriptors, weight_map, shard_files)
  - run_batch_decode_ext loop
- **CPU mode crash fix**: run_prefill and run_decode_step now check `cuda_contexts ~= nil` before dereferencing stream
- Added safety check in free_kv_pages for num_pages <= 0
- Safetensors parser fix: header JSON now null-terminated before parsing (100MB sanity check)
- GPU forward pass: cuBLAS SGEMM for lm_head matmul, FP8 dequant kernel, GPU sampling
- Pre-allocated inference buffers (no per-token allocation)
- Opaque pointer API: prefill_opaque, decode_step_opaque, free_request_opaque, run_batch_decode_ext
- Error handling: prefill failures properly clean up batch state and KV pages
- Build script: auto-detects GPU architecture (Blackwell sm_100, Hopper sm_90, Ampere sm_80, Volta sm_70)
- run.py rewritten: real engine.so loading, AutoTokenizer, --smoke/--bench/--serve modes
- All 5 smoke tests pass (FP8 roundtrip, softmax, sampling, matmul, embedding lookup)

### Current Status

- Full 92-layer transformer forward pass implemented in run_layer_forward_gpu
- Layer forward: RMSNorm → QKV projection → RoPE → paged attention → O projection → residual → post-attention RMSNorm → MoE/FFN
- FP8 dequantization before all projections when weights are FP8 (dtype==1)
- Layer-specific KV cache addressing with layer offsets
- MoE routing with top-k expert selection and per-expert matmuls
- GPU buffers for: q, k, v, gate, up, down, router_logits, expert_indices/weights/inputs/outputs, norm
- LayerWeights struct includes dtype fields for all weights
- run_prefill iterates through all 92 layers with prefill attention
- run_decode_step iterates through all 92 layers with decode attention and final RMSNorm + lm_head
- Target: 3000+ tok/s on 8x B200 with tensor parallel and expert parallel

## CUDA/C Source Files (csrc/)

- cuda_wrappers.cu/h: CUDA memory, streams, events, synchronization
- nccl_wrappers.cu/h: NCCL collective operations (AllReduce, AllGather, etc.)
- cublas_wrappers.cu/h: cuBLAS/cuBLASLt GEMM operations
- kernels.cu/h: FP8, RMSNorm, RoPE, softmax, SwiGLU, embedding, sampling
