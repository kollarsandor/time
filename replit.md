
# GLM-4.7-FP8 Inference Engine

## Overview

This project implements a high-performance inference engine for GLM-4.7-FP8 on 8x NVIDIA B200 GPUs via Modal.com.

## Architecture

### Components

1. Futhark Kernels (src/futhark/kernels.fut)
   - FP8 E4M3 quantization/dequantization
   - RMSNorm, SwiGLU/GEGLU activation
   - RoPE positional encoding
   - Paged attention (prefill + decode)
   - MoE routing and expert computation
   - GPU sampling (temperature, top-p, repetition penalty)

2. Terra Runtime (src/terra/engine.t)
   - Safetensors zero-copy loading with mmap
   - Static memory planning
   - Paged KV cache management
   - C ABI exports for Python FFI

3. Python Orchestration (src/python/)
   - Modal app deployment configuration
   - Continuous batching scheduler
   - Benchmark and validation scripts

## Building

Run scripts/build.sh to build kernels and engine.

## Deployment

Deploy to Modal with: modal deploy src/python/modal_app.py

## Specifications

- Model: GLM-4.7-FP8 (92 layers, 160 experts, top-k=8)
- Hardware: 8x NVIDIA B200 (Blackwell)
- Target throughput: 3000+ tokens/s
