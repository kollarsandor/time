#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
CSRC_DIR="$PROJECT_DIR/csrc"
SRC_DIR="$PROJECT_DIR/src"

ALLOW_CPU="${ALLOW_CPU:-0}"

mkdir -p "$BUILD_DIR"

echo "=== GLM-4.7-FP8 Inference Engine Build ==="
echo "Project dir: $PROJECT_DIR"
echo "Build dir: $BUILD_DIR"
echo "ALLOW_CPU: $ALLOW_CPU"

CUDA_AVAILABLE=0

if command -v nvcc &> /dev/null; then
    echo "Building CUDA wrappers..."
    
    CUDA_ARCH=""
    if nvcc --help | grep -q "compute_100"; then
        CUDA_ARCH="-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_90,code=sm_90"
        echo "Using Blackwell (B200) + Hopper architecture"
    elif nvcc --help | grep -q "compute_90"; then
        CUDA_ARCH="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_80,code=sm_80"
        echo "Using Hopper + Ampere architecture"
    elif nvcc --help | grep -q "compute_80"; then
        CUDA_ARCH="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_70,code=sm_70"
        echo "Using Ampere + Volta architecture"
    else
        CUDA_ARCH="-gencode=arch=compute_70,code=sm_70"
        echo "Using Volta architecture"
    fi
    
    CUDA_FLAGS="-shared -fPIC -O3 -Xcompiler -fPIC $CUDA_ARCH"
    
    nvcc $CUDA_FLAGS \
        -I"$CSRC_DIR" \
        "$CSRC_DIR/cuda_wrappers.cu" \
        -o "$BUILD_DIR/libcudawrap.so" \
        -lcudart
    echo "Built libcudawrap.so (GPU)"
    
    nvcc $CUDA_FLAGS \
        -I"$CSRC_DIR" \
        "$CSRC_DIR/nccl_wrappers.cu" \
        -o "$BUILD_DIR/libncclwrap.so" \
        -lcudart -lnccl
    echo "Built libncclwrap.so (GPU)"
    
    nvcc $CUDA_FLAGS \
        -I"$CSRC_DIR" \
        "$CSRC_DIR/cublas_wrappers.cu" \
        -o "$BUILD_DIR/libcublaswrap.so" \
        -lcudart -lcublas -lcublasLt
    echo "Built libcublaswrap.so (GPU)"
    
    nvcc $CUDA_FLAGS \
        -I"$CSRC_DIR" \
        "$CSRC_DIR/kernels.cu" \
        -o "$BUILD_DIR/libkernels.so" \
        -lcudart
    echo "Built libkernels.so (GPU)"
    
    CUDA_AVAILABLE=1
else
    if [ "$ALLOW_CPU" = "1" ]; then
        echo "WARNING: NVCC not found, building CPU-only stub libraries (ALLOW_CPU=1)"
        echo "WARNING: These stubs will FAIL at runtime with error codes"
        
        cat > "$BUILD_DIR/stub_cuda.c" << 'EOF'
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define STUB_ERROR 999
#define STUB_NO_DEVICE 100
int cwInit(void) { fprintf(stderr, "ERROR: CUDA stub - no GPU available\n"); return STUB_NO_DEVICE; }
int cwSetDevice(int d) { return STUB_NO_DEVICE; }
int cwGetDevice(int* d) { *d = -1; return STUB_NO_DEVICE; }
int cwGetDeviceCount(int* c) { *c = 0; return 0; }
int cwDeviceSynchronize(void) { return STUB_NO_DEVICE; }
int cwMalloc(uint64_t* p, size_t s) { *p = (uint64_t)malloc(s); return *p ? 0 : 2; }
int cwFree(uint64_t p) { free((void*)p); return 0; }
int cwMemcpyH2D(uint64_t d, const void* s, size_t n) { memcpy((void*)d, s, n); return 0; }
int cwMemcpyD2H(void* d, uint64_t s, size_t n) { memcpy(d, (void*)s, n); return 0; }
int cwMemcpyD2D(uint64_t d, uint64_t s, size_t n) { memcpy((void*)d, (void*)s, n); return 0; }
int cwMemset(uint64_t p, int v, size_t n) { memset((void*)p, v, n); return 0; }
int cwStreamCreate(void** s) { *s = NULL; return 0; }
int cwStreamDestroy(void* s) { return 0; }
int cwStreamSynchronize(void* s) { return 0; }
int cwEventCreate(void** e) { *e = NULL; return 0; }
int cwEventDestroy(void* e) { return 0; }
int cwEventRecord(void* e, void* s) { return 0; }
int cwEventSynchronize(void* e) { return 0; }
int cwEventElapsedTime(float* m, void* a, void* b) { *m = 0; return 0; }
size_t cwGetFreeMemory(void) { return 0; }
size_t cwGetTotalMemory(void) { return 0; }
const char* cwGetErrorString(int e) { return e == STUB_NO_DEVICE ? "CPU stub - no GPU" : "CPU stub error"; }
EOF
        gcc -shared -fPIC -O2 "$BUILD_DIR/stub_cuda.c" -o "$BUILD_DIR/libcudawrap.so"
        echo "Built libcudawrap.so (CPU stub - will report errors)"
        
        cat > "$BUILD_DIR/stub_nccl.c" << 'EOF'
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#define STUB_ERROR 999
int ncwGetUniqueId(void* id) { fprintf(stderr, "ERROR: NCCL stub - no GPU\n"); return STUB_ERROR; }
int ncwCommInitRank(void** c, int n, void* id, int r) { *c = NULL; return STUB_ERROR; }
int ncwCommDestroy(void* c) { return 0; }
int ncwCommCount(void* c, int* n) { *n = 0; return STUB_ERROR; }
int ncwCommUserRank(void* c, int* r) { *r = -1; return STUB_ERROR; }
int ncwAllReduce(const void* s, void* r, size_t c, int d, int o, void* cm, void* st) { return STUB_ERROR; }
int ncwBroadcast(const void* s, void* r, size_t c, int d, int rt, void* cm, void* st) { return STUB_ERROR; }
int ncwReduce(const void* s, void* r, size_t c, int d, int o, int rt, void* cm, void* st) { return STUB_ERROR; }
int ncwAllGather(const void* s, void* r, size_t c, int d, void* cm, void* st) { return STUB_ERROR; }
int ncwReduceScatter(const void* s, void* r, size_t c, int d, int o, void* cm, void* st) { return STUB_ERROR; }
int ncwSend(const void* s, size_t c, int d, int p, void* cm, void* st) { return STUB_ERROR; }
int ncwRecv(void* r, size_t c, int d, int p, void* cm, void* st) { return STUB_ERROR; }
int ncwGroupStart(void) { return STUB_ERROR; }
int ncwGroupEnd(void) { return STUB_ERROR; }
const char* ncwGetErrorString(int r) { return "NCCL stub - no GPU"; }
EOF
        gcc -shared -fPIC -O2 "$BUILD_DIR/stub_nccl.c" -o "$BUILD_DIR/libncclwrap.so"
        echo "Built libncclwrap.so (CPU stub - will report errors)"
        
        cat > "$BUILD_DIR/stub_cublas.c" << 'EOF'
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#define STUB_ERROR 999
int cbwCreate(void** h) { fprintf(stderr, "ERROR: cuBLAS stub - no GPU\n"); *h = NULL; return STUB_ERROR; }
int cbwDestroy(void* h) { return 0; }
int cbwSetStream(void* h, void* s) { return STUB_ERROR; }
int cbwSgemm(void* h, int ta, int tb, int m, int n, int k, const float* a, const float* A, int la, const float* B, int lb, const float* b, float* C, int lc) { return STUB_ERROR; }
int cbwHgemm(void* h, int ta, int tb, int m, int n, int k, const void* a, const void* A, int la, const void* B, int lb, const void* b, void* C, int lc) { return STUB_ERROR; }
int cbwSgemmBatched(void* h, int ta, int tb, int m, int n, int k, const float* a, const float** A, int la, const float** B, int lb, const float* b, float** C, int lc, int bc) { return STUB_ERROR; }
int cbwSgemmStridedBatched(void* h, int ta, int tb, int m, int n, int k, const float* a, const float* A, int la, long long sa, const float* B, int lb, long long sb, const float* b, float* C, int lc, long long sc, int bc) { return STUB_ERROR; }
int cbwLtCreate(void** h) { *h = NULL; return STUB_ERROR; }
int cbwLtDestroy(void* h) { return 0; }
int cbwLtMatmulDescCreate(void** d, int ct, int st) { *d = NULL; return STUB_ERROR; }
int cbwLtMatmulDescDestroy(void* d) { return 0; }
int cbwLtMatrixLayoutCreate(void** l, int t, uint64_t r, uint64_t c, int64_t ld) { *l = NULL; return STUB_ERROR; }
int cbwLtMatrixLayoutDestroy(void* l) { return 0; }
int cbwLtMatmul(void* h, void* d, const void* a, const void* A, void* Ad, const void* B, void* Bd, const void* b, const void* C, void* Cd, void* D, void* Dd, void* s) { return STUB_ERROR; }
const char* cbwGetErrorString(int s) { return "cuBLAS stub - no GPU"; }
EOF
        gcc -shared -fPIC -O2 "$BUILD_DIR/stub_cublas.c" -o "$BUILD_DIR/libcublaswrap.so"
        echo "Built libcublaswrap.so (CPU stub - will report errors)"
        
        cat > "$BUILD_DIR/stub_kernels.c" << 'EOF'
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
void launch_fp8_dequantize(const void* i, void* o, size_t c, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_f32_to_fp8(const float* i, void* o, size_t c, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_rms_norm(const float* i, const float* g, float* o, int h, int b, float e, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_rope(float* q, float* k, int hd, int nh, int sl, int sp, float t, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_softmax(float* i, int b, int sl, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_swiglu(const float* g, const float* u, float* o, size_t c, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_gelu(float* d, size_t c, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_embedding_lookup(const void* e, const int64_t* t, float* o, int h, int n, int d, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_add_residual(float* a, const float* b, size_t c, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_argmax(const float* l, int64_t* o, int v, int b, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_top_p_sampling(const float* l, int64_t* o, float t, float p, int v, int b, const uint64_t* r, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_apply_rep_penalty(float* l, const int64_t* p, int pl, int v, float pen, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_paged_attention_prefill(const float* q, const float* k, const float* v, float* o, const int* pt, int np, int ps, int nh, int hd, int sl, float sc, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_paged_attention_decode(const float* q, const float* kk, const float* kv, float* o, const int* pt, int np, int ps, int nh, int nkh, int hd, int pl, float sc, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_update_kv_cache(const float* k, const float* v, float* kk, float* kv, const int* pt, int ps, int nkh, int hd, int sl, int sp, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_moe_gate(const float* h, const void* gw, float* rl, int hd, int ne, int bs, int gd, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_moe_topk(const float* rl, int* ei, float* ew, int ne, int tk, int bs, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_moe_dispatch(const float* i, float* eo, const int* ei, int hd, int tk, int bs, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_moe_combine(const float* eo, const float* ew, float* o, int hd, int tk, int bs, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
void launch_copy_hidden(const float* src, float* dst, size_t c, void* s) { fprintf(stderr, "ERROR: GPU kernel stub called\n"); exit(1); }
EOF
        gcc -shared -fPIC -O2 "$BUILD_DIR/stub_kernels.c" -o "$BUILD_DIR/libkernels.so"
        echo "Built libkernels.so (CPU stub - will ABORT if called)"
        
        CUDA_AVAILABLE=0
    else
        echo ""
        echo "ERROR: NVCC not found and ALLOW_CPU=0 (default)"
        echo "Production builds REQUIRE CUDA. Set ALLOW_CPU=1 for development only."
        echo ""
        exit 1
    fi
fi

rm -f "$BUILD_DIR/stub_*.c" 2>/dev/null || true

FUTHARK_SRC="${SRC_DIR}/futhark/kernels.fut"
FUTHARK_OUT="${BUILD_DIR}/kernels"
if command -v futhark &> /dev/null; then
    if [ -f "${FUTHARK_SRC}" ]; then
        echo "Building Futhark kernels..."
        if futhark cuda --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark CUDA build complete"
        elif futhark opencl --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark OpenCL build complete"
        elif futhark multicore --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark multicore build complete"
        elif futhark c --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark C build complete"
        else
            echo "WARNING: Futhark build failed (optional component)"
        fi
    else
        echo "Futhark source not found: ${FUTHARK_SRC}"
    fi
else
    echo "Futhark not installed (optional component, skipping)"
fi

TERRA_SRC="${SRC_DIR}/terra/engine.t"
if command -v terra &> /dev/null; then
    if [ -f "${TERRA_SRC}" ]; then
        echo ""
        echo "=== Building Terra Engine ==="
        cd "${BUILD_DIR}"
        export LD_LIBRARY_PATH="${BUILD_DIR}:${LD_LIBRARY_PATH}"
        
        if terra "${TERRA_SRC}" 2>&1; then
            echo "Terra build complete"
        else
            echo ""
            echo "ERROR: Terra build failed"
            echo ""
            exit 1
        fi
        cd "${PROJECT_DIR}"
    else
        echo ""
        echo "ERROR: Terra source not found: ${TERRA_SRC}"
        echo ""
        exit 1
    fi
else
    echo ""
    echo "ERROR: Terra not installed"
    echo "Install Terra from: https://terralang.org/"
    echo ""
    exit 1
fi

if [ ! -f "$BUILD_DIR/engine.so" ]; then
    echo ""
    echo "ERROR: engine.so was not created"
    echo "Build failed - engine.so is required for production"
    echo ""
    exit 1
fi

echo ""
echo "=== Verifying engine.so dependencies ==="
if command -v ldd &> /dev/null; then
    MISSING_DEPS=$(ldd "$BUILD_DIR/engine.so" 2>&1 | grep "not found" || true)
    if [ -n "$MISSING_DEPS" ]; then
        echo "ERROR: engine.so has missing dependencies:"
        echo "$MISSING_DEPS"
        exit 1
    fi
    echo "All dependencies found"
fi

echo ""
echo "=== Build Complete ==="
echo "Libraries in $BUILD_DIR:"
ls -la "$BUILD_DIR"/*.so 2>/dev/null || echo "No .so files found"

if [ "$CUDA_AVAILABLE" = "1" ]; then
    echo ""
    echo "SUCCESS: GPU build complete"
    echo "engine.so and CUDA libraries ready for production"
else
    echo ""
    echo "WARNING: CPU stub build complete (ALLOW_CPU=1)"
    echo "NOT suitable for production - stubs will fail at runtime"
fi
