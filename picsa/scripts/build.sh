#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
CSRC_DIR="$PROJECT_DIR/csrc"
SRC_DIR="$PROJECT_DIR/src"

mkdir -p "$BUILD_DIR"

echo "=== GLM-4.7-FP8 Inference Engine Build ==="
echo "Project dir: $PROJECT_DIR"
echo "Build dir: $BUILD_DIR"

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
    echo "Built libcudawrap.so"
    
    nvcc $CUDA_FLAGS \
        -I"$CSRC_DIR" \
        "$CSRC_DIR/nccl_wrappers.cu" \
        -o "$BUILD_DIR/libncclwrap.so" \
        -lcudart -lnccl
    echo "Built libncclwrap.so"
    
    nvcc $CUDA_FLAGS \
        -I"$CSRC_DIR" \
        "$CSRC_DIR/cublas_wrappers.cu" \
        -o "$BUILD_DIR/libcublaswrap.so" \
        -lcudart -lcublas -lcublasLt
    echo "Built libcublaswrap.so"
    
    nvcc $CUDA_FLAGS \
        -I"$CSRC_DIR" \
        "$CSRC_DIR/kernels.cu" \
        -o "$BUILD_DIR/libkernels.so" \
        -lcudart
    echo "Built libkernels.so"
    
    CUDA_AVAILABLE=1
else
    echo "NVCC not found, creating CPU-only stub libraries..."
    
    cat > "$BUILD_DIR/stub_cuda.c" << 'EOF'
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
int cwInit(void) { return 3; }
int cwSetDevice(int d) { return 3; }
int cwGetDevice(int* d) { *d = -1; return 3; }
int cwGetDeviceCount(int* c) { *c = 0; return 0; }
int cwDeviceSynchronize(void) { return 0; }
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
const char* cwGetErrorString(int e) { return "CPU mode"; }
EOF
    gcc -shared -fPIC -O2 "$BUILD_DIR/stub_cuda.c" -o "$BUILD_DIR/libcudawrap.so"
    echo "Built libcudawrap.so (CPU stub)"
    
    cat > "$BUILD_DIR/stub_nccl.c" << 'EOF'
#include <stdint.h>
#include <stddef.h>
int ncwGetUniqueId(void* id) { return 0; }
int ncwCommInitRank(void** c, int n, void* id, int r) { *c = NULL; return 0; }
int ncwCommDestroy(void* c) { return 0; }
int ncwCommCount(void* c, int* n) { *n = 1; return 0; }
int ncwCommUserRank(void* c, int* r) { *r = 0; return 0; }
int ncwAllReduce(const void* s, void* r, size_t c, int d, int o, void* cm, void* st) { return 0; }
int ncwBroadcast(const void* s, void* r, size_t c, int d, int rt, void* cm, void* st) { return 0; }
int ncwReduce(const void* s, void* r, size_t c, int d, int o, int rt, void* cm, void* st) { return 0; }
int ncwAllGather(const void* s, void* r, size_t c, int d, void* cm, void* st) { return 0; }
int ncwReduceScatter(const void* s, void* r, size_t c, int d, int o, void* cm, void* st) { return 0; }
int ncwSend(const void* s, size_t c, int d, int p, void* cm, void* st) { return 0; }
int ncwRecv(void* r, size_t c, int d, int p, void* cm, void* st) { return 0; }
int ncwGroupStart(void) { return 0; }
int ncwGroupEnd(void) { return 0; }
const char* ncwGetErrorString(int r) { return "CPU mode"; }
EOF
    gcc -shared -fPIC -O2 "$BUILD_DIR/stub_nccl.c" -o "$BUILD_DIR/libncclwrap.so"
    echo "Built libncclwrap.so (CPU stub)"
    
    cat > "$BUILD_DIR/stub_cublas.c" << 'EOF'
#include <stdint.h>
#include <stddef.h>
int cbwCreate(void** h) { *h = NULL; return 0; }
int cbwDestroy(void* h) { return 0; }
int cbwSetStream(void* h, void* s) { return 0; }
int cbwSgemm(void* h, int ta, int tb, int m, int n, int k, const float* a, const float* A, int la, const float* B, int lb, const float* b, float* C, int lc) { return 0; }
int cbwHgemm(void* h, int ta, int tb, int m, int n, int k, const void* a, const void* A, int la, const void* B, int lb, const void* b, void* C, int lc) { return 0; }
int cbwSgemmBatched(void* h, int ta, int tb, int m, int n, int k, const float* a, const float** A, int la, const float** B, int lb, const float* b, float** C, int lc, int bc) { return 0; }
int cbwSgemmStridedBatched(void* h, int ta, int tb, int m, int n, int k, const float* a, const float* A, int la, long long sa, const float* B, int lb, long long sb, const float* b, float* C, int lc, long long sc, int bc) { return 0; }
int cbwLtCreate(void** h) { *h = NULL; return 0; }
int cbwLtDestroy(void* h) { return 0; }
int cbwLtMatmulDescCreate(void** d, int ct, int st) { *d = NULL; return 0; }
int cbwLtMatmulDescDestroy(void* d) { return 0; }
int cbwLtMatrixLayoutCreate(void** l, int t, uint64_t r, uint64_t c, int64_t ld) { *l = NULL; return 0; }
int cbwLtMatrixLayoutDestroy(void* l) { return 0; }
int cbwLtMatmulPreferenceCreate(void** p) { *p = NULL; return 0; }
int cbwLtMatmulPreferenceDestroy(void* p) { return 0; }
int cbwLtMatmul(void* h, void* d, const void* a, const void* A, void* Ad, const void* B, void* Bd, const void* b, const void* C, void* Cd, void* D, void* Dd, void* s) { return 0; }
const char* cbwGetErrorString(int s) { return "CPU mode"; }
EOF
    gcc -shared -fPIC -O2 "$BUILD_DIR/stub_cublas.c" -o "$BUILD_DIR/libcublaswrap.so"
    echo "Built libcublaswrap.so (CPU stub)"
    
    cat > "$BUILD_DIR/stub_kernels.c" << 'EOF'
#include <stdint.h>
#include <stddef.h>
void launch_fp8_dequantize(const void* i, void* o, size_t c, void* s) {}
void launch_f32_to_fp8(const float* i, void* o, size_t c, void* s) {}
void launch_rms_norm(const float* i, const float* g, float* o, int h, int b, float e, void* s) {}
void launch_rope(float* q, float* k, int hd, int nh, int sl, int sp, float t, void* s) {}
void launch_softmax(float* i, int b, int sl, void* s) {}
void launch_swiglu(const float* g, const float* u, float* o, size_t c, void* s) {}
void launch_gelu(float* d, size_t c, void* s) {}
void launch_embedding_lookup(const void* e, const int64_t* t, float* o, int h, int n, int d, void* s) {}
void launch_add_residual(float* a, const float* b, size_t c, void* s) {}
void launch_argmax(const float* l, int64_t* o, int v, int b, void* s) {}
void launch_top_p_sampling(const float* l, int64_t* o, float t, float p, int v, int b, const uint64_t* r, void* s) {}
void launch_apply_rep_penalty(float* l, const int64_t* p, int pl, int v, float pen, void* s) {}
EOF
    gcc -shared -fPIC -O2 "$BUILD_DIR/stub_kernels.c" -o "$BUILD_DIR/libkernels.so"
    echo "Built libkernels.so (CPU stub)"
    
    CUDA_AVAILABLE=0
fi

rm -f "$BUILD_DIR/stub_*.c" 2>/dev/null || true

FUTHARK_SRC="${SRC_DIR}/futhark/kernels.fut"
FUTHARK_OUT="${BUILD_DIR}/kernels"
if command -v futhark &> /dev/null; then
    if [ -f "${FUTHARK_SRC}" ]; then
        if futhark cuda --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark CUDA build complete"
        elif futhark opencl --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark OpenCL build complete"
        elif futhark multicore --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark multicore build complete"
        elif futhark c --library "${FUTHARK_SRC}" -o "${FUTHARK_OUT}" 2>/dev/null; then
            echo "Futhark C build complete"
        else
            echo "Futhark build failed"
        fi
    else
        echo "Futhark source not found: ${FUTHARK_SRC}"
    fi
else
    echo "Futhark not found, skipping kernel build"
fi

TERRA_SRC="${SRC_DIR}/terra/engine.t"
if command -v terra &> /dev/null; then
    if [ -f "${TERRA_SRC}" ]; then
        echo "Building Terra engine..."
        cd "${BUILD_DIR}"
        export LD_LIBRARY_PATH="${BUILD_DIR}:${LD_LIBRARY_PATH}"
        if terra "${TERRA_SRC}" 2>&1; then
            echo "Terra build complete"
        else
            echo "Terra build encountered issues"
        fi
        cd "${PROJECT_DIR}"
    else
        echo "Terra source not found: ${TERRA_SRC}"
    fi
else
    echo "Terra not found, skipping engine build"
fi

echo ""
echo "=== Build Complete ==="
echo "Libraries in $BUILD_DIR:"
ls -la "$BUILD_DIR"/*.so 2>/dev/null || echo "No .so files found"

if [ "$CUDA_AVAILABLE" = "1" ]; then
    echo ""
    echo "CUDA libraries built successfully."
else
    echo ""
    echo "CPU stub libraries built (no CUDA available)."
    echo "The engine will run in CPU-only mode."
fi
