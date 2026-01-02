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
