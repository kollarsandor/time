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
