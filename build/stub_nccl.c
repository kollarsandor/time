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
