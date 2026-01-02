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
