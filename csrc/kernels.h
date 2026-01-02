#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>
#include <stddef.h>
#include "cuda_wrappers.h"

#ifdef __cplusplus
extern "C" {
#endif

void launch_fp8_dequantize(const void* input, void* output, size_t count, cwStream_t stream);
void launch_f32_to_fp8(const float* input, void* output, size_t count, cwStream_t stream);
void launch_rms_norm(const float* input, const float* gamma, float* output, int hidden_dim, int batch_size, float eps, cwStream_t stream);
void launch_rope(float* q, float* k, int head_dim, int num_heads, int seq_len, int start_pos, float theta, cwStream_t stream);
void launch_softmax(float* input, int batch_size, int seq_len, cwStream_t stream);
void launch_swiglu(const float* gate, const float* up, float* output, size_t count, cwStream_t stream);
void launch_gelu(float* data, size_t count, cwStream_t stream);
void launch_embedding_lookup(const void* embedding_table, const int64_t* token_ids, float* output, int hidden_dim, int num_tokens, int dtype, cwStream_t stream);
void launch_add_residual(float* a, const float* b, size_t count, cwStream_t stream);
void launch_argmax(const float* logits, int64_t* output, int vocab_size, int batch_size, cwStream_t stream);
void launch_top_p_sampling(const float* logits, int64_t* output, float temperature, float top_p, int vocab_size, int batch_size, const uint64_t* random_seeds, cwStream_t stream);
void launch_apply_rep_penalty(float* logits, const int64_t* past_tokens, int past_len, int vocab_size, float penalty, cwStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
