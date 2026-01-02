#include "kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

__device__ float fp8_e4m3_to_float(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp = (bits >> 3) & 0xF;
    int mant = bits & 0x7;
    float val;
    if (exp == 0) {
        if (mant == 0) {
            val = 0.0f;
        } else {
            val = ((float)mant) * powf(2.0f, -9.0f);
        }
    } else if (exp == 15 && mant == 7) {
        val = 0.0f;
    } else {
        int e = exp - 7;
        float m = 1.0f + ((float)mant) / 8.0f;
        val = m * powf(2.0f, (float)e);
    }
    return sign ? -val : val;
}

__device__ uint8_t float_to_fp8_e4m3(float x) {
    int sign = (x < 0.0f) ? 1 : 0;
    float ax = fabsf(x);
    if (isnan(x)) return 0x7F;
    if (ax == 0.0f) return (uint8_t)(sign << 7);
    if (ax >= 448.0f) return (uint8_t)((sign << 7) | 0x7E);
    if (ax < powf(2.0f, -9.0f)) return (uint8_t)(sign << 7);
    float log2_ax = log2f(ax);
    int e = (int)floorf(log2_ax);
    if (e < -6) e = -6;
    if (e > 8) e = 8;
    int exp_bits = e + 7;
    float m = ax / powf(2.0f, (float)e) - 1.0f;
    int mant = (int)roundf(m * 8.0f);
    if (mant < 0) mant = 0;
    if (mant > 7) mant = 7;
    return (uint8_t)((sign << 7) | (exp_bits << 3) | mant);
}

__global__ void kernel_fp8_dequantize(const uint8_t* input, float* output, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = fp8_e4m3_to_float(input[idx]);
    }
}

__global__ void kernel_f32_to_fp8(const float* input, uint8_t* output, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = float_to_fp8_e4m3(input[idx]);
    }
}

__global__ void kernel_rms_norm(const float* input, const float* gamma, float* output, int hidden_dim, float eps) {
    int batch_idx = blockIdx.x;
    const float* x = input + batch_idx * hidden_dim;
    float* out = output + batch_idx * hidden_dim;
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        local_sum += x[i] * x[i];
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float sq_sum = shared_sum[0];
    __syncthreads();
    float rms = sqrtf(sq_sum / (float)hidden_dim + eps);
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        out[i] = (x[i] / rms) * gamma[i];
    }
}

__global__ void kernel_rope(float* q, float* k, int head_dim, int num_heads, int seq_len, int start_pos, float theta) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int pos = start_pos + seq_idx;
    float* q_head = q + (seq_idx * num_heads + head_idx) * head_dim;
    float* k_head = k + (seq_idx * num_heads + head_idx) * head_dim;
    int half = head_dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float angle = freq * (float)pos;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        float q0 = q_head[i];
        float q1 = q_head[i + half];
        q_head[i] = q0 * cos_val - q1 * sin_val;
        q_head[i + half] = q0 * sin_val + q1 * cos_val;
        float k0 = k_head[i];
        float k1 = k_head[i + half];
        k_head[i] = k0 * cos_val - k1 * sin_val;
        k_head[i + half] = k0 * sin_val + k1 * cos_val;
    }
}

__global__ void kernel_softmax(float* input, int seq_len) {
    int batch_idx = blockIdx.x;
    float* row = input + batch_idx * seq_len;
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        if (row[i] > local_max) local_max = row[i];
    }
    shared_max[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
            }
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float e = expf(row[i] - max_val);
        row[i] = e;
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = shared_sum[0];
    __syncthreads();
    for (int i = tid; i < seq_len; i += blockDim.x) {
        row[i] /= sum_val;
    }
}

__global__ void kernel_swiglu(const float* gate, const float* up, float* output, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float g = gate[idx];
        float silu = g / (1.0f + expf(-g));
        output[idx] = silu * up[idx];
    }
}

__global__ void kernel_gelu(float* data, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float x = data[idx];
        data[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void kernel_embedding_lookup_fp8(const uint8_t* embedding_table, const int64_t* token_ids, float* output, int hidden_dim, int num_tokens) {
    int token_idx = blockIdx.x;
    if (token_idx < num_tokens) {
        int64_t token_id = token_ids[token_idx];
        const uint8_t* embed_row = embedding_table + token_id * hidden_dim;
        float* out_row = output + token_idx * hidden_dim;
        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            out_row[i] = fp8_e4m3_to_float(embed_row[i]);
        }
    }
}

__global__ void kernel_embedding_lookup_fp16(const __half* embedding_table, const int64_t* token_ids, float* output, int hidden_dim, int num_tokens) {
    int token_idx = blockIdx.x;
    if (token_idx < num_tokens) {
        int64_t token_id = token_ids[token_idx];
        const __half* embed_row = embedding_table + token_id * hidden_dim;
        float* out_row = output + token_idx * hidden_dim;
        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            out_row[i] = __half2float(embed_row[i]);
        }
    }
}

__global__ void kernel_embedding_lookup_f32(const float* embedding_table, const int64_t* token_ids, float* output, int hidden_dim, int num_tokens) {
    int token_idx = blockIdx.x;
    if (token_idx < num_tokens) {
        int64_t token_id = token_ids[token_idx];
        const float* embed_row = embedding_table + token_id * hidden_dim;
        float* out_row = output + token_idx * hidden_dim;
        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            out_row[i] = embed_row[i];
        }
    }
}

__global__ void kernel_add_residual(float* a, const float* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        a[idx] += b[idx];
    }
}

__global__ void kernel_copy(const float* src, float* dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

__global__ void kernel_argmax(const float* logits, int64_t* output, int vocab_size, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx < batch_size) {
        const float* row = logits + batch_idx * vocab_size;
        float max_val = row[0];
        int max_idx = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (row[i] > max_val) {
                max_val = row[i];
                max_idx = i;
            }
        }
        output[batch_idx] = max_idx;
    }
}

__global__ void kernel_top_p_sample(const float* logits, int64_t* output, float temperature, float top_p, int vocab_size, int batch_size, const uint64_t* rand_seeds) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    const float* row = logits + batch_idx * vocab_size;
    uint64_t seed = rand_seeds[batch_idx];
    float max_val = row[0];
    for (int i = 1; i < vocab_size; i++) {
        if (row[i] > max_val) max_val = row[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += expf((row[i] - max_val) / temperature);
    }
    float threshold = top_p * sum_exp;
    float cumsum = 0.0f;
    int selected = 0;
    float rand_val = ((float)(seed & 0xFFFFFFFF)) / 4294967296.0f;
    float target = rand_val * threshold;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += expf((row[i] - max_val) / temperature);
        if (cumsum >= target) {
            selected = i;
            break;
        }
    }
    output[batch_idx] = selected;
}

__global__ void kernel_apply_rep_penalty(float* logits, const int64_t* past_tokens, int past_len, int vocab_size, float penalty) {
    for (int i = threadIdx.x; i < past_len; i += blockDim.x) {
        int64_t tid = past_tokens[i];
        if (tid >= 0 && tid < vocab_size) {
            float val = logits[tid];
            if (val > 0.0f) {
                logits[tid] = val / penalty;
            } else {
                logits[tid] = val * penalty;
            }
        }
    }
}

__global__ void kernel_paged_attention_prefill(const float* q, const float* k, const float* v, float* output, int num_heads, int head_dim, int seq_len, float scale) {
    int head_idx = blockIdx.x;
    int seq_i = blockIdx.y;
    if (head_idx >= num_heads || seq_i >= seq_len) return;
    __shared__ float scores[1024];
    __shared__ float max_score[1];
    __shared__ float sum_exp[1];
    const float* q_head = q + (seq_i * num_heads + head_idx) * head_dim;
    float* out_head = output + (seq_i * num_heads + head_idx) * head_dim;
    int tid = threadIdx.x;
    if (tid == 0) {
        max_score[0] = -FLT_MAX;
        sum_exp[0] = 0.0f;
    }
    __syncthreads();
    for (int j = tid; j <= seq_i && j < 1024; j += blockDim.x) {
        const float* k_head = k + (j * num_heads + head_idx) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_head[d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();
    float local_max = -FLT_MAX;
    for (int j = tid; j <= seq_i && j < 1024; j += blockDim.x) {
        if (scores[j] > local_max) local_max = scores[j];
    }
    atomicMax((int*)max_score, __float_as_int(local_max));
    __syncthreads();
    float m = max_score[0];
    float local_sum = 0.0f;
    for (int j = tid; j <= seq_i && j < 1024; j += blockDim.x) {
        float e = expf(scores[j] - m);
        scores[j] = e;
        local_sum += e;
    }
    atomicAdd(sum_exp, local_sum);
    __syncthreads();
    float s = sum_exp[0];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j <= seq_i && j < 1024; j++) {
            const float* v_head = v + (j * num_heads + head_idx) * head_dim;
            acc += (scores[j] / s) * v_head[d];
        }
        out_head[d] = acc;
    }
}

__global__ void kernel_paged_attention_decode(const float* q, const float* kv_cache_k, const float* kv_cache_v, float* output, const int32_t* page_table, int num_pages, int page_size, int num_heads, int num_kv_heads, int head_dim, int past_len, float scale) {
    int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const float* q_head = q + head_idx * head_dim;
    float* out_head = output + head_idx * head_dim;
    __shared__ float max_score[1];
    __shared__ float sum_exp[1];
    int tid = threadIdx.x;
    if (tid == 0) {
        max_score[0] = -FLT_MAX;
        sum_exp[0] = 0.0f;
    }
    __syncthreads();
    float local_max = -FLT_MAX;
    for (int pos = tid; pos < past_len; pos += blockDim.x) {
        int page_idx = pos / page_size;
        int page_offset = pos % page_size;
        int physical_page = page_table[page_idx];
        const float* k_ptr = kv_cache_k + ((int64_t)physical_page * page_size + page_offset) * num_kv_heads * head_dim + kv_head_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_ptr[d];
        }
        dot *= scale;
        if (dot > local_max) local_max = dot;
    }
    atomicMax((int*)max_score, __float_as_int(local_max));
    __syncthreads();
    float m = max_score[0];
    float local_sum = 0.0f;
    for (int pos = tid; pos < past_len; pos += blockDim.x) {
        int page_idx = pos / page_size;
        int page_offset = pos % page_size;
        int physical_page = page_table[page_idx];
        const float* k_ptr = kv_cache_k + ((int64_t)physical_page * page_size + page_offset) * num_kv_heads * head_dim + kv_head_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_ptr[d];
        }
        dot *= scale;
        local_sum += expf(dot - m);
    }
    atomicAdd(sum_exp, local_sum);
    __syncthreads();
    float s = sum_exp[0];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < past_len; pos++) {
            int page_idx = pos / page_size;
            int page_offset = pos % page_size;
            int physical_page = page_table[page_idx];
            const float* k_ptr = kv_cache_k + ((int64_t)physical_page * page_size + page_offset) * num_kv_heads * head_dim + kv_head_idx * head_dim;
            const float* v_ptr = kv_cache_v + ((int64_t)physical_page * page_size + page_offset) * num_kv_heads * head_dim + kv_head_idx * head_dim;
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += q_head[dd] * k_ptr[dd];
            }
            dot *= scale;
            float weight = expf(dot - m) / s;
            acc += weight * v_ptr[d];
        }
        out_head[d] = acc;
    }
}

__global__ void kernel_update_kv_cache(const float* k, const float* v, float* kv_cache_k, float* kv_cache_v, const int32_t* page_table, int page_size, int num_kv_heads, int head_dim, int seq_len, int start_pos) {
    int seq_idx = blockIdx.x;
    int kv_head_idx = blockIdx.y;
    if (seq_idx >= seq_len || kv_head_idx >= num_kv_heads) return;
    int pos = start_pos + seq_idx;
    int page_idx = pos / page_size;
    int page_offset = pos % page_size;
    int physical_page = page_table[page_idx];
    const float* k_src = k + (seq_idx * num_kv_heads + kv_head_idx) * head_dim;
    const float* v_src = v + (seq_idx * num_kv_heads + kv_head_idx) * head_dim;
    float* k_dst = kv_cache_k + ((int64_t)physical_page * page_size + page_offset) * num_kv_heads * head_dim + kv_head_idx * head_dim;
    float* v_dst = kv_cache_v + ((int64_t)physical_page * page_size + page_offset) * num_kv_heads * head_dim + kv_head_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        k_dst[d] = k_src[d];
        v_dst[d] = v_src[d];
    }
}

__global__ void kernel_moe_gate(const float* hidden, const float* gate_weight, float* router_logits, int hidden_dim, int num_experts, int batch_size) {
    int batch_idx = blockIdx.x;
    int expert_idx = blockIdx.y;
    if (batch_idx >= batch_size || expert_idx >= num_experts) return;
    const float* h = hidden + batch_idx * hidden_dim;
    const float* w = gate_weight + expert_idx * hidden_dim;
    __shared__ float partial[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        sum += h[i] * w[i];
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        router_logits[batch_idx * num_experts + expert_idx] = partial[0];
    }
}

__global__ void kernel_moe_topk(const float* router_logits, int32_t* expert_indices, float* expert_weights, int num_experts, int top_k, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    const float* logits = router_logits + batch_idx * num_experts;
    int32_t* indices = expert_indices + batch_idx * top_k;
    float* weights = expert_weights + batch_idx * top_k;
    float max_vals[16];
    int max_indices[16];
    for (int k = 0; k < top_k && k < 16; k++) {
        max_vals[k] = -FLT_MAX;
        max_indices[k] = -1;
    }
    for (int e = 0; e < num_experts; e++) {
        float val = logits[e];
        for (int k = 0; k < top_k && k < 16; k++) {
            if (val > max_vals[k]) {
                for (int j = top_k - 1; j > k && j < 16; j--) {
                    max_vals[j] = max_vals[j - 1];
                    max_indices[j] = max_indices[j - 1];
                }
                max_vals[k] = val;
                max_indices[k] = e;
                break;
            }
        }
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < top_k && k < 16; k++) {
        sum_exp += expf(max_vals[k]);
    }
    for (int k = 0; k < top_k && k < 16; k++) {
        indices[k] = max_indices[k];
        weights[k] = expf(max_vals[k]) / sum_exp;
    }
}

__global__ void kernel_moe_dispatch(const float* input, float* expert_inputs, const int32_t* expert_indices, int hidden_dim, int top_k, int batch_size) {
    int batch_idx = blockIdx.x;
    int k_idx = blockIdx.y;
    if (batch_idx >= batch_size || k_idx >= top_k) return;
    const float* src = input + batch_idx * hidden_dim;
    float* dst = expert_inputs + (batch_idx * top_k + k_idx) * hidden_dim;
    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        dst[d] = src[d];
    }
}

__global__ void kernel_moe_combine(const float* expert_outputs, const float* expert_weights, float* output, int hidden_dim, int top_k, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    float* out = output + batch_idx * hidden_dim;
    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; k++) {
            const float* exp_out = expert_outputs + (batch_idx * top_k + k) * hidden_dim;
            acc += exp_out[d] * expert_weights[batch_idx * top_k + k];
        }
        out[d] = acc;
    }
}

extern "C" {

void launch_fp8_dequantize(const void* input, void* output, size_t count, cwStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_fp8_dequantize<<<blocks, threads, 0, (cudaStream_t)stream>>>((const uint8_t*)input, (float*)output, count);
}

void launch_f32_to_fp8(const float* input, void* output, size_t count, cwStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_f32_to_fp8<<<blocks, threads, 0, (cudaStream_t)stream>>>(input, (uint8_t*)output, count);
}

void launch_rms_norm(const float* input, const float* gamma, float* output, int hidden_dim, int batch_size, float eps, cwStream_t stream) {
    kernel_rms_norm<<<batch_size, 256, 0, (cudaStream_t)stream>>>(input, gamma, output, hidden_dim, eps);
}

void launch_rope(float* q, float* k, int head_dim, int num_heads, int seq_len, int start_pos, float theta, cwStream_t stream) {
    dim3 grid(seq_len, num_heads);
    kernel_rope<<<grid, 128, 0, (cudaStream_t)stream>>>(q, k, head_dim, num_heads, seq_len, start_pos, theta);
}

void launch_softmax(float* input, int batch_size, int seq_len, cwStream_t stream) {
    kernel_softmax<<<batch_size, 256, 0, (cudaStream_t)stream>>>(input, seq_len);
}

void launch_swiglu(const float* gate, const float* up, float* output, size_t count, cwStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_swiglu<<<blocks, threads, 0, (cudaStream_t)stream>>>(gate, up, output, count);
}

void launch_gelu(float* data, size_t count, cwStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_gelu<<<blocks, threads, 0, (cudaStream_t)stream>>>(data, count);
}

void launch_embedding_lookup(const void* embedding_table, const int64_t* token_ids, float* output, int hidden_dim, int num_tokens, int dtype, cwStream_t stream) {
    if (dtype == 1) {
        kernel_embedding_lookup_fp8<<<num_tokens, 256, 0, (cudaStream_t)stream>>>((const uint8_t*)embedding_table, token_ids, output, hidden_dim, num_tokens);
    } else if (dtype == 2) {
        kernel_embedding_lookup_fp16<<<num_tokens, 256, 0, (cudaStream_t)stream>>>((const __half*)embedding_table, token_ids, output, hidden_dim, num_tokens);
    } else {
        kernel_embedding_lookup_f32<<<num_tokens, 256, 0, (cudaStream_t)stream>>>((const float*)embedding_table, token_ids, output, hidden_dim, num_tokens);
    }
}

void launch_add_residual(float* a, const float* b, size_t count, cwStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_add_residual<<<blocks, threads, 0, (cudaStream_t)stream>>>(a, b, count);
}

void launch_copy_hidden(const float* src, float* dst, size_t count, cwStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_copy<<<blocks, threads, 0, (cudaStream_t)stream>>>(src, dst, count);
}

void launch_argmax(const float* logits, int64_t* output, int vocab_size, int batch_size, cwStream_t stream) {
    kernel_argmax<<<batch_size, 1, 0, (cudaStream_t)stream>>>(logits, output, vocab_size, batch_size);
}

void launch_top_p_sampling(const float* logits, int64_t* output, float temperature, float top_p, int vocab_size, int batch_size, const uint64_t* random_seeds, cwStream_t stream) {
    if (temperature < 0.01f) {
        kernel_argmax<<<batch_size, 1, 0, (cudaStream_t)stream>>>(logits, output, vocab_size, batch_size);
    } else {
        kernel_top_p_sample<<<batch_size, 1, 0, (cudaStream_t)stream>>>(logits, output, temperature, top_p, vocab_size, batch_size, random_seeds);
    }
}

void launch_apply_rep_penalty(float* logits, const int64_t* past_tokens, int past_len, int vocab_size, float penalty, cwStream_t stream) {
    kernel_apply_rep_penalty<<<1, 256, 0, (cudaStream_t)stream>>>(logits, past_tokens, past_len, vocab_size, penalty);
}

void launch_paged_attention_prefill(const float* q, const float* k, const float* v, float* output, const int32_t* page_table, int num_pages, int page_size, int num_heads, int head_dim, int seq_len, float scale, cwStream_t stream) {
    dim3 grid(num_heads, seq_len);
    kernel_paged_attention_prefill<<<grid, 128, 0, (cudaStream_t)stream>>>(q, k, v, output, num_heads, head_dim, seq_len, scale);
}

void launch_paged_attention_decode(const float* q, const float* kv_cache_k, const float* kv_cache_v, float* output, const int32_t* page_table, int num_pages, int page_size, int num_heads, int num_kv_heads, int head_dim, int past_len, float scale, cwStream_t stream) {
    kernel_paged_attention_decode<<<num_heads, 128, 0, (cudaStream_t)stream>>>(q, kv_cache_k, kv_cache_v, output, page_table, num_pages, page_size, num_heads, num_kv_heads, head_dim, past_len, scale);
}

void launch_update_kv_cache(const float* k, const float* v, float* kv_cache_k, float* kv_cache_v, const int32_t* page_table, int page_size, int num_kv_heads, int head_dim, int seq_len, int start_pos, cwStream_t stream) {
    dim3 grid(seq_len, num_kv_heads);
    kernel_update_kv_cache<<<grid, 128, 0, (cudaStream_t)stream>>>(k, v, kv_cache_k, kv_cache_v, page_table, page_size, num_kv_heads, head_dim, seq_len, start_pos);
}

void launch_moe_gate(const float* hidden, const void* gate_weight, float* router_logits, int hidden_dim, int num_experts, int batch_size, int gate_dtype, cwStream_t stream) {
    dim3 grid(batch_size, num_experts);
    kernel_moe_gate<<<grid, 256, 0, (cudaStream_t)stream>>>(hidden, (const float*)gate_weight, router_logits, hidden_dim, num_experts, batch_size);
}

void launch_moe_topk(const float* router_logits, int32_t* expert_indices, float* expert_weights, int num_experts, int top_k, int batch_size, cwStream_t stream) {
    kernel_moe_topk<<<batch_size, 1, 0, (cudaStream_t)stream>>>(router_logits, expert_indices, expert_weights, num_experts, top_k, batch_size);
}

void launch_moe_dispatch(const float* input, float* expert_inputs, const int32_t* expert_indices, int hidden_dim, int top_k, int batch_size, cwStream_t stream) {
    dim3 grid(batch_size, top_k);
    kernel_moe_dispatch<<<grid, 256, 0, (cudaStream_t)stream>>>(input, expert_inputs, expert_indices, hidden_dim, top_k, batch_size);
}

void launch_moe_combine(const float* expert_outputs, const float* expert_weights, float* output, int hidden_dim, int top_k, int batch_size, cwStream_t stream) {
    kernel_moe_combine<<<batch_size, 256, 0, (cudaStream_t)stream>>>(expert_outputs, expert_weights, output, hidden_dim, top_k, batch_size);
}

}
