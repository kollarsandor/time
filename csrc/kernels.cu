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
        val = nanf("");
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

void launch_argmax(const float* logits, int64_t* output, int vocab_size, int batch_size, cwStream_t stream) {
    kernel_argmax<<<batch_size, 1, 0, (cudaStream_t)stream>>>(logits, output, vocab_size, batch_size);
}

void launch_top_p_sampling(const float* logits, int64_t* output, float temperature, float top_p, int vocab_size, int batch_size, const uint64_t* random_seeds, cwStream_t stream) {
    kernel_argmax<<<batch_size, 1, 0, (cudaStream_t)stream>>>(logits, output, vocab_size, batch_size);
}

void launch_apply_rep_penalty(float* logits, const int64_t* past_tokens, int past_len, int vocab_size, float penalty, cwStream_t stream) {
    kernel_apply_rep_penalty<<<1, 256, 0, (cudaStream_t)stream>>>(logits, past_tokens, past_len, vocab_size, penalty);
}

}
