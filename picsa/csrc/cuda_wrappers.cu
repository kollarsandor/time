#include "cuda_wrappers.h"
#include <cuda_runtime.h>
#include <stdio.h>

static int g_initialized = 0;

static cwError_t translate_cuda_error(cudaError_t err) {
    switch (err) {
        case cudaSuccess: return CW_SUCCESS;
        case cudaErrorInvalidValue: return CW_ERROR_INVALID_VALUE;
        case cudaErrorMemoryAllocation: return CW_ERROR_OUT_OF_MEMORY;
        case cudaErrorNotReady: return CW_ERROR_NOT_INITIALIZED;
        default: return CW_ERROR_UNKNOWN;
    }
}

extern "C" {

cwError_t cwInit(void) {
    if (g_initialized) return CW_SUCCESS;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA init failed: %s\n", cudaGetErrorString(err));
        return translate_cuda_error(err);
    }
    if (count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return CW_ERROR_NOT_INITIALIZED;
    }
    g_initialized = 1;
    return CW_SUCCESS;
}

cwError_t cwSetDevice(int device) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device, cudaGetErrorString(err));
    }
    return translate_cuda_error(err);
}

cwError_t cwGetDevice(int* device) {
    cudaError_t err = cudaGetDevice(device);
    return translate_cuda_error(err);
}

cwError_t cwGetDeviceCount(int* count) {
    cudaError_t err = cudaGetDeviceCount(count);
    return translate_cuda_error(err);
}

cwError_t cwDeviceSynchronize(void) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }
    return translate_cuda_error(err);
}

cwError_t cwMalloc(cwDevicePtr_t* ptr, size_t size) {
    void* dptr = NULL;
    cudaError_t err = cudaMalloc(&dptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(%zu) failed: %s\n", size, cudaGetErrorString(err));
        *ptr = 0;
        return translate_cuda_error(err);
    }
    *ptr = (cwDevicePtr_t)dptr;
    return CW_SUCCESS;
}

cwError_t cwFree(cwDevicePtr_t ptr) {
    if (ptr == 0) return CW_SUCCESS;
    cudaError_t err = cudaFree((void*)ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
    }
    return translate_cuda_error(err);
}

cwError_t cwMemcpyH2D(cwDevicePtr_t dst, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy((void*)dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyH2D(%zu) failed: %s\n", size, cudaGetErrorString(err));
    }
    return translate_cuda_error(err);
}

cwError_t cwMemcpyD2H(void* dst, cwDevicePtr_t src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, (void*)src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyD2H(%zu) failed: %s\n", size, cudaGetErrorString(err));
    }
    return translate_cuda_error(err);
}

cwError_t cwMemcpyD2D(cwDevicePtr_t dst, cwDevicePtr_t src, size_t size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyD2D(%zu) failed: %s\n", size, cudaGetErrorString(err));
    }
    return translate_cuda_error(err);
}

cwError_t cwMemset(cwDevicePtr_t ptr, int value, size_t size) {
    cudaError_t err = cudaMemset((void*)ptr, value, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
    }
    return translate_cuda_error(err);
}

cwError_t cwStreamCreate(cwStream_t* stream) {
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        *stream = NULL;
        return translate_cuda_error(err);
    }
    *stream = (cwStream_t)s;
    return CW_SUCCESS;
}

cwError_t cwStreamDestroy(cwStream_t stream) {
    if (stream == NULL) return CW_SUCCESS;
    cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
    return translate_cuda_error(err);
}

cwError_t cwStreamSynchronize(cwStream_t stream) {
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }
    return translate_cuda_error(err);
}

cwError_t cwEventCreate(cwEvent_t* event) {
    cudaEvent_t e;
    cudaError_t err = cudaEventCreate(&e);
    if (err != cudaSuccess) {
        *event = NULL;
        return translate_cuda_error(err);
    }
    *event = (cwEvent_t)e;
    return CW_SUCCESS;
}

cwError_t cwEventDestroy(cwEvent_t event) {
    if (event == NULL) return CW_SUCCESS;
    cudaError_t err = cudaEventDestroy((cudaEvent_t)event);
    return translate_cuda_error(err);
}

cwError_t cwEventRecord(cwEvent_t event, cwStream_t stream) {
    cudaError_t err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    return translate_cuda_error(err);
}

cwError_t cwEventSynchronize(cwEvent_t event) {
    cudaError_t err = cudaEventSynchronize((cudaEvent_t)event);
    return translate_cuda_error(err);
}

cwError_t cwEventElapsedTime(float* ms, cwEvent_t start, cwEvent_t end) {
    cudaError_t err = cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)end);
    return translate_cuda_error(err);
}

size_t cwGetFreeMemory(void) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t cwGetTotalMemory(void) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

const char* cwGetErrorString(cwError_t error) {
    switch (error) {
        case CW_SUCCESS: return "Success";
        case CW_ERROR_INVALID_VALUE: return "Invalid value";
        case CW_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case CW_ERROR_NOT_INITIALIZED: return "Not initialized";
        case CW_ERROR_DEINITIALIZED: return "Deinitialized";
        default: return "Unknown error";
    }
}

}
