#ifndef CUDA_WRAPPERS_H
#define CUDA_WRAPPERS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CW_SUCCESS = 0,
    CW_ERROR_INVALID_VALUE = 1,
    CW_ERROR_OUT_OF_MEMORY = 2,
    CW_ERROR_NOT_INITIALIZED = 3,
    CW_ERROR_DEINITIALIZED = 4,
    CW_ERROR_UNKNOWN = 999
} cwError_t;

typedef void* cwStream_t;
typedef void* cwEvent_t;
typedef uint64_t cwDevicePtr_t;

cwError_t cwInit(void);
cwError_t cwSetDevice(int device);
cwError_t cwGetDevice(int* device);
cwError_t cwGetDeviceCount(int* count);
cwError_t cwDeviceSynchronize(void);
cwError_t cwMalloc(cwDevicePtr_t* ptr, size_t size);
cwError_t cwFree(cwDevicePtr_t ptr);
cwError_t cwMemcpyH2D(cwDevicePtr_t dst, const void* src, size_t size);
cwError_t cwMemcpyD2H(void* dst, cwDevicePtr_t src, size_t size);
cwError_t cwMemcpyD2D(cwDevicePtr_t dst, cwDevicePtr_t src, size_t size);
cwError_t cwMemset(cwDevicePtr_t ptr, int value, size_t size);
cwError_t cwStreamCreate(cwStream_t* stream);
cwError_t cwStreamDestroy(cwStream_t stream);
cwError_t cwStreamSynchronize(cwStream_t stream);
cwError_t cwEventCreate(cwEvent_t* event);
cwError_t cwEventDestroy(cwEvent_t event);
cwError_t cwEventRecord(cwEvent_t event, cwStream_t stream);
cwError_t cwEventSynchronize(cwEvent_t event);
cwError_t cwEventElapsedTime(float* ms, cwEvent_t start, cwEvent_t end);
size_t cwGetFreeMemory(void);
size_t cwGetTotalMemory(void);
const char* cwGetErrorString(cwError_t error);

#ifdef __cplusplus
}
#endif

#endif
