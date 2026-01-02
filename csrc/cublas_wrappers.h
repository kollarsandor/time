#ifndef CUBLAS_WRAPPERS_H
#define CUBLAS_WRAPPERS_H

#include <stdint.h>
#include <stddef.h>
#include "cuda_wrappers.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CBW_SUCCESS = 0,
    CBW_ERROR_NOT_INITIALIZED = 1,
    CBW_ERROR_ALLOC_FAILED = 3,
    CBW_ERROR_INVALID_VALUE = 7,
    CBW_ERROR_EXECUTION_FAILED = 13,
    CBW_ERROR_INTERNAL = 14,
    CBW_ERROR_NOT_SUPPORTED = 15
} cbwStatus_t;

typedef enum {
    CBW_OP_N = 0,
    CBW_OP_T = 1,
    CBW_OP_C = 2
} cbwOperation_t;

typedef enum {
    CBW_R_16F = 0,
    CBW_R_32F = 1,
    CBW_R_64F = 2,
    CBW_R_16BF = 3,
    CBW_R_8F_E4M3 = 4,
    CBW_R_8F_E5M2 = 5
} cbwDataType_t;

typedef enum {
    CBW_COMPUTE_16F = 0,
    CBW_COMPUTE_32F = 1,
    CBW_COMPUTE_64F = 2,
    CBW_COMPUTE_32F_FAST_16F = 3,
    CBW_COMPUTE_32F_FAST_TF32 = 4
} cbwComputeType_t;

typedef void* cbwHandle_t;
typedef void* cbwLtHandle_t;
typedef void* cbwLtMatmulDesc_t;
typedef void* cbwLtMatrixLayout_t;
typedef void* cbwLtMatmulPreference_t;

cbwStatus_t cbwCreate(cbwHandle_t* handle);
cbwStatus_t cbwDestroy(cbwHandle_t handle);
cbwStatus_t cbwSetStream(cbwHandle_t handle, cwStream_t stream);
cbwStatus_t cbwSgemm(cbwHandle_t handle, cbwOperation_t transa, cbwOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
cbwStatus_t cbwHgemm(cbwHandle_t handle, cbwOperation_t transa, cbwOperation_t transb, int m, int n, int k, const void* alpha, const void* A, int lda, const void* B, int ldb, const void* beta, void* C, int ldc);
cbwStatus_t cbwSgemmBatched(cbwHandle_t handle, cbwOperation_t transa, cbwOperation_t transb, int m, int n, int k, const float* alpha, const float** Aarray, int lda, const float** Barray, int ldb, const float* beta, float** Carray, int ldc, int batchCount);
cbwStatus_t cbwSgemmStridedBatched(cbwHandle_t handle, cbwOperation_t transa, cbwOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, const float* beta, float* C, int ldc, long long strideC, int batchCount);
cbwStatus_t cbwLtCreate(cbwLtHandle_t* handle);
cbwStatus_t cbwLtDestroy(cbwLtHandle_t handle);
cbwStatus_t cbwLtMatmulDescCreate(cbwLtMatmulDesc_t* desc, cbwComputeType_t computeType, cbwDataType_t scaleType);
cbwStatus_t cbwLtMatmulDescDestroy(cbwLtMatmulDesc_t desc);
cbwStatus_t cbwLtMatrixLayoutCreate(cbwLtMatrixLayout_t* layout, cbwDataType_t type, uint64_t rows, uint64_t cols, int64_t ld);
cbwStatus_t cbwLtMatrixLayoutDestroy(cbwLtMatrixLayout_t layout);
cbwStatus_t cbwLtMatmulPreferenceCreate(cbwLtMatmulPreference_t* pref);
cbwStatus_t cbwLtMatmulPreferenceDestroy(cbwLtMatmulPreference_t pref);
cbwStatus_t cbwLtMatmul(cbwLtHandle_t ltHandle, cbwLtMatmulDesc_t computeDesc, const void* alpha, const void* A, cbwLtMatrixLayout_t Adesc, const void* B, cbwLtMatrixLayout_t Bdesc, const void* beta, const void* C, cbwLtMatrixLayout_t Cdesc, void* D, cbwLtMatrixLayout_t Ddesc, cwStream_t stream);
const char* cbwGetErrorString(cbwStatus_t status);

#ifdef __cplusplus
}
#endif

#endif
