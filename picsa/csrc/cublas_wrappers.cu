#include "cublas_wrappers.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <stdio.h>

static cbwStatus_t translate_cublas_error(cublasStatus_t err) {
    switch (err) {
        case CUBLAS_STATUS_SUCCESS: return CBW_SUCCESS;
        case CUBLAS_STATUS_NOT_INITIALIZED: return CBW_ERROR_NOT_INITIALIZED;
        case CUBLAS_STATUS_ALLOC_FAILED: return CBW_ERROR_ALLOC_FAILED;
        case CUBLAS_STATUS_INVALID_VALUE: return CBW_ERROR_INVALID_VALUE;
        case CUBLAS_STATUS_EXECUTION_FAILED: return CBW_ERROR_EXECUTION_FAILED;
        case CUBLAS_STATUS_INTERNAL_ERROR: return CBW_ERROR_INTERNAL;
        case CUBLAS_STATUS_NOT_SUPPORTED: return CBW_ERROR_NOT_SUPPORTED;
        default: return CBW_ERROR_INTERNAL;
    }
}

static cublasOperation_t translate_op(cbwOperation_t op) {
    switch (op) {
        case CBW_OP_N: return CUBLAS_OP_N;
        case CBW_OP_T: return CUBLAS_OP_T;
        case CBW_OP_C: return CUBLAS_OP_C;
        default: return CUBLAS_OP_N;
    }
}

static cudaDataType_t translate_datatype_cuda(cbwDataType_t dt) {
    switch (dt) {
        case CBW_R_16F: return CUDA_R_16F;
        case CBW_R_32F: return CUDA_R_32F;
        case CBW_R_64F: return CUDA_R_64F;
        case CBW_R_16BF: return CUDA_R_16BF;
#if CUDA_VERSION >= 11080
        case CBW_R_8F_E4M3: return CUDA_R_8F_E4M3;
        case CBW_R_8F_E5M2: return CUDA_R_8F_E5M2;
#else
        case CBW_R_8F_E4M3: return CUDA_R_16F;
        case CBW_R_8F_E5M2: return CUDA_R_16F;
#endif
        default: return CUDA_R_32F;
    }
}

static cublasComputeType_t translate_compute_type(cbwComputeType_t ct) {
    switch (ct) {
        case CBW_COMPUTE_16F: return CUBLAS_COMPUTE_16F;
        case CBW_COMPUTE_32F: return CUBLAS_COMPUTE_32F;
        case CBW_COMPUTE_64F: return CUBLAS_COMPUTE_64F;
        case CBW_COMPUTE_32F_FAST_16F: return CUBLAS_COMPUTE_32F_FAST_16F;
        case CBW_COMPUTE_32F_FAST_TF32: return CUBLAS_COMPUTE_32F_FAST_TF32;
        default: return CUBLAS_COMPUTE_32F;
    }
}

extern "C" {

cbwStatus_t cbwCreate(cbwHandle_t* handle) {
    cublasHandle_t h;
    cublasStatus_t err = cublasCreate(&h);
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed: %d\n", (int)err);
        *handle = NULL;
        return translate_cublas_error(err);
    }
    *handle = (cbwHandle_t)h;
    return CBW_SUCCESS;
}

cbwStatus_t cbwDestroy(cbwHandle_t handle) {
    if (handle == NULL) return CBW_SUCCESS;
    cublasStatus_t err = cublasDestroy((cublasHandle_t)handle);
    return translate_cublas_error(err);
}

cbwStatus_t cbwSetStream(cbwHandle_t handle, cwStream_t stream) {
    cublasStatus_t err = cublasSetStream((cublasHandle_t)handle, (cudaStream_t)stream);
    return translate_cublas_error(err);
}

cbwStatus_t cbwSgemm(cbwHandle_t handle, cbwOperation_t transa, cbwOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    cublasStatus_t err = cublasSgemm((cublasHandle_t)handle, translate_op(transa), translate_op(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed: %d\n", (int)err);
    }
    return translate_cublas_error(err);
}

cbwStatus_t cbwHgemm(cbwHandle_t handle, cbwOperation_t transa, cbwOperation_t transb, int m, int n, int k, const void* alpha, const void* A, int lda, const void* B, int ldb, const void* beta, void* C, int ldc) {
    cublasStatus_t err = cublasHgemm((cublasHandle_t)handle, translate_op(transa), translate_op(transb), m, n, k, (const __half*)alpha, (const __half*)A, lda, (const __half*)B, ldb, (const __half*)beta, (__half*)C, ldc);
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasHgemm failed: %d\n", (int)err);
    }
    return translate_cublas_error(err);
}

cbwStatus_t cbwSgemmBatched(cbwHandle_t handle, cbwOperation_t transa, cbwOperation_t transb, int m, int n, int k, const float* alpha, const float** Aarray, int lda, const float** Barray, int ldb, const float* beta, float** Carray, int ldc, int batchCount) {
    cublasStatus_t err = cublasSgemmBatched((cublasHandle_t)handle, translate_op(transa), translate_op(transb), m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemmBatched failed: %d\n", (int)err);
    }
    return translate_cublas_error(err);
}

cbwStatus_t cbwSgemmStridedBatched(cbwHandle_t handle, cbwOperation_t transa, cbwOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, const float* beta, float* C, int ldc, long long strideC, int batchCount) {
    cublasStatus_t err = cublasSgemmStridedBatched((cublasHandle_t)handle, translate_op(transa), translate_op(transb), m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemmStridedBatched failed: %d\n", (int)err);
    }
    return translate_cublas_error(err);
}

cbwStatus_t cbwLtCreate(cbwLtHandle_t* handle) {
    cublasLtHandle_t h;
    cublasStatus_t err = cublasLtCreate(&h);
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasLtCreate failed: %d\n", (int)err);
        *handle = NULL;
        return translate_cublas_error(err);
    }
    *handle = (cbwLtHandle_t)h;
    return CBW_SUCCESS;
}

cbwStatus_t cbwLtDestroy(cbwLtHandle_t handle) {
    if (handle == NULL) return CBW_SUCCESS;
    cublasStatus_t err = cublasLtDestroy((cublasLtHandle_t)handle);
    return translate_cublas_error(err);
}

cbwStatus_t cbwLtMatmulDescCreate(cbwLtMatmulDesc_t* desc, cbwComputeType_t computeType, cbwDataType_t scaleType) {
    cublasLtMatmulDesc_t d;
    cublasStatus_t err = cublasLtMatmulDescCreate(&d, translate_compute_type(computeType), translate_datatype_cuda(scaleType));
    if (err != CUBLAS_STATUS_SUCCESS) {
        *desc = NULL;
        return translate_cublas_error(err);
    }
    *desc = (cbwLtMatmulDesc_t)d;
    return CBW_SUCCESS;
}

cbwStatus_t cbwLtMatmulDescDestroy(cbwLtMatmulDesc_t desc) {
    if (desc == NULL) return CBW_SUCCESS;
    cublasStatus_t err = cublasLtMatmulDescDestroy((cublasLtMatmulDesc_t)desc);
    return translate_cublas_error(err);
}

cbwStatus_t cbwLtMatrixLayoutCreate(cbwLtMatrixLayout_t* layout, cbwDataType_t type, uint64_t rows, uint64_t cols, int64_t ld) {
    cublasLtMatrixLayout_t l;
    cublasStatus_t err = cublasLtMatrixLayoutCreate(&l, translate_datatype_cuda(type), rows, cols, ld);
    if (err != CUBLAS_STATUS_SUCCESS) {
        *layout = NULL;
        return translate_cublas_error(err);
    }
    *layout = (cbwLtMatrixLayout_t)l;
    return CBW_SUCCESS;
}

cbwStatus_t cbwLtMatrixLayoutDestroy(cbwLtMatrixLayout_t layout) {
    if (layout == NULL) return CBW_SUCCESS;
    cublasStatus_t err = cublasLtMatrixLayoutDestroy((cublasLtMatrixLayout_t)layout);
    return translate_cublas_error(err);
}

cbwStatus_t cbwLtMatmulPreferenceCreate(cbwLtMatmulPreference_t* pref) {
    cublasLtMatmulPreference_t p;
    cublasStatus_t err = cublasLtMatmulPreferenceCreate(&p);
    if (err != CUBLAS_STATUS_SUCCESS) {
        *pref = NULL;
        return translate_cublas_error(err);
    }
    *pref = (cbwLtMatmulPreference_t)p;
    return CBW_SUCCESS;
}

cbwStatus_t cbwLtMatmulPreferenceDestroy(cbwLtMatmulPreference_t pref) {
    if (pref == NULL) return CBW_SUCCESS;
    cublasStatus_t err = cublasLtMatmulPreferenceDestroy((cublasLtMatmulPreference_t)pref);
    return translate_cublas_error(err);
}

cbwStatus_t cbwLtMatmul(cbwLtHandle_t ltHandle, cbwLtMatmulDesc_t computeDesc, const void* alpha, const void* A, cbwLtMatrixLayout_t Adesc, const void* B, cbwLtMatrixLayout_t Bdesc, const void* beta, const void* C, cbwLtMatrixLayout_t Cdesc, void* D, cbwLtMatrixLayout_t Ddesc, cwStream_t stream) {
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtMatmulPreference_t preference = nullptr;
    cublasStatus_t err = cublasLtMatmulPreferenceCreate(&preference);
    if (err != CUBLAS_STATUS_SUCCESS) {
        return translate_cublas_error(err);
    }
    int returnedResults = 0;
    err = cublasLtMatmulAlgoGetHeuristic((cublasLtHandle_t)ltHandle, (cublasLtMatmulDesc_t)computeDesc, (cublasLtMatrixLayout_t)Adesc, (cublasLtMatrixLayout_t)Bdesc, (cublasLtMatrixLayout_t)Cdesc, (cublasLtMatrixLayout_t)Ddesc, preference, 1, &heuristicResult, &returnedResults);
    if (err != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        fprintf(stderr, "cublasLtMatmulAlgoGetHeuristic failed\n");
        return translate_cublas_error(err);
    }
    err = cublasLtMatmul((cublasLtHandle_t)ltHandle, (cublasLtMatmulDesc_t)computeDesc, alpha, A, (cublasLtMatrixLayout_t)Adesc, B, (cublasLtMatrixLayout_t)Bdesc, beta, C, (cublasLtMatrixLayout_t)Cdesc, D, (cublasLtMatrixLayout_t)Ddesc, &heuristicResult.algo, nullptr, 0, (cudaStream_t)stream);
    cublasLtMatmulPreferenceDestroy(preference);
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasLtMatmul failed: %d\n", (int)err);
    }
    return translate_cublas_error(err);
}

const char* cbwGetErrorString(cbwStatus_t status) {
    switch (status) {
        case CBW_SUCCESS: return "Success";
        case CBW_ERROR_NOT_INITIALIZED: return "Not initialized";
        case CBW_ERROR_ALLOC_FAILED: return "Allocation failed";
        case CBW_ERROR_INVALID_VALUE: return "Invalid value";
        case CBW_ERROR_EXECUTION_FAILED: return "Execution failed";
        case CBW_ERROR_INTERNAL: return "Internal error";
        case CBW_ERROR_NOT_SUPPORTED: return "Not supported";
        default: return "Unknown error";
    }
}

}
