#include "nccl_wrappers.h"
#include <nccl.h>
#include <stdio.h>
#include <string.h>

static ncwResult_t translate_nccl_error(ncclResult_t err) {
    switch (err) {
        case ncclSuccess: return NCW_SUCCESS;
        case ncclUnhandledCudaError: return NCW_ERROR_UNHANDLED;
        case ncclSystemError: return NCW_ERROR_SYSTEM;
        case ncclInternalError: return NCW_ERROR_INTERNAL;
        case ncclInvalidArgument: return NCW_ERROR_INVALID_ARGUMENT;
        case ncclInvalidUsage: return NCW_ERROR_INVALID_USAGE;
        default: return NCW_ERROR_INTERNAL;
    }
}

static ncclDataType_t translate_datatype(ncwDataType_t dt) {
    switch (dt) {
        case NCW_INT8: return ncclInt8;
        case NCW_UINT8: return ncclUint8;
        case NCW_INT32: return ncclInt32;
        case NCW_UINT32: return ncclUint32;
        case NCW_INT64: return ncclInt64;
        case NCW_UINT64: return ncclUint64;
        case NCW_FLOAT16: return ncclFloat16;
        case NCW_FLOAT32: return ncclFloat32;
        case NCW_FLOAT64: return ncclFloat64;
        case NCW_BFLOAT16: return ncclBfloat16;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 16, 0)
        case NCW_FP8_E4M3: return ncclFp8E4M3;
        case NCW_FP8_E5M2: return ncclFp8E5M2;
#else
        case NCW_FP8_E4M3: return ncclFloat16;
        case NCW_FP8_E5M2: return ncclFloat16;
#endif
        default: return ncclFloat32;
    }
}

static ncclRedOp_t translate_op(ncwReduceOp_t op) {
    switch (op) {
        case NCW_SUM: return ncclSum;
        case NCW_PROD: return ncclProd;
        case NCW_MAX: return ncclMax;
        case NCW_MIN: return ncclMin;
        case NCW_AVG: return ncclAvg;
        default: return ncclSum;
    }
}

extern "C" {

ncwResult_t ncwGetUniqueId(ncwUniqueId* uniqueId) {
    ncclUniqueId id;
    ncclResult_t err = ncclGetUniqueId(&id);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclGetUniqueId failed: %s\n", ncclGetErrorString(err));
        return translate_nccl_error(err);
    }
    memcpy(uniqueId->internal, id.internal, sizeof(id.internal));
    return NCW_SUCCESS;
}

ncwResult_t ncwCommInitRank(ncwComm_t* comm, int nranks, ncwUniqueId uniqueId, int rank) {
    ncclComm_t ncomm;
    ncclUniqueId id;
    memcpy(id.internal, uniqueId.internal, sizeof(id.internal));
    ncclResult_t err = ncclCommInitRank(&ncomm, nranks, id, rank);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclCommInitRank failed: %s\n", ncclGetErrorString(err));
        *comm = NULL;
        return translate_nccl_error(err);
    }
    *comm = (ncwComm_t)ncomm;
    return NCW_SUCCESS;
}

ncwResult_t ncwCommDestroy(ncwComm_t comm) {
    if (comm == NULL) return NCW_SUCCESS;
    ncclResult_t err = ncclCommDestroy((ncclComm_t)comm);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclCommDestroy failed: %s\n", ncclGetErrorString(err));
    }
    return translate_nccl_error(err);
}

ncwResult_t ncwCommCount(ncwComm_t comm, int* count) {
    ncclResult_t err = ncclCommCount((ncclComm_t)comm, count);
    return translate_nccl_error(err);
}

ncwResult_t ncwCommUserRank(ncwComm_t comm, int* rank) {
    ncclResult_t err = ncclCommUserRank((ncclComm_t)comm, rank);
    return translate_nccl_error(err);
}

ncwResult_t ncwAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncwDataType_t datatype, ncwReduceOp_t op, ncwComm_t comm, cwStream_t stream) {
    ncclResult_t err = ncclAllReduce(sendbuff, recvbuff, count, translate_datatype(datatype), translate_op(op), (ncclComm_t)comm, (cudaStream_t)stream);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclAllReduce failed: %s\n", ncclGetErrorString(err));
    }
    return translate_nccl_error(err);
}

ncwResult_t ncwBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncwDataType_t datatype, int root, ncwComm_t comm, cwStream_t stream) {
    ncclResult_t err = ncclBroadcast(sendbuff, recvbuff, count, translate_datatype(datatype), root, (ncclComm_t)comm, (cudaStream_t)stream);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclBroadcast failed: %s\n", ncclGetErrorString(err));
    }
    return translate_nccl_error(err);
}

ncwResult_t ncwReduce(const void* sendbuff, void* recvbuff, size_t count, ncwDataType_t datatype, ncwReduceOp_t op, int root, ncwComm_t comm, cwStream_t stream) {
    ncclResult_t err = ncclReduce(sendbuff, recvbuff, count, translate_datatype(datatype), translate_op(op), root, (ncclComm_t)comm, (cudaStream_t)stream);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclReduce failed: %s\n", ncclGetErrorString(err));
    }
    return translate_nccl_error(err);
}

ncwResult_t ncwAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncwDataType_t datatype, ncwComm_t comm, cwStream_t stream) {
    ncclResult_t err = ncclAllGather(sendbuff, recvbuff, sendcount, translate_datatype(datatype), (ncclComm_t)comm, (cudaStream_t)stream);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclAllGather failed: %s\n", ncclGetErrorString(err));
    }
    return translate_nccl_error(err);
}

ncwResult_t ncwReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncwDataType_t datatype, ncwReduceOp_t op, ncwComm_t comm, cwStream_t stream) {
    ncclResult_t err = ncclReduceScatter(sendbuff, recvbuff, recvcount, translate_datatype(datatype), translate_op(op), (ncclComm_t)comm, (cudaStream_t)stream);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclReduceScatter failed: %s\n", ncclGetErrorString(err));
    }
    return translate_nccl_error(err);
}

ncwResult_t ncwSend(const void* sendbuff, size_t count, ncwDataType_t datatype, int peer, ncwComm_t comm, cwStream_t stream) {
    ncclResult_t err = ncclSend(sendbuff, count, translate_datatype(datatype), peer, (ncclComm_t)comm, (cudaStream_t)stream);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclSend failed: %s\n", ncclGetErrorString(err));
    }
    return translate_nccl_error(err);
}

ncwResult_t ncwRecv(void* recvbuff, size_t count, ncwDataType_t datatype, int peer, ncwComm_t comm, cwStream_t stream) {
    ncclResult_t err = ncclRecv(recvbuff, count, translate_datatype(datatype), peer, (ncclComm_t)comm, (cudaStream_t)stream);
    if (err != ncclSuccess) {
        fprintf(stderr, "ncclRecv failed: %s\n", ncclGetErrorString(err));
    }
    return translate_nccl_error(err);
}

ncwResult_t ncwGroupStart(void) {
    ncclResult_t err = ncclGroupStart();
    return translate_nccl_error(err);
}

ncwResult_t ncwGroupEnd(void) {
    ncclResult_t err = ncclGroupEnd();
    return translate_nccl_error(err);
}

const char* ncwGetErrorString(ncwResult_t result) {
    switch (result) {
        case NCW_SUCCESS: return "Success";
        case NCW_ERROR_UNHANDLED: return "Unhandled CUDA error";
        case NCW_ERROR_SYSTEM: return "System error";
        case NCW_ERROR_INTERNAL: return "Internal error";
        case NCW_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case NCW_ERROR_INVALID_USAGE: return "Invalid usage";
        default: return "Unknown error";
    }
}

}
