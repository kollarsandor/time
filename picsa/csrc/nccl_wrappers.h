#ifndef NCCL_WRAPPERS_H
#define NCCL_WRAPPERS_H

#include <stdint.h>
#include <stddef.h>
#include "cuda_wrappers.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NCW_SUCCESS = 0,
    NCW_ERROR_UNHANDLED = 1,
    NCW_ERROR_SYSTEM = 2,
    NCW_ERROR_INTERNAL = 3,
    NCW_ERROR_INVALID_ARGUMENT = 4,
    NCW_ERROR_INVALID_USAGE = 5
} ncwResult_t;

typedef enum {
    NCW_INT8 = 0,
    NCW_UINT8 = 1,
    NCW_INT32 = 2,
    NCW_UINT32 = 3,
    NCW_INT64 = 4,
    NCW_UINT64 = 5,
    NCW_FLOAT16 = 6,
    NCW_FLOAT32 = 7,
    NCW_FLOAT64 = 8,
    NCW_BFLOAT16 = 9,
    NCW_FP8_E4M3 = 10,
    NCW_FP8_E5M2 = 11
} ncwDataType_t;

typedef enum {
    NCW_SUM = 0,
    NCW_PROD = 1,
    NCW_MAX = 2,
    NCW_MIN = 3,
    NCW_AVG = 4
} ncwReduceOp_t;

typedef void* ncwComm_t;
typedef struct { char internal[128]; } ncwUniqueId;

ncwResult_t ncwGetUniqueId(ncwUniqueId* uniqueId);
ncwResult_t ncwCommInitRank(ncwComm_t* comm, int nranks, ncwUniqueId uniqueId, int rank);
ncwResult_t ncwCommDestroy(ncwComm_t comm);
ncwResult_t ncwCommCount(ncwComm_t comm, int* count);
ncwResult_t ncwCommUserRank(ncwComm_t comm, int* rank);
ncwResult_t ncwAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncwDataType_t datatype, ncwReduceOp_t op, ncwComm_t comm, cwStream_t stream);
ncwResult_t ncwBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncwDataType_t datatype, int root, ncwComm_t comm, cwStream_t stream);
ncwResult_t ncwReduce(const void* sendbuff, void* recvbuff, size_t count, ncwDataType_t datatype, ncwReduceOp_t op, int root, ncwComm_t comm, cwStream_t stream);
ncwResult_t ncwAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncwDataType_t datatype, ncwComm_t comm, cwStream_t stream);
ncwResult_t ncwReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncwDataType_t datatype, ncwReduceOp_t op, ncwComm_t comm, cwStream_t stream);
ncwResult_t ncwSend(const void* sendbuff, size_t count, ncwDataType_t datatype, int peer, ncwComm_t comm, cwStream_t stream);
ncwResult_t ncwRecv(void* recvbuff, size_t count, ncwDataType_t datatype, int peer, ncwComm_t comm, cwStream_t stream);
ncwResult_t ncwGroupStart(void);
ncwResult_t ncwGroupEnd(void);
const char* ncwGetErrorString(ncwResult_t result);

#ifdef __cplusplus
}
#endif

#endif
