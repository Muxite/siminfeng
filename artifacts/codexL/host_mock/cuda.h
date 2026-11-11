#ifndef MOCK_CUDA_H
#define MOCK_CUDA_H
#include <stddef.h>
#include <stdint.h>
#define CUDAAPI
typedef int CUresult;
typedef uintptr_t CUdeviceptr;
typedef void *CUstream;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_SUPPORTED 801
#define CU_STREAM_DEFAULT 0
#define cuMemAlloc cuMemAlloc_v2
#define cuMemFree cuMemFree_v2
#define cuMemGetInfo cuMemGetInfo_v2
#define cuStreamDestroy cuStreamDestroy_v2
#ifdef __cplusplus
extern "C" {
#endif
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t);
CUresult cuMemAllocAsync(CUdeviceptr *, size_t, CUstream);
CUresult cuMemFree_v2(CUdeviceptr);
CUresult cuMemFreeAsync(CUdeviceptr, CUstream);
CUresult cuMemGetInfo_v2(size_t *, size_t *);
CUresult cuStreamSynchronize(CUstream);
CUresult cuStreamCreate(CUstream *, unsigned);
CUresult cuStreamDestroy_v2(CUstream);
#ifdef __cplusplus
}
#endif
#endif
