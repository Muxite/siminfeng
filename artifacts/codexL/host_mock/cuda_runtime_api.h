#ifndef MOCK_RUNTIME_H
#define MOCK_RUNTIME_H
#include <stddef.h>
#define CUDARTAPI
typedef int cudaError_t;
typedef void *cudaStream_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorMemoryAllocation 2
#define cudaErrorNotSupported 801
enum cudaDeviceAttr { cudaDevAttrMaxPitch = 11, cudaDevAttrTextureAlignment = 14,
                      cudaDevAttrTexturePitchAlignment = 51 };
struct cudaExtent { size_t width, height, depth; };
struct cudaPitchedPtr { void *ptr; size_t pitch, xsize, ysize; };
#ifdef __cplusplus
extern "C" {
#endif
cudaError_t cudaMalloc(void **, size_t);
cudaError_t cudaMallocManaged(void **, size_t, unsigned
#ifdef __cplusplus
 = 1
#endif
);
cudaError_t cudaMallocAsync(void **, size_t, cudaStream_t);
cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t);
cudaError_t cudaMalloc3D(struct cudaPitchedPtr *, struct cudaExtent);
cudaError_t cudaFree(void *);
cudaError_t cudaFreeAsync(void *, cudaStream_t);
cudaError_t cudaMemGetInfo(size_t *, size_t *);
cudaError_t cudaGetDevice(int *);
cudaError_t cudaDeviceGetAttribute(int *, enum cudaDeviceAttr, int);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamCreate(cudaStream_t *);
cudaError_t cudaStreamDestroy(cudaStream_t);
const char *cudaGetErrorString(cudaError_t);
#ifdef __cplusplus
}
#endif
#endif
