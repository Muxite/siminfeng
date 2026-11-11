/* CPU-only fixture. Deliberately call driver symbols from runtime functions so
 * real ELF interposition exercises suppression of nested double accounting. */
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <limits.h>
#include <stdlib.h>

CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) {
    if (!p) return 1;
    if (n == 13) return 2; /* predictable real-backend failure */
    *p = n ? (CUdeviceptr)malloc(1) : 0;
    return n && !*p ? 2 : 0;
}
CUresult cuMemAllocAsync(CUdeviceptr *p, size_t n, CUstream s) {
    (void)s; return cuMemAlloc_v2(p, n);
}
CUresult cuMemFree_v2(CUdeviceptr p) { free((void *)p); return 0; }
CUresult cuMemFreeAsync(CUdeviceptr p, CUstream s) {
    (void)s; return cuMemFree_v2(p);
}
CUresult cuMemGetInfo_v2(size_t *f, size_t *t) {
    if (!f || !t) return 1;
    *f = *t = (size_t)4 * 1024 * 1024 * 1024; return 0;
}
CUresult cuStreamSynchronize(CUstream s) { return s == (void *)1 ? 1 : 0; }
CUresult cuStreamCreate(CUstream *s, unsigned f) {
    (void)f; *s = malloc(1); return *s ? 0 : 2;
}
CUresult cuStreamDestroy_v2(CUstream s) { free(s); return 0; }
cudaError_t cudaMalloc(void **p, size_t n) { return cuMemAlloc_v2((CUdeviceptr *)p, n); }
cudaError_t cudaMallocManaged(void **p, size_t n, unsigned f) {
    (void)f; return cudaMalloc(p, n);
}
cudaError_t cudaMallocAsync(void **p, size_t n, cudaStream_t s) {
    return cuMemAllocAsync((CUdeviceptr *)p, n, s);
}
cudaError_t cudaFree(void *p) { return cuMemFree_v2((CUdeviceptr)p); }
cudaError_t cudaFreeAsync(void *p, cudaStream_t s) {
    return cuMemFreeAsync((CUdeviceptr)p, s);
}
cudaError_t cudaMallocPitch(void **p, size_t *pitch, size_t w, size_t h) {
    *pitch = w; return cudaMalloc(p, w * h);
}
cudaError_t cudaMalloc3D(struct cudaPitchedPtr *p, struct cudaExtent e) {
    p->pitch = e.width; p->xsize = e.width; p->ysize = e.height;
    return cudaMalloc(&p->ptr, e.width * e.height * e.depth);
}
cudaError_t cudaMemGetInfo(size_t *f, size_t *t) { return cuMemGetInfo_v2(f, t); }
cudaError_t cudaGetDevice(int *d) { *d = 0; return 0; }
cudaError_t cudaDeviceGetAttribute(int *v, enum cudaDeviceAttr a, int d) {
    (void)d; *v = a == cudaDevAttrMaxPitch ? INT_MAX : 512; return 0;
}
cudaError_t cudaStreamSynchronize(cudaStream_t s) { return cuStreamSynchronize(s); }
cudaError_t cudaStreamCreate(cudaStream_t *s) { return cuStreamCreate(s, 0); }
cudaError_t cudaStreamDestroy(cudaStream_t s) { return cuStreamDestroy_v2(s); }
const char *cudaGetErrorString(cudaError_t e) { return e ? "mock failure" : "mock success"; }
