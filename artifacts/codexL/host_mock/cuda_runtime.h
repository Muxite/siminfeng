#include "cuda_runtime_api.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) {
    cudaExtent e = {w, h, d}; return e;
}
