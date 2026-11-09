#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>

static const size_t MiB = 1048576;
static int failures;
static void check(const char *label, bool good) {
    std::printf("%-62s | %s\n", label, good ? "PASS" : "FAIL");
    if (!good) ++failures;
}
static bool info(size_t expected_free) {
    size_t free_bytes = 0, total = 0;
    cudaError_t rc = cudaMemGetInfo(&free_bytes, &total);
    if (rc != cudaSuccess || total != 250 * MiB || free_bytes != expected_free) {
        std::printf("  meminfo: status=%d free=%zu total=%zu expected_free=%zu\n",
                    (int)rc, free_bytes, total, expected_free);
        return false;
    }
    return true;
}
int main() {
    std::printf("GPU budget test: cap=250 MiB, reserve=0, managed=1\n");
    std::printf("%-62s | RESULT\n", "CHECK");
    cudaError_t init = cudaFree(nullptr);
    check("CUDA initialization", init == cudaSuccess);
    if (init != cudaSuccess) {
        std::fprintf(stderr, "CUDA initialization: %s\n", cudaGetErrorString(init));
        return 1;
    }
    check("Initial meminfo: total=250 MiB, free=250 MiB", info(250 * MiB));
    void *a = nullptr, *b = nullptr, *c = nullptr;
    check("cudaMalloc first 100 MiB", cudaMalloc(&a, 100 * MiB) == cudaSuccess);
    check("cudaMalloc second 100 MiB", cudaMalloc(&b, 100 * MiB) == cudaSuccess);
    check("Meminfo after 200 MiB: free=50 MiB", info(50 * MiB));
    cudaError_t denied = cudaMalloc(&c, 100 * MiB);
    check("Third 100 MiB returns cudaErrorMemoryAllocation", denied == cudaErrorMemoryAllocation);
    check("Denied allocation leaves accounting unchanged", info(50 * MiB));
    if (denied == cudaSuccess) cudaFree(c);
    c = nullptr;
    check("Free first allocation", cudaFree(a) == cudaSuccess);
    a = nullptr;
    check("Freed budget permits another 100 MiB", cudaMalloc(&c, 100 * MiB) == cudaSuccess);
    check("Meminfo after reallocation: free=50 MiB", info(50 * MiB));
    check("Free second allocation", cudaFree(b) == cudaSuccess);
    check("Free replacement allocation", cudaFree(c) == cudaSuccess);
    check("Meminfo after frees: free=250 MiB", info(250 * MiB));

    cudaStream_t stream = nullptr;
    cudaError_t stream_rc = cudaStreamCreate(&stream);
    check("Create stream for async allocation test", stream_rc == cudaSuccess);
    if (stream_rc == cudaSuccess) {
        void *p = nullptr, *q = nullptr;
        cudaError_t rc = cudaMallocAsync(&p, 200 * MiB, stream);
        check("cudaMallocAsync 200 MiB", rc == cudaSuccess);
        if (rc != cudaSuccess) std::fprintf(stderr, "cudaMallocAsync: %s\n", cudaGetErrorString(rc));
        check("Async allocation charged before stream completion", info(50 * MiB));
        cudaError_t over = cudaMallocAsync(&q, 100 * MiB, stream);
        check("Async allocation above cap denied", over == cudaErrorMemoryAllocation);
        if (over == cudaSuccess) cudaFreeAsync(q, stream);
        if (rc == cudaSuccess) check("cudaFreeAsync returns budget after completion", cudaFreeAsync(p, stream) == cudaSuccess);
        check("Meminfo after cudaFreeAsync: free=250 MiB", info(250 * MiB));
        p = nullptr;
        check("Async budget can be reused", cudaMallocAsync(&p, 100 * MiB, stream) == cudaSuccess);
        if (p) check("Free reused async allocation", cudaFreeAsync(p, stream) == cudaSuccess);
        check("Destroy stream", cudaStreamDestroy(stream) == cudaSuccess);
    }

    size_t pitch = 0;
    void *pitched = nullptr;
    cudaError_t prc = cudaMallocPitch(&pitched, &pitch, 1000, 1000);
    check("cudaMallocPitch succeeds with valid aligned pitch", prc == cudaSuccess && pitch >= 1000);
    if (prc == cudaSuccess) {
        check("Pitch padding counts against cap", info(250 * MiB - pitch * 1000));
        check("Free pitched allocation", cudaFree(pitched) == cudaSuccess);
    }
    cudaPitchedPtr volume = {};
    cudaExtent extent = make_cudaExtent(1000, 10, 10);
    cudaError_t vrc = cudaMalloc3D(&volume, extent);
    check("cudaMalloc3D succeeds", vrc == cudaSuccess);
    if (vrc == cudaSuccess) {
        check("3D padded storage counts against cap", info(250 * MiB - volume.pitch * 100));
        check("Free 3D allocation", cudaFree(volume.ptr) == cudaSuccess);
    }
    void *managed = nullptr;
    cudaError_t mrc = cudaMallocManaged(&managed, 10 * MiB);
    check("cudaMallocManaged succeeds", mrc == cudaSuccess);
    if (mrc == cudaSuccess) {
        check("Managed allocation charged at requested size", info(240 * MiB));
        check("Free managed allocation", cudaFree(managed) == cudaSuccess);
    }

    CUdeviceptr driver = 0;
    CUresult drc = cuMemAlloc(&driver, 100 * MiB);
    check("Driver cuMemAlloc_v2 succeeds", drc == CUDA_SUCCESS);
    if (drc == CUDA_SUCCESS) {
        size_t f = 0, t = 0;
        check("Runtime sees driver allocation", info(150 * MiB));
        check("cuMemGetInfo_v2 reports shared accounting",
              cuMemGetInfo(&f, &t) == CUDA_SUCCESS && f == 150 * MiB && t == 250 * MiB);
        CUdeviceptr over = 0;
        CUresult rc = cuMemAlloc(&over, 200 * MiB);
        check("Driver allocation above cap returns OUT_OF_MEMORY", rc == CUDA_ERROR_OUT_OF_MEMORY);
        if (rc == CUDA_SUCCESS) cuMemFree(over);
        check("Driver cuMemFree_v2 returns budget", cuMemFree(driver) == CUDA_SUCCESS);
    }
    CUstream ds = nullptr;
    CUresult src = cuStreamCreate(&ds, CU_STREAM_DEFAULT);
    check("Create driver stream", src == CUDA_SUCCESS);
    if (src == CUDA_SUCCESS) {
        CUdeviceptr p = 0, q = 0;
        CUresult rc = cuMemAllocAsync(&p, 200 * MiB, ds);
        check("Driver cuMemAllocAsync succeeds", rc == CUDA_SUCCESS);
        if (rc == CUDA_SUCCESS) {
            check("Runtime sees driver async allocation", info(50 * MiB));
            CUresult denied_rc = cuMemAllocAsync(&q, 100 * MiB, ds);
            check("Driver async allocation above cap denied", denied_rc == CUDA_ERROR_OUT_OF_MEMORY);
            if (denied_rc == CUDA_SUCCESS) cuMemFreeAsync(q, ds);
            check("Driver cuMemFreeAsync returns budget", cuMemFreeAsync(p, ds) == CUDA_SUCCESS);
        }
        check("Destroy driver stream", cuStreamDestroy(ds) == CUDA_SUCCESS);
    }
    check("Final meminfo: free=250 MiB", info(250 * MiB));
    std::printf("%s (%d failed checks)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
