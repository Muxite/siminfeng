// Read-only, bounded UMA probe. Run on the operator's exclusive GPU queue.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using clock_type = std::chrono::steady_clock;
static double ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}
static void check(cudaError_t e) {
    if (e != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(e)); }
}
__global__ void read_all(const volatile uint32_t * p, size_t words, unsigned long long * out) {
    unsigned long long v = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += size_t(gridDim.x) * blockDim.x) {
        v += p[i];
    }
    __shared__ unsigned long long sums[256];
    sums[threadIdx.x] = v;
    __syncthreads();
    for (unsigned s = 128; s; s >>= 1) {
        if (threadIdx.x < s) { sums[threadIdx.x] += sums[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[blockIdx.x] = sums[0]; }
}
static int attribute(const char * name, cudaDeviceAttr attr) {
    int value = -1;
    cudaError_t e = cudaDeviceGetAttribute(&value, attr, 0);
    if (e != cudaSuccess) { (void) cudaGetLastError(); }
    std::printf("# %s=%d\n", name, value);
    return value;
}
static void skip(const char * kind, const char * reason, double setup_ms = 0) {
    std::printf("%s\tSKIP:%s\t%.6f\t0\t0\t0\t0\n", kind, reason, setup_ms);
}
static void bench(const char * kind, const void * p, size_t bytes, double setup_ms,
                  unsigned long long expected, unsigned long long * out) {
    const unsigned blocks = 256;
    cudaEvent_t a, b;
    check(cudaEventCreate(&a));
    check(cudaEventCreate(&b));
    std::vector<float> times;
    for (int r = 0; r < 21; ++r) {
        check(cudaEventRecord(a));
        read_all<<<blocks, 256>>>(static_cast<const volatile uint32_t *>(p), bytes / 4, out);
        check(cudaGetLastError());
        check(cudaEventRecord(b));
        check(cudaEventSynchronize(b));
        float elapsed;
        check(cudaEventElapsedTime(&elapsed, a, b));
        times.push_back(elapsed);
    }
    std::vector<unsigned long long> sums(blocks);
    check(cudaMemcpy(sums.data(), out, blocks * sizeof(*out), cudaMemcpyDeviceToHost));
    unsigned long long sum = 0;
    for (auto v : sums) { sum += v; }
    if (sum != expected) { throw std::runtime_error("checksum mismatch"); }
    const double first = times.front();
    std::sort(times.begin() + 1, times.end());
    const double warm = (times[10] + times[11]) / 2;
    std::printf("%s\tOK\t%.6f\t%.6f\t%.6f\t%.6f\t%llu\n",
                kind, setup_ms, first, warm, bytes / (warm * 1e-3) / (1ULL << 30), sum);
    check(cudaEventDestroy(a));
    check(cudaEventDestroy(b));
}
int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 4) {
            std::fprintf(stderr, "usage: probe_uma FILE [BYTES=67108864] [PAGE_ALIGNED_OFFSET=0]\n");
            return 2;
        }
        const size_t bytes = argc > 2 ? std::stoull(argv[2]) : 64ULL << 20;
        const off_t offset = argc > 3 ? std::stoull(argv[3]) : 0;
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        if (!bytes || bytes > (256ULL << 20) || bytes % page || offset < 0 || size_t(offset) % page) {
            throw std::runtime_error("require page-aligned bytes/offset and 0 < bytes <= 256 MiB");
        }
        check(cudaSetDeviceFlags(cudaDeviceMapHost));
        check(cudaSetDevice(0));
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, 0));
        int runtime, driver;
        check(cudaRuntimeGetVersion(&runtime));
        check(cudaDriverGetVersion(&driver));
        std::printf("# device=%s cc=%d.%d runtime=%d driver=%d bytes=%zu offset=%lld\n",
                    prop.name, prop.major, prop.minor, runtime, driver, bytes, (long long) offset);
        const int pageable = attribute("pageableMemoryAccess", cudaDevAttrPageableMemoryAccess);
        attribute("pageableMemoryAccessUsesHostPageTables", cudaDevAttrPageableMemoryAccessUsesHostPageTables);
        attribute("concurrentManagedAccess", cudaDevAttrConcurrentManagedAccess);
        attribute("canUseHostPointerForRegisteredMem", cudaDevAttrCanUseHostPointerForRegisteredMem);
        attribute("integrated", cudaDevAttrIntegrated);
        const int mapped = attribute("canMapHostMemory", cudaDevAttrCanMapHostMemory);
        const int registered = attribute("hostRegisterSupported", cudaDevAttrHostRegisterSupported);
        const int readonly = attribute("hostRegisterReadOnlySupported", cudaDevAttrHostRegisterReadOnlySupported);
        std::puts("kind\tstatus\tsetup_ms\tfirst_read_ms\twarm_median_ms\twarm_GiB_s\tchecksum");
        const int fd = open(argv[1], O_RDONLY);
        if (fd < 0) { throw std::runtime_error("open FILE failed"); }
        struct stat st;
        if (fstat(fd, &st) || uint64_t(offset) > uint64_t(st.st_size) || bytes > uint64_t(st.st_size) - uint64_t(offset)) {
            throw std::runtime_error("file range outside FILE");
        }
        auto start = clock_type::now();
        void * file = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, offset);
        close(fd);
        if (file == MAP_FAILED) { throw std::runtime_error("mmap failed"); }
        unsigned long long expected = 0;
        const auto * words = static_cast<const uint32_t *>(file);
        for (size_t i = 0; i < bytes / 4; ++i) { expected += words[i]; }
        std::printf("# mmap_and_cpu_prefault_ms=%.6f checksum=%llu\n", ms(start), expected);
        unsigned long long * out = nullptr;
        check(cudaMalloc(&out, 256 * sizeof(*out)));

        void * device = nullptr;
        start = clock_type::now();
        check(cudaMalloc(&device, bytes));
        check(cudaMemcpy(device, file, bytes, cudaMemcpyHostToDevice));
        check(cudaDeviceSynchronize());
        bench("cudaMalloc", device, bytes, ms(start), expected, out);
        check(cudaFree(device));

        if (registered == 1 && readonly == 1 && mapped == 1) {
            start = clock_type::now();
            const cudaError_t e = cudaHostRegister(file, bytes, cudaHostRegisterMapped | cudaHostRegisterReadOnly);
            if (e == cudaSuccess) {
                void * alias = nullptr;
                const cudaError_t alias_e = cudaHostGetDevicePointer(&alias, file, 0);
                if (alias_e == cudaSuccess) {
                    bench("registered_ro_mmap", alias, bytes, ms(start), expected, out);
                } else {
                    skip("registered_ro_mmap", cudaGetErrorName(alias_e), ms(start));
                    (void) cudaGetLastError();
                }
                check(cudaHostUnregister(file));
            } else {
                skip("registered_ro_mmap", cudaGetErrorName(e), ms(start));
                (void) cudaGetLastError();
            }
        } else { skip("registered_ro_mmap", "missing_registration_readonly_or_mapping_attribute"); }

        if (mapped == 1) {
            void * pinned = nullptr;
            start = clock_type::now();
            check(cudaMallocHost(&pinned, bytes));
            std::memcpy(pinned, file, bytes);
            void * alias = nullptr;
            check(cudaHostGetDevicePointer(&alias, pinned, 0));
            bench("cudaMallocHost", alias, bytes, ms(start), expected, out);
            check(cudaFreeHost(pinned));
        } else { skip("cudaMallocHost", "canMapHostMemory_is_not_1"); }

        if (pageable == 1) {
            start = clock_type::now();
            void * plain = std::malloc(bytes);
            if (!plain) { throw std::bad_alloc(); }
            std::memcpy(plain, file, bytes);
            bench("pageable_malloc_direct", plain, bytes, ms(start), expected, out);
            std::free(plain);
            bench("pageable_ro_mmap_direct", file, bytes, 0, expected, out);
        } else { skip("pageable_malloc_direct", "pageableMemoryAccess_is_not_1"); }
        check(cudaFree(out));
        munmap(file, bytes);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "probe_uma: %s\n", e.what());
        return 1;
    }
}
