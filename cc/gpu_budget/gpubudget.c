#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build as C99. No CUDA library dependency: every real call uses RTLD_NEXT.
 * A process-wide cap deliberately aggregates devices and contexts. */
#define BUCKETS 4096
#define MIB ((size_t)1048576)
struct allocation { uintptr_t key; size_t bytes; struct allocation *next; };
static struct allocation *table[BUCKETS];
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static __thread unsigned nesting;
static size_t budget, reserve, capacity, used, peak;
static int enabled, count_managed = 1, finalized;
static FILE *logfile;

static void event(const char *kind, const char *api, uintptr_t key, size_t bytes, int status) {
    if (!logfile) return;
    fprintf(logfile, "{\"event\":\"%s\",\"api\":\"%s\",\"pointer\":\"0x%" PRIxPTR
            "\",\"bytes\":%zu,\"used\":%zu,\"peak\":%zu,\"budget\":%zu,"
            "\"reserve\":%zu,\"capacity\":%zu,\"status\":%d}\n",
            kind, api, key, bytes, used, peak, budget, reserve, capacity, status);
    /* Unbuffered log: every event, including deny, survives a subsequent crash. */
    fflush(logfile);
}

static void finish(void) {
    pthread_mutex_lock(&lock);
    if (!finalized) {
        finalized = 1;
        event("exit", "process", 0, 0, 0);
        if (logfile) { fclose(logfile); logfile = NULL; }
    }
    pthread_mutex_unlock(&lock);
}
__attribute__((destructor)) static void on_unload(void) { finish(); }

static size_t env_mib(const char *name, int *present) {
    const char *s = getenv(name);
    if (!s) { *present = 0; return 0; }
    *present = 1;
    errno = 0;
    char *end;
    unsigned long long n = strtoull(s, &end, 10);
    if (!*s || *s < '0' || *s > '9' || *end || errno || n > SIZE_MAX / MIB) {
        fprintf(stderr, "gpubudget: invalid %s (expected integer MiB); refusing to run\n", name);
        _Exit(125);
    }
    return (size_t)n * MIB;
}
static void initialize(void) {
    int present;
    budget = env_mib("GPU_BUDGET_MIB", &enabled);
    reserve = env_mib("GPU_BUDGET_RESERVE_MIB", &present);
    capacity = budget > reserve ? budget - reserve : 0;
    const char *managed = getenv("GPU_BUDGET_COUNT_MANAGED");
    if (managed) {
        if (strcmp(managed, "0") && strcmp(managed, "1")) {
            fprintf(stderr, "gpubudget: GPU_BUDGET_COUNT_MANAGED must be 0 or 1\n");
            _Exit(125);
        }
        count_managed = strcmp(managed, "0") != 0;
    }
    const char *path = getenv("GPU_BUDGET_LOG");
    if (path && *path) {
        logfile = fopen(path, "a");
        if (!logfile) { perror("gpubudget: GPU_BUDGET_LOG"); _Exit(125); }
        setvbuf(logfile, NULL, _IONBF, 0);
    }
    atexit(finish);
    event("init", "process", 0, 0, 0);
}
static void enter(void) { ++nesting; pthread_once(&once, initialize); pthread_mutex_lock(&lock); }
static void leave(void) { pthread_mutex_unlock(&lock); --nesting; }
static size_t hash(uintptr_t key) { return ((key >> 8) ^ (key >> 20)) % BUCKETS; }
static struct allocation **find(uintptr_t key) {
    struct allocation **p = &table[hash(key)];
    while (*p && (*p)->key != key) p = &(*p)->next;
    return p;
}
static int prepare(size_t bytes, const char *api, struct allocation **node) {
    if (bytes > SIZE_MAX - used || (enabled && bytes > capacity - used)) {
        event("deny", api, 0, bytes, 2); return 0;
    }
    *node = malloc(sizeof(**node));
    if (!*node) { event("deny", api, 0, bytes, 2); return 0; }
    return 1;
}
static void record(struct allocation *node, uintptr_t key, size_t bytes, const char *api, int status) {
    if (status || !key) { free(node); event("error", api, key, bytes, status); return; }
    node->key = key; node->bytes = bytes;
    size_t b = hash(key); node->next = table[b]; table[b] = node;
    used += bytes; if (used > peak) peak = used;
    event("alloc", api, key, bytes, status);
}
static void released(uintptr_t key, const char *api, int status) {
    struct allocation **p = find(key);
    size_t bytes = *p ? (*p)->bytes : 0;
    if (!status && *p) {
        struct allocation *node = *p; *p = node->next; used -= bytes; free(node);
    }
    event(status ? "error" : "free", api, key, bytes, status);
}
#define REAL(name) __typeof__(&name) real = (__typeof__(&name))dlsym(RTLD_NEXT, #name)
#define RBEGIN(name, args) REAL(name); if (!real) { return cudaErrorNotSupported; } \
    if (nesting) { return real args; } enter()
#define DBEGIN(name, args) REAL(name); if (!real) { return CUDA_ERROR_NOT_SUPPORTED; } \
    if (nesting) { return real args; } enter()

cudaError_t CUDARTAPI cudaMalloc(void **ptr, size_t bytes) {
    RBEGIN(cudaMalloc, (ptr, bytes));
    struct allocation *node;
    if (!prepare(bytes, "cudaMalloc", &node)) { leave(); return cudaErrorMemoryAllocation; }
    cudaError_t rc = real(ptr, bytes);
    record(node, rc == cudaSuccess && ptr ? (uintptr_t)*ptr : 0, bytes, "cudaMalloc", rc);
    leave(); return rc;
}
cudaError_t CUDARTAPI cudaMallocManaged(void **ptr, size_t bytes, unsigned flags) {
    RBEGIN(cudaMallocManaged, (ptr, bytes, flags));
    if (!count_managed) { cudaError_t rc = real(ptr, bytes, flags); leave(); return rc; }
    struct allocation *node;
    if (!prepare(bytes, "cudaMallocManaged", &node)) { leave(); return cudaErrorMemoryAllocation; }
    cudaError_t rc = real(ptr, bytes, flags);
    record(node, rc == cudaSuccess && ptr ? (uintptr_t)*ptr : 0, bytes, "cudaMallocManaged", rc);
    leave(); return rc;
}
cudaError_t CUDARTAPI cudaMallocAsync(void **ptr, size_t bytes, cudaStream_t stream) {
    RBEGIN(cudaMallocAsync, (ptr, bytes, stream));
    struct allocation *node;
    if (!prepare(bytes, "cudaMallocAsync", &node)) { leave(); return cudaErrorMemoryAllocation; }
    cudaError_t rc = real(ptr, bytes, stream);
    record(node, rc == cudaSuccess && ptr ? (uintptr_t)*ptr : 0, bytes, "cudaMallocAsync", rc);
    leave(); return rc;
}
cudaError_t CUDARTAPI cudaFree(void *ptr) {
    RBEGIN(cudaFree, (ptr));
    cudaError_t rc = real(ptr); released((uintptr_t)ptr, "cudaFree", rc); leave(); return rc;
}
cudaError_t CUDARTAPI cudaFreeAsync(void *ptr, cudaStream_t stream) {
    RBEGIN(cudaFreeAsync, (ptr, stream));
    __typeof__(&cudaStreamSynchronize) sync = (__typeof__(&cudaStreamSynchronize))dlsym(RTLD_NEXT, "cudaStreamSynchronize");
    if (!sync) { leave(); return cudaErrorNotSupported; }
    cudaError_t rc = real(ptr, stream);
    if (rc == cudaSuccess) rc = sync(stream);
    released((uintptr_t)ptr, "cudaFreeAsync", rc); leave(); return rc;
}

/* Synthesized pitched storage avoids speculative real allocations above the cap.
 * Align to both device texture constraints and at least 512 bytes. */
static cudaError_t pitched(void **ptr, size_t *pitch, size_t width, size_t height,
                           size_t depth, const char *api) {
    __typeof__(&cudaMalloc) alloc = (__typeof__(&cudaMalloc))dlsym(RTLD_NEXT, "cudaMalloc");
    __typeof__(&cudaGetDevice) device = (__typeof__(&cudaGetDevice))dlsym(RTLD_NEXT, "cudaGetDevice");
    __typeof__(&cudaDeviceGetAttribute) attr = (__typeof__(&cudaDeviceGetAttribute))dlsym(RTLD_NEXT, "cudaDeviceGetAttribute");
    if (!alloc || !device || !attr) return cudaErrorNotSupported;
    int dev, align_pitch, align_texture, max_pitch;
    cudaError_t rc = device(&dev);
    if (!rc) rc = attr(&align_pitch, cudaDevAttrTexturePitchAlignment, dev);
    if (!rc) rc = attr(&align_texture, cudaDevAttrTextureAlignment, dev);
    if (!rc) rc = attr(&max_pitch, cudaDevAttrMaxPitch, dev);
    if (rc) return rc;
    size_t alignment = 512;
    if (align_pitch > 0 && (size_t)align_pitch > alignment) alignment = (size_t)align_pitch;
    if (align_texture > 0 && (size_t)align_texture > alignment) alignment = (size_t)align_texture;
    if (!ptr || !pitch || !width || !height || !depth) return cudaErrorInvalidValue;
    if (width > SIZE_MAX - (alignment - 1)) goto overflow;
    size_t row = ((width + alignment - 1) / alignment) * alignment;
    if (max_pitch <= 0 || row > (size_t)max_pitch) return cudaErrorInvalidValue;
    if (height > SIZE_MAX / row || depth > SIZE_MAX / (row * height)) goto overflow;
    size_t bytes = row * height * depth;
    struct allocation *node;
    if (!prepare(bytes, api, &node)) return cudaErrorMemoryAllocation;
    rc = alloc(ptr, bytes);
    if (!rc) *pitch = row;
    record(node, rc == cudaSuccess ? (uintptr_t)*ptr : 0, bytes, api, rc);
    return rc;
overflow:
    event("deny", api, 0, SIZE_MAX, cudaErrorMemoryAllocation);
    return cudaErrorMemoryAllocation;
}
cudaError_t CUDARTAPI cudaMallocPitch(void **ptr, size_t *pitch, size_t width, size_t height) {
    RBEGIN(cudaMallocPitch, (ptr, pitch, width, height));
    cudaError_t rc = enabled ? pitched(ptr, pitch, width, height, 1, "cudaMallocPitch") : real(ptr, pitch, width, height);
    leave(); return rc;
}
cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr *ptr, struct cudaExtent extent) {
    RBEGIN(cudaMalloc3D, (ptr, extent));
    cudaError_t rc;
    if (!enabled) rc = real(ptr, extent);
    else if (!ptr) rc = cudaErrorInvalidValue;
    else {
        void *base = NULL; size_t pitch = 0;
        rc = pitched(&base, &pitch, extent.width, extent.height, extent.depth, "cudaMalloc3D");
        if (!rc) { ptr->ptr = base; ptr->pitch = pitch; ptr->xsize = extent.width; ptr->ysize = extent.height; }
    }
    leave(); return rc;
}
cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    RBEGIN(cudaMemGetInfo, (free_bytes, total_bytes));
    cudaError_t rc = real(free_bytes, total_bytes);
    if (!rc && enabled && free_bytes && total_bytes) { *total_bytes = budget; *free_bytes = capacity - used; }
    leave(); return rc;
}
CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *ptr, size_t bytes) {
    DBEGIN(cuMemAlloc_v2, (ptr, bytes));
    struct allocation *node;
    if (!prepare(bytes, "cuMemAlloc_v2", &node)) { leave(); return CUDA_ERROR_OUT_OF_MEMORY; }
    CUresult rc = real(ptr, bytes);
    record(node, rc == CUDA_SUCCESS && ptr ? (uintptr_t)*ptr : 0, bytes, "cuMemAlloc_v2", rc);
    leave(); return rc;
}
CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *ptr, size_t bytes, CUstream stream) {
    DBEGIN(cuMemAllocAsync, (ptr, bytes, stream));
    struct allocation *node;
    if (!prepare(bytes, "cuMemAllocAsync", &node)) { leave(); return CUDA_ERROR_OUT_OF_MEMORY; }
    CUresult rc = real(ptr, bytes, stream);
    record(node, rc == CUDA_SUCCESS && ptr ? (uintptr_t)*ptr : 0, bytes, "cuMemAllocAsync", rc);
    leave(); return rc;
}
CUresult CUDAAPI cuMemFree_v2(CUdeviceptr ptr) {
    DBEGIN(cuMemFree_v2, (ptr));
    CUresult rc = real(ptr); released((uintptr_t)ptr, "cuMemFree_v2", rc); leave(); return rc;
}
CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr ptr, CUstream stream) {
    DBEGIN(cuMemFreeAsync, (ptr, stream));
    __typeof__(&cuStreamSynchronize) sync = (__typeof__(&cuStreamSynchronize))dlsym(RTLD_NEXT, "cuStreamSynchronize");
    if (!sync) { leave(); return CUDA_ERROR_NOT_SUPPORTED; }
    CUresult rc = real(ptr, stream);
    if (!rc) rc = sync(stream);
    released((uintptr_t)ptr, "cuMemFreeAsync", rc); leave(); return rc;
}
CUresult CUDAAPI cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes) {
    DBEGIN(cuMemGetInfo_v2, (free_bytes, total_bytes));
    CUresult rc = real(free_bytes, total_bytes);
    if (!rc && enabled && free_bytes && total_bytes) { *total_bytes = budget; *free_bytes = capacity - used; }
    leave(); return rc;
}
