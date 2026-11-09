#define _POSIX_C_SOURCE 200809L
#include "cuda_runtime_api.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB ((size_t)1048576)
static pthread_barrier_t barrier;
static int successes;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static void *worker(void *unused) {
    (void)unused;
    void *p = NULL;
    int rc = cudaMalloc(&p, MIB);
    assert(rc == cudaSuccess || rc == cudaErrorMemoryAllocation);
    if (!rc) { pthread_mutex_lock(&mutex); ++successes; pthread_mutex_unlock(&mutex); }
    pthread_barrier_wait(&barrier);
    if (!rc) assert(cudaFree(p) == cudaSuccess);
    return NULL;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    size_t f, t;
    assert(cudaMemGetInfo(&f, &t) == cudaSuccess);
    if (!strcmp(argv[1], "threads")) {
        pthread_t threads[16];
        assert(t == 4 * MIB && f == t);
        assert(!pthread_barrier_init(&barrier, NULL, 16));
        for (int i = 0; i < 16; ++i) assert(!pthread_create(&threads[i], NULL, worker, NULL));
        for (int i = 0; i < 16; ++i) assert(!pthread_join(threads[i], NULL));
        assert(successes == 4);
        assert(!cudaMemGetInfo(&f, &t) && f == t);
    } else if (!strcmp(argv[1], "reserve")) {
        assert(t == 4 * MIB && f == 3 * MIB);
        void *p = NULL, *q = NULL;
        assert(!cudaMalloc(&p, 3 * MIB));
        assert(cudaMalloc(&q, 1) == cudaErrorMemoryAllocation);
        assert(!cudaFree(p));
    } else if (!strcmp(argv[1], "managed0")) {
        void *p = NULL;
        assert(!cudaMallocManaged(&p, 100 * MIB, 1));
        assert(!cudaMemGetInfo(&f, &t) && f == t && t == 4 * MIB);
        assert(!cudaFree(p));
    } else if (!strcmp(argv[1], "failures")) {
        void *p = NULL;
        assert(cudaMalloc(&p, 13) == cudaErrorMemoryAllocation);
        assert(!cudaMemGetInfo(&f, &t) && f == t);
        /* Unknown pointer: backend owns it, bookkeeping must tolerate it. */
        assert(!cudaFree(malloc(1)));
        assert(!cudaFree(NULL));
        assert(cudaMemGetInfo(NULL, &t) == cudaErrorInvalidValue);
        size_t pitch;
        assert(cudaMallocPitch(&p, &pitch, SIZE_MAX, 2) == cudaErrorMemoryAllocation);
        struct cudaPitchedPtr volume;
        struct cudaExtent e = {512, SIZE_MAX, 2};
        assert(cudaMalloc3D(&volume, e) == cudaErrorMemoryAllocation);
        assert(!cudaMemGetInfo(&f, &t) && f == t);
    } else if (!strcmp(argv[1], "syncfail")) {
        void *p = NULL;
        assert(!cudaMallocAsync(&p, MIB, NULL));
        assert(cudaFreeAsync(p, (void *)1) != cudaSuccess);
        assert(!cudaMemGetInfo(&f, &t) && f == t - MIB);
    } else if (!strcmp(argv[1], "zero")) {
        void *p = NULL;
        assert(f == 0);
        assert(cudaMalloc(&p, 1) == cudaErrorMemoryAllocation);
    } else if (!strcmp(argv[1], "disabled")) {
        assert(t == (size_t)4 * 1024 * MIB && f == t);
    } else abort();
    printf("PASS host check: %s\n", argv[1]);
    return 0;
}
