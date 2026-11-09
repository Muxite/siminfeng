#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(const int *a, const int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024;
    int ha[N], hb[N], hc[N];
    for (int i = 0; i < N; ++i) { ha[i] = i; hb[i] = 2 * i; hc[i] = 0; }

    cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, 0);
    if (e != cudaSuccess) {
        printf("cudaGetDeviceProperties: %s\nFAIL\n", cudaGetErrorString(e));
        return 1;
    }
    printf("name: %s\n", p.name);
    printf("compute capability: %d.%d\n", p.major, p.minor);
    printf("totalGlobalMem: %zu bytes\n", p.totalGlobalMem);
    printf("integrated: %d\n", p.integrated);
    printf("canMapHostMemory: %d\n", p.canMapHostMemory);
    printf("unifiedAddressing: %d\n", p.unifiedAddressing);
    printf("concurrentManagedAccess: %d\n", p.concurrentManagedAccess);
    printf("pageableMemoryAccess: %d\n", p.pageableMemoryAccess);

    int *da, *db, *dc;
    cudaMalloc(&da, N * sizeof(int));
    cudaMalloc(&db, N * sizeof(int));
    cudaMalloc(&dc, N * sizeof(int));
    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(int), cudaMemcpyHostToDevice);
    add<<<(N + 255) / 256, 256>>>(da, db, dc, N);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("kernel error: %s\nFAIL\n", cudaGetErrorString(e));
        return 1;
    }
    cudaMemcpy(hc, dc, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dc);

    int bad = 0;
    for (int i = 0; i < N; ++i) if (hc[i] != 3 * i) ++bad;
    printf("mismatches: %d / %d\n", bad, N);
    printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
