// Peak explicit storage: 256 MiB pinned host + 256 MiB GPU; no overlap between sizes.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 1;} } while(0)
int main() {
    cudaStream_t stream; cudaEvent_t begin,end;
    CK(cudaStreamCreate(&stream)); CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
    for (size_t mib=1; mib<=256; mib*=2) {
        size_t bytes=mib*1024*1024; void *host,*device;
        CK(cudaMallocHost(&host,bytes)); CK(cudaMalloc(&device,bytes));
        memset(host,1,bytes);
        CK(cudaMemcpyAsync(device,host,bytes,cudaMemcpyHostToDevice,stream));
        CK(cudaStreamSynchronize(stream));
        for(int r=0;r<3;r++) {
            CK(cudaEventRecord(begin,stream));
            for(int i=0;i<8;i++) CK(cudaMemcpyAsync(device,host,bytes,cudaMemcpyHostToDevice,stream));
            CK(cudaEventRecord(end,stream)); CK(cudaEventSynchronize(end));
            float ms; CK(cudaEventElapsedTime(&ms,begin,end));
            printf("{\"direction\":\"H2D\",\"pinned\":true,\"bytes\":%zu,\"repeat\":%d,\"ms\":%.9f}\n",bytes,r,ms/8);
        }
        CK(cudaFree(device)); CK(cudaFreeHost(host));
    }
    CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end)); CK(cudaStreamDestroy(stream));
}
