// Coalesced GEMV-like dot products over eight rotating weight buffers.
#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 1;} } while(0)
__global__ void dot(const float *weights,const float *x,float *out,size_t n) {
    float sum=0;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
        sum+=weights[i]*x[i%1024];
    __shared__ float partial[256]; partial[threadIdx.x]=sum; __syncthreads();
    for(int step=128;step;step/=2) {
        if(threadIdx.x<step) partial[threadIdx.x]+=partial[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) out[blockIdx.x]=partial[0];
}
int main() {
    alarm(24);
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    size_t cache=prop.l2CacheSize, bytes=512ULL*1048576;
    if(!cache || cache>128ULL*1048576) {fprintf(stderr,"unknown L2 or 8x L2 exceeds 1 GiB\n"); return 2;}
    if(bytes<8*cache) bytes=8*cache;
    bytes=(bytes+32*1048576-1)/(32*1048576)*(32*1048576);
    if(bytes>1024ULL*1048576) return 2;
    float *weights,*x,*out;
    CK(cudaMalloc(&weights,bytes)); CK(cudaMalloc(&x,4096)); CK(cudaMalloc(&out,4096));
    CK(cudaMemset(weights,0x3f,bytes)); CK(cudaMemset(x,0x3f,4096));
    size_t n=bytes/8/sizeof(float);
    // Full-bank warmup; measured passes always visit ALL buffers before reuse.
    for(int b=0;b<8;b++) dot<<<1024,256>>>(weights+b*n,x,out,n);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t begin,end; CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
    for(int r=0;r<3;r++) {
        CK(cudaEventRecord(begin));
        for(int pass=0;pass<8;pass++) for(int b=0;b<8;b++)
            dot<<<1024,256>>>(weights+((b+pass)%8)*n,x,out,n);
        CK(cudaGetLastError()); CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
        float ms; CK(cudaEventElapsedTime(&ms,begin,end));
        float checksum[1024]; CK(cudaMemcpy(checksum,out,4096,cudaMemcpyDeviceToHost));
        double sum=0; for(int i=0;i<1024;i++) sum+=checksum[i];
        if(!(sum>0) || !(ms>0)) return 1;
        printf("{\"probe\":\"cuda-stream-v1\",\"threads\":1,\"repeat\":%d,\"working_set_bytes\":%zu,\"cache_bytes\":%zu,\"buffers\":8,\"bytes_read\":%zu,\"ms\":%.9f,\"checksum\":%.9g,\"gbs\":%.9f}\n",r,bytes,cache,bytes*8,ms,sum,bytes*8/(ms*1e6));
    }
    CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end));
    CK(cudaFree(out)); CK(cudaFree(x)); CK(cudaFree(weights));
}
