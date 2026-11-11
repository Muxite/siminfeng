// cuda_alloc_probe: allocate N MiB in chunks via cudaMalloc (or cudaMallocManaged/cudaHostAlloc), touch it, report.
// usage: cuda_alloc_probe <total_mib> <chunk_mib> [device|managed|host]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
__global__ void touch(char* p, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; for(;i<n;i+=(size_t)gridDim.x*blockDim.x) p[i]=1; }
int main(int c,char**v){
  size_t total=atol(v[1]),chunk=atol(v[2]); const char* kind=c>3?v[3]:"device"; size_t done=0;
  while(done<total){ char* p=nullptr; size_t b=chunk<<20; cudaError_t e;
    if(!strcmp(kind,"managed")) e=cudaMallocManaged(&p,b); else if(!strcmp(kind,"host")) e=cudaHostAlloc((void**)&p,b,cudaHostAllocMapped); else e=cudaMalloc(&p,b);
    if(e!=cudaSuccess){ printf("ALLOC_FAIL at %zu MiB: %s\n",done,cudaGetErrorString(e)); return 2; }
    char* d=p; if(!strcmp(kind,"host")) cudaHostGetDevicePointer((void**)&d,p,0);
    touch<<<64,256>>>(d,b); e=cudaDeviceSynchronize(); if(e!=cudaSuccess){ printf("TOUCH_FAIL at %zu MiB: %s\n",done,cudaGetErrorString(e)); return 3; }
    done+=chunk; printf("ok %zu MiB\n",done); fflush(stdout); }
  size_t f,t; cudaMemGetInfo(&f,&t); printf("DONE %zu MiB kind=%s memGetInfo free=%zu total=%zu MiB\n",done,kind,f>>20,t>>20); return 0; }
