#include <cstdio>
__global__ void add(const float*a,const float*b,float*c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
int main(){int n=1<<22;float*a,*b,*c;cudaMallocManaged(&a,n*4);cudaMallocManaged(&b,n*4);cudaMallocManaged(&c,n*4);for(int i=0;i<n;i++){a[i]=1;b[i]=2;}
 for(int r=0;r<5;r++)add<<<n/256,256>>>(a,b,c,n);cudaDeviceSynchronize();printf("c[0]=%f err=%s\n",c[0],cudaGetErrorString(cudaGetLastError()));}
