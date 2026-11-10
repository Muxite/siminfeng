// Rotating, read-only DRAM dot-product screen. No host configuration changes.
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <omp.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

static size_t l3_size(void) {
    glob_t paths;
    char keys[1024][256]; int count=0; size_t total=0;
    if (glob("/sys/devices/system/cpu/cpu[0-9]*/cache/index*/level",0,NULL,&paths)) return 0;
    for (size_t i=0;i<paths.gl_pathc;i++) {
        char path[512], key[256], size[64]; int level=0;
        FILE *f=fopen(paths.gl_pathv[i],"r");
        if (!f) continue;
        int ok=fscanf(f,"%d",&level); fclose(f);
        if (ok!=1 || level!=3) continue;
        snprintf(path,sizeof(path),"%s",paths.gl_pathv[i]);
        strcpy(strrchr(path,'/')+1,"shared_cpu_list");
        f=fopen(path,"r"); if (!f) {total=0; break;}
        char *got=fgets(key,sizeof(key),f); fclose(f);
        if (!got) {total=0; break;}
        int seen=0; for(int j=0;j<count;j++) if(!strcmp(keys[j],key)) seen=1;
        if(seen) continue;
        if(count==1024) {total=0; break;}
        strcpy(keys[count++],key);
        strcpy(strrchr(path,'/')+1,"size");
        f=fopen(path,"r"); if(!f) {total=0; break;}
        got=fgets(size,sizeof(size),f); fclose(f);
        if(!got) {total=0; break;}
        char *unit; unsigned long long n=strtoull(size,&unit,10);
        size_t factor=*unit=='K'?1024:*unit=='M'?1048576:0;
        if(!factor || n>1024*1024) {total=0; break;}
        total+=n*factor;
    }
    globfree(&paths); return total;
}
int main(int argc,char **argv) {
    alarm(24); // External plan also has a 25 s deadline.
    int max_threads=argc>1?atoi(argv[1]):8;
    if(argc>2 || max_threads<1 || max_threads>64) {fprintf(stderr,"usage: bw_cpu [max_threads 1..64]\n"); return 2;}
    size_t l3=l3_size();
    if(!l3 || l3>128ULL*1048576) {fprintf(stderr,"unknown L3 or 8x L3 exceeds 1 GiB; refusing\n");return 2;}
    size_t bytes=512ULL*1048576;
    if(bytes<8*l3) bytes=8*l3;
    bytes=(bytes+32*1048576-1)/(32*1048576)*(32*1048576);
    if(bytes>1024ULL*1048576) return 2;
    float *bank[8]; size_t n=bytes/8/sizeof(float);
    for(int b=0;b<8;b++) {
        if(posix_memalign((void **)&bank[b],64,n*sizeof(float))) return 1;
        for(size_t i=0;i<n;i++) bank[b][i]=1.0f;
    }
    omp_set_dynamic(0);
    for(int threads=1;threads<=max_threads;threads++) {
        omp_set_num_threads(threads);
        for(int repeat=0;repeat<3;repeat++) {
            double begin=omp_get_wtime(),elapsed=0,checksum=0; int passes=0,actual=0;
            do {
                double pass_sum=0;
                #pragma omp parallel reduction(+:pass_sum)
                {
                    #pragma omp single
                    actual=omp_get_num_threads();
                    __m256 a=_mm256_setzero_ps(),b=_mm256_setzero_ps();
                    #pragma omp for schedule(static)
                    for(size_t j=0;j<bytes/sizeof(float);j+=16) {
                        size_t block=(j/n+passes)%8, i=j%n;
                        a=_mm256_add_ps(a,_mm256_load_ps(bank[block]+i));
                        b=_mm256_add_ps(b,_mm256_load_ps(bank[block]+i+8));
                    }
                    float lanes[8]; _mm256_storeu_ps(lanes,_mm256_add_ps(a,b));
                    for(int k=0;k<8;k++) pass_sum+=lanes[k];
                }
                checksum+=pass_sum; passes++; elapsed=omp_get_wtime()-begin;
            } while(elapsed<0.08);
            if(actual!=threads || !isfinite(checksum) || checksum<=0) return 1;
            printf("{\"probe\":\"cpu-stream-v1\",\"threads\":%d,\"repeat\":%d,\"working_set_bytes\":%zu,\"cache_bytes\":%zu,\"buffers\":8,\"bytes_read\":%zu,\"ms\":%.9f,\"checksum\":%.9g,\"gbs\":%.9f}\n",threads,repeat,bytes,l3,bytes*(size_t)passes,elapsed*1000,checksum,bytes*passes/elapsed/1e9);
            fflush(stdout);
        }
    }
    for(int b=0;b<8;b++) free(bank[b]);
    return 0;
}
