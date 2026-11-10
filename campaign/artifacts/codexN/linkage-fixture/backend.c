extern int cudaMalloc(void), cudaFree(void), cudaMemGetInfo(void);
int probe(void) { return cudaMalloc() + cudaFree() + cudaMemGetInfo(); }
