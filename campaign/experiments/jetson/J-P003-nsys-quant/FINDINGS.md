# J-P003 + J-E006 fusion screen — quantized decode on Orin (image jetson-c1-p01q)
- Q8_0 decode kernels: fused gate/up mul_mat_vec_q<Q8_0,...,fused=1> 265 µs (8.5 MB → ~32 GB/s); down 98 µs (4.25 MB → 43 GB/s); other matvec 50 µs avg.
- mixE4K (experts Q4_K, rest Q8_0): fused gate/up mul_mat_vec_q<Q4_K(12),...,1,1> 255 µs (≈4.7 MB → ~18 GB/s); down 100 µs (≈2.4 MB → 24 GB/s). Q4_K MMVQ time ≈ Q8_0 time for half the bytes → dequant/compute-bound on Orin; explains tg mixE4K 84.7 ≈ Q8_0 83.1.
- Screen GGML_CUDA_DISABLE_FUSION=1 (A,F,F,A; tg128 r5 steady): Q8_0 83.3/83.2 → 73.5/73.5 (−12%); mixE4K 85.0/85.0 → 96.7/97.1 (+14%). bf16 (J-E004) +14%.
- Applicability rule (observed): the value of the MoE gate/up fusion depends on weight type x platform; a global fusion switch is wrong in both directions. Category: runtime-configuration screen; targeted patch pending (J-T3b).
- Tool friction: `nsys stats -f csv -o` produced 0-byte CSVs after a stale .sqlite from a failed run; text output via `--force-export=true` worked. Build CC J-T1b side effect: host build/bin RUNPATH points to deleted staging dir → need LD_LIBRARY_PATH (images unaffected).
