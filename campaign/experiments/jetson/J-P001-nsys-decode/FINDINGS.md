# J-P001 — nsys decode timeline, bf16, tg64, host-native build of jetson-c1 (same source/flags as image)
Tool: nsys 2024.5.4 (`-t cuda,nvtx,osrt --cuda-graph-trace=node`); run under profiler (~43 t/s vs 50.2 unprofiled → overhead ~15%, ratios only).
- CUDA graphs active: 64 cudaGraphLaunch for 64 tokens; host blocks in cudaStreamSynchronize (50% API time) → GPU-kernel bound, NOT launch-bound. Launch-overhead candidate rejected for Jetson decode.
- GPU time: mul_mat_vec_f bf16 ids-variant 41.1% (1560 inst = 24 layers x 65, avg 397 us), mul_mat_vec_f bf16 non-ids 40.2% (6305 inst, avg 96 us), mul_mat_f bf16 (ids, MMF) 12.2% (1560 inst, avg 118 us), flash-attn 1.4%, rest <1% each.
- Derived effective bandwidth (weights bytes / kernel time; Granite: per expert gate,up 1024x512 each, down 512x1024, bf16; 8 experts/token):
  gate+up via mul_mat_vec_f(ids): 16 MiB / 397 us ≈ 40 GB/s; down via mul_mat_f(ids): 8 MiB / 118 us ≈ 68 GB/s.
- Per token ≈ 820 MB weights touched (experts 576, attention 144, lm_head 100); measured ~23 ms/token under nsys ≈ 36 GB/s vs ~102 GB/s nominal.
- Hypothesis H-J1 (source-supported, untested): the MUL_MAT_ID batch-1 dispatch for gate/up picks the vector kernel, which underuses Orin's 8 SMs; the MMF path or a different vector config would raise bandwidth. Category: hypothesis.
