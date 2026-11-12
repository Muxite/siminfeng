# J-OM14: first Nsight Systems trace through the lab loop

Factory engine `factory/llamacpp:bd173c08-restored-0b514fba` (run 4 final), its kept env (fill + pregate K=8),
contract argv (Qwen3.6-35B-A3B, p1, -n 64, -ngl 99 -lm mmap -cmoe). Run via `cc/om_run/om_run.sh` with `OM_NSYS=1`
(nsys 2024.5.4 mounted read-only into the container; `-t cuda,nvtx,osrt --cuda-graph-trace=node`).

- Output sha256 after moving nsys's own messages to nsys.stdout.txt: **7a1541792364** = frozen reference.
- Traced decode 1907.5 ms/token vs ~711 untraced (J-OM13): tracing (mostly osrt) costs ~2.7x. Breakdown only, never speed evidence.
- Whole run (load + prompt + decode), not decode-only:
  - GPU kernels 10.1 s total; top mul_mat_vec_q (Q4_K-family type 8) 52%.
  - **Host-to-device memcpy 17.6 s / 10,645 copies**: more than all kernels. Likely the 195-token prompt offloading CPU-resident
    expert weights to the GPU; needs a decode-window view to confirm.
  - CUDA API: cudaMemcpyAsync 71%, cudaStreamSynchronize 21%.
  - OS runtime (summed thread time): pthread_cond_wait 59%, futex 20%, pread 19% (68,719 calls).
- Report nsys.nsys-rep (14.3 MB) not committed; sha256 f9e9db8d6436eeee…
