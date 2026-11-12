# F18 — storage ceiling + decode-phase fault/latency evidence

Two parts. Work only in `/home/muk/projects/siminfeng/cc/io_bench/` (new) and `/home/muk/projects/xs-dev/llama-obs` (private llama.cpp clone, branch `xs-obs` at `6e9d96cf4` = cf34d3113 + expert-streaming patches 01–08). No Docker/GPU, no commits in siminfeng (operator commits), no network. Other workers are compiling concurrently: build with `-j2`.

Context (from J-OM13 run stats, factory engine on Qwen3.6-35B-A3B, 63 decode tokens): per token the engine requests ~618 MB of expert ranges; 84% already resident; demand fill reads ~98 MB/token in ~289 ms (≈340 MB/s effective, ~10 pread syscalls per layer of ~230 KB avg, 16 worker threads, 2 MiB chunks) while async pregate reads ~118 MB/token; total token ≈ 711 ms. Published Orin Nano NVMe random-read figures: 1.27–2.5 GB/s. We need (1) our drive's real ceiling at our request pattern, and (2) evidence for where the non-IO ~420 ms/token goes (minor faults? waits?).

## Part A — `cc/io_bench/io_bench.py` (+ fio job generator)
- Input: model file (default `/home/muk/models/campaign/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`), optional `--ranges ranges.json` (expert ranges; produce one from the GGUF header with a minimal stdlib GGUF reader: every `*_exps` tensor's [offset,size], and per-expert slices of size nb[2]), request sizes (default 256K, 640K, 1M, 1.83M, 2M, 4M), concurrency (1,2,4,8,16,32), access pattern: `random-expert` (random expert slices, as decode does), `sequential`.
- Cold protocol: before each cell, `posix_fadvise(DONTNEED)` the byte ranges it will read (no root needed); record residency (mincore) of a sample before/after to prove coldness.
- Engine A: python threads + `os.pread` (buffered, like the engine). Engine B: `fio` (installed at /usr/bin/fio) with `ioengine=psync` and `io_uring` (if supported) and `direct=1` variants, driven via generated job files with `--output-format=json`. Keep total bytes per cell bounded (e.g. 512 MiB) and the whole default sweep < 10 minutes; `--quick` mode < 90 s.
- Output: `results.json` + a markdown table: MB/s, p50/p95 latency per request, per cell. Also print "ceiling at engine pattern" = best MB/s for 2 MiB/16 threads buffered.
- Tests `cc/io_bench/test_io_bench.py` on a temp file (quick: tiny sizes), plus GGUF range extraction test on a synthetic GGUF.
- Do NOT run the full sweep yourself on the model file (the operator runs it as an exclusive queue job); run `--quick` on a 1 GiB temp file only.

## Part B — engine patch P16 `phase_rusage` (observability) in `llama-obs`
Env `LLAMA_EXPERT_PHASE_STATS=0|1` (default 0; register in the strict `LLAMA_EXPERT_*` validator and the `expert_config: effective` line). When 1: snapshot `getrusage(RUSAGE_SELF)` and a monotonic clock at the first decode ubatch (n_tokens==1) and at context teardown, and record per-decode-token wall time (time between successive decode `llama_decode` completions; store in a preallocated vector, no per-token syscalls beyond clock_gettime). At teardown print ONE line:
`expert_phase: stats decode_tokens=%llu decode_wall_ns=%llu minflt=%ld majflt=%ld nvcsw=%ld nivcsw=%ld utime_us=%lld stime_us=%lld tok_ns_p50=%llu tok_ns_p95=%llu tok_ns_max=%llu first8_ns_mean=%llu rest_ns_mean=%llu`
(deltas over the decode phase only). Hot-path cost when disabled: one branch. It must work with or without fill/pregate enabled (so it can also measure the stock-equivalent configuration of this engine).
Checks into `/home/muk/projects/xs-dev/obs-artifacts/F18-check.md`: patch file `16-phase-rusage.patch` (git format-patch -1), `git apply --check` onto 6e9d96cf4 in a throwaway worktree, CPU-only build (`cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release`, `cmake --build build-cpu -j2 --target llama-completion`), run on `/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf` with `-p "The capital of France is" -n 32 -c 512 --temp 0 -s 1 -no-cnv -cmoe -lm mmap -t 4`, switch off/on, identical stdout sha256, stats line printed; also with fill+pregate on (`LLAMA_EXPERT_FILL=1 LLAMA_EXPERT_FILL_RANDOM=decode LLAMA_EXPERT_FILL_MAX_TOKENS=1 LLAMA_EXPERT_PREGATE=1 LLAMA_EXPERT_PREGATE_K=8`).

## Acceptance (run verbatim)
- I1 `python3 -m pytest /home/muk/projects/siminfeng/cc/io_bench -q -p no:cacheprovider` passes.
- I2 `python3 /home/muk/projects/siminfeng/cc/io_bench/io_bench.py --help` works; `--quick` run output pasted in `cc/io_bench/README.md` (temp file run).
- I3 `/home/muk/projects/xs-dev/obs-artifacts/16-phase-rusage.patch` exists; F18-check.md shows identical sha off/on and the stats line.
Final message: what each tool measures, how to run the full sweep, caveats.
