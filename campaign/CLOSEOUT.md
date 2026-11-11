# Jetson lane closeout (FINAL) - campaign c1, lane jetson-arm, branch jetson/c1

Drafted 2026-11-10 ~15:35 UTC on host `satay` (host id `jetson-arm`, Orin Nano 8GB Super) from files in this repo. Rules used: numbers are copied from the cited files; "not recorded" means I did not find it; nothing uncompiled or unrun is called validated. Raw t/s are local to this host and must not be pooled with x86 (different weights file hash, shared vs discrete memory).

Final pass by the Jetson Opus lead 2026-11-10 ~17:30 UTC: J-OM7, KS, J-SPEC1, J-SPEC2 results filled in; tables below that still describe "uncommitted" state reflect the draft time - section 1.5 is authoritative.

Jetson decisions are numbered D021-D033 and collide with x86's D021-D048 (different content). Proposed rename J-D021..J-D033 is unconfirmed (handoff/jetson/REPLY-1.md); this document cites them as "Jetson D0xx".

---------------------------------------------------------------------

## 1. Identity

### 1.1 Repo and branch
| item | value |
|---|---|
| branch | `jetson/c1` (tracks `origin/jetson/c1`, which is at the same commit as local HEAD) |
| HEAD | `01fa56c71cfaab12febbec22dd3f52e69121ae80` ("jetson-arm: D031 upstream +7% & NVMe 570 MiB/s ceiling model, D032 ..., D033 ...; kernel screen tool (J-T6), mmap advice patch, pipelines") |
| origin/main (local ref, not re-fetched) | `f0e676b28c927d01ce19de95e3987160937a5fc5` ("Campaign c1 snapshot: Granite MoE ...") |
| commits ahead of origin/main | 14 (c9e7a29 ... 01fa56c; `git log --oneline origin/main..HEAD`) |
| uncommitted at drafting time | 36 `git status` entries (modified evidence/telemetry, new J-SPEC1 files, om_spec, pipeline5/6, mmap_advice patch 02 + README, research/x86-c2-digest.md, handoff/jetson/, several build-result dirs). History was rewritten once before publication (D024). Nothing here is committed. |
| x86 branch | none from x86 seen on Jetson besides `origin/x86/c2-resume` (digest in research/x86-c2-digest.md); x86 has not answered round 2 (handoff/x86-requests-2.md) |

### 1.5 Final state at closeout (authoritative)
- Branch `jetson/c1` pushed to `origin/jetson/c1`; closeout commit = the commit titled "jetson-arm: CLOSEOUT ..." (parent `0402e2d`, checkpoint with J-OM7/J-SPEC1). `origin/main` untouched (f0e676b); `origin/x86/c2-resume` untouched (read only). No force-push.
- Working tree clean after the closeout commit except gitignored build outputs (staging trees, `engine-image.docker.tar` archives, ccache) - deliberately not committed (D024).
- Active processes: none campaign-owned (no pipeline/om_run/om_spec/build_engine/codex/llama processes). Delegated agents: all finished (Codex J-OM7, J-SPEC1; Sonnet REPLY-1/digest and CLOSEOUT draft). No Codex sessions running.
- GPU: no containers (running or stopped) on the host; tegrastats GR3D_FREQ 0%. GPU released by the campaign; no service occupies it (Jetson has no stopped/restored GPU service - none was stopped, so nothing restored). docker/containerd and stock NVIDIA L4T services (nvargus, nvfancontrol, nvphs, ...) are running as before the campaign; none were touched.
- Host settings unchanged by the campaign except user-approved installs: docker.io + nvidia runtime + user `docker` group (D022), buildx user plugin, `pip --user pytest`, `~/venvs/convert`. No clocks/power/driver changes.

### 1.6 Large artifacts NOT in git (local only on host `satay`; no approved remote artifact store exists for the Jetson lane)
| artifact | location | size | identity | retrieval |
|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-Q4_K_XL | `~/models/campaign/qwen36-35b-a3b/` | 22,360,456,160 B | sha256 in 1.4 table (707a55a8...) | HF `unsloth/Qwen3.6-35B-A3B-GGUF` revision a483e9e6, file `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`; verify sha256 |
| Granite safetensors | `~/models/campaign/granite-3.1-1b-a400m-instruct/` | 2.5 GB | HF revision 0da7a48b | HF `ibm-granite/granite-3.1-1b-a400m-instruct@0da7a48b` |
| Granite GGUFs (bf16, Q8_0, mixE4K) | `~/models/campaign/gguf/` | 4.6 GB | sha256 in 1.4 table | regenerate: convert_hf_to_gguf.py (pin 31288c9c6, `~/venvs/convert`) + llama-quantize recipes in experiments/jetson/J-E006-quant-track |
| KL base logits | `~/models/campaign/kl/` | 48 MB | not hashed | regenerate with llama-perplexity `--kl-divergence-base` per J-E005 cmd files |
| wikitext-2-raw | `~/models/campaign/data/` | 18 MB | - | standard dataset |
| 12 engine images + archives | docker store; `cc/build_engine/results/*/engine-image.docker.tar` | 3.7 GB each | image Id + archive sha256 in 1.3 table | rebuild: `cc/build_engine/build_engine.py --target jetson-orin --source <tree> --commit <sha> --patch <patch>` (commands in each build-record.json) |
| llama.cpp source trees | `ref/llama.cpp` (31288c9c6), `ref/llama.cpp-up1108` (cf34d311332e), `ref/Strata` (504b1433) | - | commit ids | `git clone https://github.com/ggml-org/llama.cpp` + checkout |
Untransferred: everything in this table (by design: models/images are re-downloadable or rebuildable from committed recipes + checksums). All scripts, patches, manifests, raw logs/telemetry/results are committed.

### 1.7 Resume (if the campaign restarts)
1. `git fetch origin && git checkout jetson/c1`; read this file then START-HERE Jetson section.
2. Verify models by sha256 (1.4/1.6); `sg docker -c "docker images"` should list `campaign/llamacpp:jetson-*` (else rebuild per 1.6).
3. Highest-value next steps: (a) x86 to run D027 per-type unfuse on sm_120 (request R1); (b) SPEC2 chat arm with chat template + 2 more rounds; n-gram + `-lm mmap` default recommendation for copy-heavy workloads; (c) an aligned expert-read loader (pread 1.9 MB slices into pinned host buffers with >=1-layer lead time, using the J-OM7 host hook for IDs) - the only loader design not yet refuted; (d) KS q4_K rerun with `--types q4_K`; graph-level screen.
4. Use `cc/om_run/om_run.sh` / `cc/om_spec/om_spec.py` for every over-memory run (watchdog + IO telemetry); chain jobs by PID, never `pgrep -f` patterns.

### 1.2 Host
Orin Nano 8GB Super devkit (p3767-0005), L4T R36.5.2, JetPack 6.2.3, CUDA 12.6.68 (sm_87), kernel 5.15.199-tegra, 6x Cortex-A78AE, 7.4 GiB shared DRAM, zram swap 6 x 634 MB (~3.8 GiB), NVMe "X15 Lite SSD 128GB" PCIe Gen3 x4, nvpmodel MAXN_SUPER (queried only), GPU devfreq governor nvhost_podgov 306-1020 MHz (never pinned), nsys 2024.5.4, ncu 2024.3.1 (counters admin-only), Docker 29.1.3 (containerd snapshotter) + nvidia runtime, buildx v0.29.1 user plugin. Source: hosts/jetson-arm.json, handoff/jetson-checkpoint-1.md, D031.

### 1.3 Engine pins
| name | commit | notes |
|---|---|---|
| llama.cpp pin (campaign pin) | `31288c9c66e59377f7788d920678a565bb89dd37` (tree `158a7ade930818efdb06c5561920a8d018d404ba`, 2026-10-18) | `ref/llama.cpp`; all Granite work, D025/D027, routing trace, kernel screen |
| llama.cpp upstream worktree | `cf34d311332e362198f0a410c466d1e940287d02` (tree `2265e2c6dcf4294ba1cdc8b64ead0519925ef4f6`, 2026-11-08, "moe-cache : ensure CPU part does not wait for the cached experts (#31888)") | `ref/llama.cpp-up1108`; over-memory D031-D033, moe-cache, mmap advice |
| llama.cpp PR #31469 | `fc4a5a77a61b6ff965a5aa8b0f9e6d6a9c796583` | `ref/llama.cpp-pr31469`; build started and never recorded an outcome (see 2.2); study only |
| Strata (MIT, reference only, not built) | `504b1433` | `ref/Strata`; tool/method patterns only |

Note: the x86-side START-HERE lists the same 31288c9c6 pin; the upstream worktree is Jetson-only.

### 1.4 Models
| model | file | bytes | sha256 | source of hash |
|---|---|---|---|---|
| Granite 3.1 1B-A400M instruct, HF safetensors | `~/models/campaign/granite-3.1-1b-a400m-instruct/model.safetensors` | 2669283096 | not recorded on Jetson (x86 `artifacts/granite-0da7a48b.sha256`) | HF revision `0da7a48b0276d500ce5922fd2b33944091fc6c09` |
| Granite bf16 GGUF | `~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf` | 2672669600 | `bb010fb6aecc5636dabc5e95ba1f413ab645ed7b1ea5dc963d64237b78704e42`; tensor_data_sha256 `e1c1b4d5f2f4ae1967bb48c61694b45c6b169263fc67a6136dc6977f3dceec3d` (242 tensors) | experiments/jetson/bf16-gguf-manifest.json. x86's file sha is `af76ade03d516a82111cac78441e7882fb0392f3099f354dc55340477ff8f050`; tensor equality with x86 UNKNOWN (request R2 open) |
| Granite Q8_0 GGUF | `.../granite-3.1-1b-a400m-instruct-Q8_0.gguf` | 1422239840 | `a52205df70ad71f268af0690c07ce3e7bb4f148a8945e51115b9090558684bba` | computed with `sha256sum` while drafting this file (no earlier record found) |
| Granite mixE4K GGUF (experts Q4_K, rest Q8_0) | `.../granite-3.1-1b-a400m-instruct-mixE4K.gguf` | 818260064 | `5a83f2633ca3da16ad17e9ecbcbbbc14b1d79b875e90a6e34fab3b9de2bf55a5` | computed while drafting |
| Qwen3.6-35B-A3B UD-Q4_K_XL | `~/models/campaign/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (symlink `model.gguf`) | 22360456160 (20.8 GiB) | `707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450` | recorded value in evidence/jetson-arm/claude-code.jsonl; D029 says it equals the HF LFS oid. Not re-hashed (22 GB). HF `unsloth/Qwen3.6-35B-A3B-GGUF@a483e9e6cbd595906af30beda3187c2663a1118c` (handoff/x86-requests-2.md) |
| Qwen layout | experiments/jetson/OM/qwen36-layout.json | tensor bytes 22349466112; expert 19671285760; non-expert 2678180352; 40 layers x 256 experts, top-8; 1,900,544 B/expert (layer 1: 2,301,952) | - | gguf reader |
| bf16 base logits for KL | `~/models/campaign/kl/base-ub1-c128x8.kld` | not recorded | not recorded | local only, regenerate with `llama-perplexity -c 128 --chunks 8 -b 128 -ub 1 --kl-divergence-base` |

### 1.5 Docker images (read-only `docker images --no-trunc`, 2026-11-10 ~15:30)
With the containerd snapshotter `.Id` is the manifest/index digest (not the config digest) and RepoDigests are not pullable (handoff/manager-jetson-findings.md section 7). Config digests and archive sha are in each `cc/build_engine/results/<name>/build-record.json`. Binaries are NOT bit-reproducible across attempts (D026), so identity = record + patch sha + toolchain + base digest + archive sha, never binary hash.

| tag (`campaign/llamacpp:`) | local image Id | archive sha256 (engine-image.docker.tar) |
|---|---|---|
| jetson-c1 | `sha256:0b6c97b1ef6dab8523013f515c1b567f9ca9e16e991891e6785b6a571d98b1e1` | `323c787c1f2b5e33351403a528a8366232a0858588c74d9d2561bf9d69e254ac` |
| jetson-c1-j3 | `sha256:29d0041dc0975c6b85e1fc034ab703785bcaa592bfa564f11547cb07116ecf00` | `b730a81fcf08a90219ee2a83b18eac98d4e16014a305b6198f6f363b9e21e893` |
| jetson-c1-p01 | `sha256:5ded7cddc76c8e6627c8d3285f1a454106bd073c46c3d1ec26f8412faf32e6ea` | `5fb37f5bd797a724b6589c24743fcf7f294ae6e877e04f4a5a9128b80ac2da2f` |
| jetson-c1-rebuild | `sha256:cf8e40de68c93e1e5e6efa4637d2f26f355d0c4a392d879db826e3e7680e35cc` | `0991631ce22011cbddedd937e42d5a8d3e146d87011c5c412334c958aeaf2801` |
| jetson-c1-p01q | `sha256:ad20ebabe24f4cb81091929197e70fbd2fc141f9ee19e7f992de2a66ba931fea` | `3b42bcacfa5698a6ced401f211e67ff26afd014a928ac78ebdf5cfc1fb28b496` |
| jetson-c1-p03 | `sha256:09075b53c619c71b85bd23a1b3bf72eb4c0d5911db8e8327108189b75afa95a0` | `6158efc81e14fa11a3f9b12e9ab91404cd67328e2429f0bb863ac8da0a91dcb6` |
| jetson-c1-tr03 | `sha256:bec0080503a3313aa1555d12f959385968c7e6f8f88815701bb9225ea1159cce` | `75be40ee6b27c6b402d383851ba3b43bc69ceae5cbffbe8f8909f31c15145ef9` |
| jetson-up1108 (built by dir `jetson-up1108-a2`) | `sha256:7127623fe906093630e866e8f76f1b51cce32c18a94265f4c7ac323bcdcde87f` | `1db13059fce785df50beeb9a978214c6f334bfca50f4edad4b65d21d223ce75b` |
| jetson-up1108-mc | `sha256:69b8ae28bf49e93b7ce20190023dc5819977852aabbf4b78508df639239efda1` | `7220d4fc952aca07dc0e19daa29632aef127040cf18cac8ff13e1fc827bb12a4` |
| jetson-up1108-ra | `sha256:289ed0622cd1e6636ac8f746a9a3630083a32a788fe334d86d6ba54676b7a720` | `75ddea71785b14128c23e261d38856edf7e29cc9d7c63591ce0cd80a3f1dd3ca` |
| jetson-up1108-om7 | `sha256:4d0d6e6a04ffbd35816cc54d8aad5a484dda2a979bd18deb0c9e526f026527b6` | `fee04c8f34b50b12d25708c4e51bd34ce42e60553fbef791fe46db9a96684033` |
| jetson-c1-ks | `sha256:8ec53cfa6fbf8e9cd3f1e46a62903ade1eb8503546a86a2f28fb06b8b2e93cae` | `c9b06f037acbb6ddca85adcac49724f98356fdeb80156c67385d4c81da9c5d96` |
| jetson-pr31469 | not present (no record) | - |
| ubuntu:22.04 (D022 smoke, D028 probe base) | `sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11` | - |

Patch sha256 (full values are in the build records): 01-unfuse-bf16 `623340cc...`; 02-block128 `2d50eaa9...`; 03-unfuse-by-type `1369518f...`; combined-01-02 (moe_dispatch) `9cd5745b...`; combined-trace-p03 `c5e909e8...`; moe_cache_profile combined-01-02-03 `1215119e...`; mmap_advice 01 `1554b5b6...`, 02-router-prefetch `9fce108d...` (not in any finished build), combined-01-02 (mmap_advice) `631f1ed4...`; kernel_screen combined-p03-perf `72d7c8a0...`.

---------------------------------------------------------------------

## 2. Experiment and attempt ledger

Conventions: "steady" = mean of llama-bench samples 2..n (sample 1 is GPU clock ramp). "n" counts independent runs per arm. Outcome vocabulary: ok / failed / watchdog_killed / invalid / unscored / pending.

### 2.1 Granite experiments (experiments/jetson/)
| id | question / settings | image | outcome | key numbers | evidence | caveat |
|---|---|---|---|---|---|---|
| J-E001 | 2-chunk ppl, ctx512, -ngl 99, A/A x5, bf16 | jetson-c1 | ok | 7.5479 +/- 0.849 in 5/5 (spread 0); x86 7.5325 (+0.20%) | J-E001-ppl/a1..a5.txt | cross-host diff unexplained (kernels vs weights) |
| J-E002 | llama-bench bf16 pp512/tg128 r5 | jetson-c1 | ok | pp 1149.9 t/s (first sample 1020 = ramp); tg 50.2 +/- 0.08 | J-E002-bench/bf16.json, telemetry-bf16.summary.json | local baseline |
| J-E003 | does mmap double DRAM residency (mmap vs none, 2 pairs, page cache evicted per file) | jetson-c1 | ok (4 earlier attempts failed: `-mmp` flag not in this build) | cached +2.5 GB (mmap) vs +1.8 GB (none); MemAvailable min drop 1.92 vs 2.02 GB; tg r2 45-49 t/s (too noisy) | J-E003-mmap-residency/r1..r14 | hypothesis not supported; nvmap attribution unresolved |
| J-P001 | nsys decode profile | jetson-c1 | ok | GPU-kernel bound, CUDA graphs on; bf16 matvec 81%; fused MoE gate/up MMVF ~40 GB/s vs down MMF ~68 GB/s | J-P001-nsys-decode/FINDINGS.md | profile = hypothesis source |
| J-E004 | `GGML_CUDA_DISABLE_FUSION=1` ABAB tg128 r5 | jetson-c1 | ok | steady 50.4 -> 57.4 (+14%); means 49.69/49.96 vs 56.69/56.51; ppl20 10.0094 vs 10.0096; greedy 64 tok identical | J-E004-fusion-screen | all fusions off, attribution pending patch |
| J-E005 | patch A/B on bf16: base, P1 (gate/up->MMF), P2 (MMVF block128), F (all-fusion-off); order base,P1,P2,F,F,P2,P1,base | jetson-c1-j3 | ok | steady base 50.48/50.20; P1 60.07/60.05 (+19.3%); P2 59.56/59.30 (+18.0%); F 57.34/57.33 (+13.9%); depth1024 48.7/48.6 -> 57.8/57.8; pp512 unchanged (1105-1149) | J-E005-patch-ab/ | n=2 x r5/arm |
| J-E005q | decode-path quality (ub1 c128x8) | jetson-c1-j3 | ok | PPL base 8.0332 / P1 8.0163 / P2 8.0358; KL vs base logits: A/A mean ~0 (max 6e-5); P1 mean 2.7e-4, 99% 4.2e-3, max 9.1e-3, top-1 99.6%; P2 2.0e-4, 99.2% | J-E005-patch-ab/kl-*.txt, pplub1-*.txt | batch-512 ppl20 was identical (blind to decode path) |
| J-P002 | nsys kernel inventory base vs P1 | j3 (per D025) | ok | fused MMVF 1560 x 385 us -> MMF 3120 x 131 us; +3120 kernels, none dropped; GPU time -8.8% under tracing | J-P002-nsys-P1/*_kern_sum.csv | under-tracing number |
| J-E006 | quantized track Q8_0 / mixE4K (x86 recipes), plus all-fusion-off screen | jetson-c1-p01q (+quantize) | ok | Q8_0 1.42 GB pp 3204 tg 83.1 KLD 0.0041 top-1 97.0%; mixE4K 0.82 GB pp 3254 tg 84.7 KLD 0.060 top-1 87.1%; all-fusion-off Q8_0 -12%, mixE4K +14% | J-E006-quant-track/ | single model |
| J-P003 | nsys quantized decode | p01q | ok | fused Q4_K gate/up ~18 GB/s (about Q8_0 time for half the bytes) -> dequant/compute bound | J-P003-nsys-quant/FINDINGS.md | |
| J-E007 | per-type unfuse (patch 03), order base,on,on,base per model, tg128 r5 | jetson-c1-p03 | ok | bf16 50.4/50.3 -> 60.1/60.0 (+19.3%); mixE4K 85.1/85.0 -> 105.3/105.0 (+23.6%); Q8_0 83.4/83.2 -> 78.2/78.0 (-6.2%); mixE4K quality unfused vs fused identical (PPL 8.548401, KLD 0.060395, top-1 87.103%, text sha a2beba5c...) | J-E007-unfuse-by-type/ | n=2/arm; unpinned devfreq |
| SPEC1 | window-union replay (J-SPEC1) | - (CPU only) | ok (simulation) | experiments/jetson/SPEC1/{SPEC1.md,results.json}; `expert_replay.py specwin` (Codex J-SPEC1; 14 tests + 5 subtests pass). 864 scenarios (4 workloads x W 1-8 x a 0.5/0.7/0.9 x LRU/Belady x 650/1300/2600 slots x optimistic/pessimistic). Best W>1 case 1.0011x (optimistic, LRU, code, 650 slots, a=0.9, W=2); a=0.7, W=4, LRU, 1300: 0.62-0.63x; all pessimistic cases slower. Reading: once a page/expert cache exists, consecutive-token expert overlap is already captured as hits at W=1, so verify batching adds no IO dedupe, while rejected drafts add misses. Simulation only; traces are natural text (no copy-heavy workload), so it does not cover n-gram on repeat text. | artifacts/codex/J-SPEC1.final.txt | |

Other Granite-side attempts recorded in evidence but not under experiments/: J-EV1 run_op fixture (success, nonzero, non-UTF-8, background child, timeout; D021); J-ISO1 cgroup probe (D028); J5 lab jobs (below).

### 2.2 Build CC attempts (cc/build_engine/results/)
| dir | target / source | patch | outcome | wall | ccache hit/miss (this build) | evidence |
|---|---|---|---|---|---|---|
| jetson-c1 | pin 31288c9c6, no patch | - | ok | 3154 s (52.6 min) | not recorded (pre-J-T1b; later inventory says 0/454 hits) | build-record.json |
| jetson-c1-j3 | pin | combined-01-02 (9cd5745b) | ok | 2283 s | not recorded | |
| jetson-c1-p01 | pin | 01 (623340cc) | ok | 2262 s | 0 / 360 (cache populated) | |
| jetson-c1-rebuild | pin, no patch (reproducibility test) | - | ok | 141 s | 348 / 12 | binaries differ from jetson-c1 (D026) |
| jetson-c1-p01q | pin + llama-quantize | 01 | ok | 138 s | 347 / 15 | |
| jetson-c1-p03 | pin + llama-quantize | 03 (1369518f) | ok | 145 s | 350 / 12 | |
| jetson-c1-tr03 | pin + llama-routing-trace | combined-trace-p03 (c5e909e8) | ok | 136 s | 347 / 14 | |
| jetson-up1108 | upstream; requested commit `cf34d3113eb6d8e3a6c34bd8f4d1e9c5c3a7b9f1` | - | **failed at verify_source** (git rev-parse exit 128: mistyped commit hash; correct hash is cf34d311332e...) | 0.01 s | - | build-record.json (status failed) |
| jetson-up1108-a2 | upstream cf34d311332e (retry, tag `jetson-up1108`) + llama-quantize | - | ok | 2515 s | 19 / 305 (cold for the new tree) | |
| jetson-up1108-mc | upstream + moe_cache_profile 01+02+03 | combined-01-02-03 (1215119e) | ok | 519 s | 201 / 121 | |
| jetson-up1108-ra | upstream + mmap advice 01 | 01-mmap-advice-random (1554b5b6) | ok | 152 s | 306 / 16 | |
| jetson-pr31469 | PR #31469 `fc4a5a77a` (queued by build_queue.sh) | - | **dropped / no outcome**: intent+started ops (op-c63b2bd59050, 04:35:40) but no outcome record and no build-record.json; only build.stdout (ccache stats) and a staging tree remain. Reason not recorded. Study concluded reuse as ideas only (UMA triple copy, qwen4exp-only graph path; research/upstream-moe-cache-study.md section 4) | - | - | cc/om_run/build_queue.sh |
| jetson-up1108-om7 | upstream + mmap advice combined-01-02 (incl. router prefetch patch 02) | combined-01-02 (631f1ed4) | ok (first compile of patch 02 succeeded) | 2409 s (CUDA recompiled: patch touches ggml-backend) | - | build-record.json |
| jetson-c1-ks (kernel-screen image) | pin + p03 + perf cases + test-backend-ops | combined-p03-perf (72d7c8a0) | ok (pipeline5, after the pipeline4 deadlock) | 199 s | - | build-record.json |

Build-time facts: first native Jetson build 52 min; patched rebuild with ccache 2.3-2.4 min; BUILD_JOBS=3; each image archive ~1.39 GB (all archives and build/staging trees are gitignored, D024).

### 2.3 Lab jobs (J5; lab/jobs, summarized in handoff/manager-jetson-findings.md)
job-ecf12fd41a accepted+ran; job-0ec3b98dbb rejected (platform linux/amd64 x2, cuda_arch, missing runtime/l4t_major; `bin/lab run` exit 10); job-82c4a257d3 needs_information (model path missing); job-c7046d34b6 needs_information (unknown image record); job-5c142dd5df sandbox kill (exit 137 -> interrupted_sandbox_terminated, unscored, att-2 ok ppl 10.0094); job-43ef050ce9 runner SIGKILL (rerun refused rc 8, reconcile, att-2 ok); job-4433960e61 gen_smoke without -c picked n_ctx=45568 and 6568 MiB high-water (D023). Not exercised on Jetson: death during export, conflicting registration, cancel.

### 2.4 Over-memory experiments (experiments/jetson/OM/; Qwen3.6-35B-A3B UD-Q4_K_XL, 20.8 GiB on 7.4 GiB, run through cc/om_run/om_run.sh with watchdog + telemetry)
| dir | settings | image | outcome | key numbers | notes |
|---|---|---|---|---|---|
| feas-b1 | llama-bench `-lm mmap -ngl 99 -cmoe -p 64 -n 32 -r 1` | jetson-c1-p03 | failed | `-cmoe` is not a llama-bench flag | recorded failure |
| feas-b1-a2 | `-lm mmap -ngl 99 -ncmoe 40 -p 64 -n 32 -r 1` | p03 | unscored; result.json status `watchdog_killed` but exit_code 0 | pp64 2.49, tg32 1.02 t/s, wall 131.89 s; NVMe ~52 GiB in 132 s, peak 825 MiB/s (likely sampling artefact, D031), MemAvailable min 2.7 GiB | a C++ build ran concurrently (D029); the trip was a 33 MiB/s swap-rate limit; I infer the run survived because the kill command was broken (see watchdog-killtest) - status label is misleading |
| watchdog-killtest | forced trip on a `tail` container | p03 | watchdog_killed (intended) | killed only the owned container, exit 137, wall 0.49 s, 0 containers left | validates quoting fix `sg docker -c 'docker kill NAME'` (D029) |
| base-20261110-0435 | 6 configs, llama-bench with `-c` | p03 | **all 6 failed** (exit 1, `-c` invalid for llama-bench) | - | evidence of flag drift between tools |
| base-20261110-0448 | llama-bench `-p 128 -n 64 -r 2` | p03 | 3 ok, 3 watchdog_killed | ok: `-lm mmap -ncmoe 40` pp 4.49 tg 1.30 (wall 238 s); ncmoe 38 pp 4.67 tg 1.21 (244 s); `-lm mmap -ngl 0` pp 4.09 tg 1.09 (257 s). Killed: default `-lm auto` ncmoe 40 (MemAvailable 321 MiB at 7 s, wall 7.2 s); `-lm none` ncmoe 40 (401 MiB at 11 s); repeat of ncmoe 40 (zram swap-rate burst 286 MiB/s at 3.5 GiB available = false positive, wall 134.6 s) | n=1 per config; limits then raised (swap-rate to 1000 MiB/s) |
| up-20261110-0436 | llama-completion p1 (800-char wiki prompt) `-n 64 -lm mmap -ngl 99 -cmoe`; upstream moe-cache arms | jetson-up1108 | ok x5 | decode no cache 1.18/1.19; cache 1024 MiB 0.91; 2048 MiB 0.98; 2048 + cpu-share 0.5 1.13 | clean |
| upv-20261110-0519 | same with `-lv 4` counters | jetson-up1108 | ok x6, **timing contaminated** (CPU-heavy replay simulation at nice 0 during part) | no cache 1.21/1.16; 512 MiB 0.98 (hit 37.7%, uploaded 23.0 GB); 2048 0.98 (64.0%, 13.8 GB); 2048+cpu-share 1.0 0.87 (36.2%, 2.0 GB); 1024+cpu 0.5 1.02 (54.7%, 8.9 GB) | hit rates usable, speeds not |
| trace-20261110-0504 | routing trace, 4 workloads (wiki 251 s, code 225 s, math 207 s, chat 180 s wall) | jetson-c1-tr03 | ok x4 | routing.jsonl per workload (chat 1.25 MB); prompt/decode token counts not recorded in files I read | trace run is for capture only (cb_eval syncs); directories created chmod 777 because container writes root-owned files |
| cmp-20261110-0534 | `llama-bench -p 128 -n 64 -r 2 -lm mmap -ngl 99 -ncmoe 40`, order pin,up,up,pin | c1-p03 vs up1108 | ok x4 (pin-b was re-run; its first run tripped the swap-rate false positive) | decode pin 1.255/1.284, upstream 1.365/1.353 (+7%); prefill 4.52/4.24 vs 4.77/4.75 | partly overlapped nice-19 replay simulation |
| fio | O_DIRECT read-only on model file | host | ok x8 | rand4k qd1 19.6 MiB/s (181 us); rand4k qd32 74; rand128k qd8 475; rand576k qd1 416; rand576k qd8 536; rand2m qd4 543; seq1m qd8 573; 576k psync x4 535 MiB/s | drive-level ceiling (D031) |
| replay | expert_replay sweep, 4 workloads x 5 policies x 5 capacities (512/1024/1300/2048/3072 slots), leave-one-workload-out profiles | - (CPU) | ok | see 3.D | simulation only |
| mc-20261110-0602 | upstream + cache components, p1, `-n 64 -lm mmap -ngl 99 -cmoe -lv 4` | jetson-up1108-mc | ok x7 | base 1.14/1.20; cache2048 0.97 (64.0% hit); +dedupe 0.89 (22,437 madvise, 0 fail); +direct-fill 0.88 (**fill never activated: GGUF tensor offset not 4 KiB aligned, mmap fallback**); direct-fill + cpu-share 0.5 1.12; cache1024+direct-fill 0.93; NVMe 56.7-62.7 GiB/run in all arms | cache arms change output text (sha dc4b37... vs base 7a1541...); no KL check done |
| numa-20261110-0622 | base,numa,numa,base with `--numa distribute` | jetson-up1108 | ok x4 but **invalid test**: `ggml_is_numa()` is n_nodes>1 so the flag is inert on single-node Orin | eval 1.17/1.19/1.20/1.22, NVMe 55.4-57.0 GiB, identical output | any A/B read from it is meaningless (D032 addendum) |
| ra-20261110-0635 | base,rand,rand,base; `LLAMA_MMAP_ADVICE=random` (advice_log=1 only in rand arms) | jetson-up1108-ra | ok x4 | base prompt 5.68/5.69, decode 1.19/1.20, NVMe 56.6/55.7 GiB, 93/95 KiB per IO; rand prompt 1.42/1.47, decode 0.56/0.59, NVMe 24.1/23.5 GiB, 4 KiB per IO; output identical (7a1541...) | switch verified active |
| om7-20261110-1555 | base/pf/pfn ABCCBA with router-informed prefetch (`LLAMA_EXPERT_PREFETCH[_NEXT]`) | jetson-up1108-om7 | ok x6, ABCCBA (om7-20261110-1555): base 1.18/1.22 t/s (prompt 5.69, NVMe 57.2/55.1 GiB, 95 KiB/IO); prefetch 0.72/0.76 (prompt 1.38/1.44, 24.4/23.7 GiB, 5 KiB/IO; 20,823 advise calls, 13.5 GB advised, 0 failures); prefetch+next 0.64/0.65 (29.1 GiB, 6 KiB/IO; 50,343 calls, next-layer precision 614/20263 = 3.0%); output identical in all 6 (7a1541...). **Negative: -38% vs base**, +30% vs whole-file RANDOM (0.57). WILLNEED issued at the first CPU consumer has ~zero lead time, so faults still read 4 KiB pages; next-layer reuse of the current union is useless (3%). | |
| SPEC2-20261110-1634 | n-gram self-speculation ABBA on Qwen3.6 (cc/om_spec/om_spec.py) | jetson-up1108 | ok x4 servers (SPEC2-20261110-1634: llama-server, `-ngl 99 -cmoe -lm mmap -c 4096 -np 1`, n_predict 96, temp 0, seed 1, ABBA base/ngram-mod. rewrite: base 0.932/0.890 t/s -> ngram-mod 1.455/1.518 (**+63%**), draft 67/67 accepted; repeat: base 0.800/0.787 -> 1.534/1.590 (**+97%**), draft 72/72; text sha identical across arms (3f4bcde6..., 8377001f...); NVMe per request 54.9/58.1 -> 43.1/43.0 GiB (rewrite), 61.4/63.1 -> 40.0/40.6 GiB (repeat), IO size kept 81-101 KiB. chat arm INVALID: raw chat.txt prompt ends generation after 1 token in all arms (needs chat template). Server base decode (0.79-0.93) is below llama-completion's 1.2 (different harness/prompts); compare only within this run.) | first attempt SPEC2-20261110-1634-FAILED-pscheck-bug: my container-alive check used `docker ps -q` (IDs, not names) and aborted every arm before the health check |
| KS-20261110-1620 | kernel screen bf16/q4_K/q8_0 on test-backend-ops | jetson-c1-ks | KS-20261110-1620, image jetson-c1-ks (built 199 s). bf16 and q8_0 screened (MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION x n_tokens 1/8 x b 0/1, A/A2/B x3): all A/B ratios 0.96-1.10, inside the A/A spread (0.98-1.06); decode-shape (n_tokens=1) MUL_MAT_ID A/B 0.99-1.03 for both types. **The micro-screen does NOT reproduce D027** (+19.3% bf16 / -6.2% q8_0 end-to-end): the unfuse effect lives in graph-level fusion that single-op test-backend-ops cases do not exercise. q4_K arm FAILED: pipeline upper-cased the type to `Q4_K` and test-backend-ops matched no cases (pipeline bug, not rerun at closeout). | |
| (see Addendum c2) | | | | |
| qwen36-layout.json | per-expert slice layout | - | ok | see 1.4 | |

---------------------------------------------------------------------

## 3. Claims by class

Class key follows D012/NEEDS-MAP: full-engine improvement / quantization-offload trade-off / component benchmark / simulation / negative result. "Local" = Orin Nano sm_87 only.

### A. Full-engine improvements (local; x86 transfer untested)
| claim | numbers | evidence | validity caveats |
|---|---|---|---|
| A1 (D025) bf16 Granite decode: route MoE gate/up (batch 1) to MMF instead of fused MMVF with `GGML_CUDA_J_MMF_IDS=1` | tg128 steady 50.3 -> 60.06 t/s (+19.3%); depth 1024 +19%; pp unchanged | J-E005, J-P002, D025 | n=2 runs x 5 reps per arm; effect ~10 t/s vs A/A spread <=0.3 t/s; same image all arms (env switch only); NOT bit-exact: MMF rounds F32 activations to bf16; ub1 KL mean 2.7e-4, 99% 4.2e-3, max 9.1e-3, top-1 99.6% (x86 Q8_0-vs-bf16 for scale: 3.4e-3, 97.0%); devfreq unpinned; one host/model/precision; batch 1 only |
| A2 (D027) per-type switch `GGML_CUDA_J_UNFUSE_IDS_TYPES`: Q4_K experts | mixE4K 85.1/85.0 -> 105.3/105.0 (+23.6%); bf16 +19.3% reproduced; **Q8_0 -6.2%** (keep fused) | J-E007, D027 | outputs identical at measured precision (PPL 8.548401 both, KLD 0.060395, top-1 87.103%, greedy text sha equal); n=2/arm; other types (q2_K..q6_K, f16) unmeasured; rule is per type x platform |
| A3 (D031) upstream cf34d311 over-memory decode | pin 1.255/1.284 (mean 1.27) -> 1.365/1.353 (mean 1.36), +7%; prefill ~+9% | cmp-20261110-0534 | dependency update, not our code; n=2/arm, r2; pin arms showed larger spread; runs overlapped a nice-19 replay; unscored-adjacent; 519 upstream commits between pins, attribution unknown |

### B. Quantization / offload trade-offs
| claim | numbers | evidence | caveats |
|---|---|---|---|
| B1 representation frontier on Orin (J-E006 + D027): mixE4K 0.82 GB vs Q8_0 1.42 GB | before D027 decode equal (84.7 vs 83.1); after, mixE4K 105 vs Q8_0 83 (+26%) at 42% less memory; quality cost KLD 0.060, top-1 87.1% vs Q8_0 KLD 0.0041, 97.0% | J-E006, J-E007 | quality is the cost: mixE4K is not "free"; one model |
| B2 over-memory placement baselines (D030): which placement runs at all | `-lm mmap -ngl 99 -ncmoe 40` tg 1.30 (pp 4.49); ncmoe 38 1.21 (no gain); `-ngl 0` 1.09; default `-lm auto` and `-lm none` get watchdog-killed (MemAvailable 321/401 MiB) | base-20261110-0448 | n=1 per config, `-r 2`; Orin's `-lm auto` disables mmap on the iGPU so the default baseline is unfair - report both |
| B3 mmap does not double residency for a resident model (J-E003) | page cache reclaimable; no available-memory loss | J-E003 | 2 pairs; hypothesis not supported, CUDA/nvmap attribution unresolved |

### C. Component benchmarks and mechanism measurements
- C1 nsys kernel inventories (J-P001/2/3): mechanism for A1/A2 (fused MMVF 385 us vs 2 x MMF 131 us; Q4_K fused ~18 GB/s). Under profiling only.
- C2 NVMe ceiling (fio, D031): drive is 570 MiB/s class; 4 KiB qd1 19.6 MiB/s; PCIe Gen3 x4; one drive, read-only.
- C3 first-order model (D031): decode_tps ~ disk_BW / (615 MB x miss_rate); at 570 MiB/s and miss 1.0 -> ~1.0 t/s; measured 1.27-1.36 implies miss ~0.70-0.75. One model, a fit not a validation.
- C4 readahead accounting (D033): default mmap reads ~56 GiB/run at 93-95 KiB/IO, ~57% of bytes unused; whole-file MADV_RANDOM reads ~24 GiB at 4 KiB/IO but halves speed. Output identical. n=2/arm, switch verified active.
- C5 upstream moe-cache hit rates (D030/D032): 37.7% at 512 MiB, 54.7% (1024 + cpu-share 0.5), 64.0% at 2048 MiB. Hit rate counters are valid; the associated speeds in upv are contaminated.
- C6 telemetry/watchdog (J-OM1): 33 tests passed; kill path validated live only after the D029 quoting fix (watchdog-killtest).

### D. Simulation (cc/expert_replay on 4 routing traces; no engine validation of speed)
- Replay sweep (experiments/jetson/OM/replay/sweep.csv, 100 rows). At 1300 slots (12.7% of 40 x 256): decode hit rate by workload, wiki/code/chat/math: LRU 0.536/0.662/0.694/0.605 (mean 0.624); Belady 0.763/0.819/0.842/0.794 (mean 0.805); LFU 0.307/0.282/0.462/0.347; static-profile with leave-one-workload-out profiles 0.130/0.180/0.206/0.149 (mean 0.166). Means are computed by me from sweep.csv. At 2048 slots wiki LRU decode 0.657, Belady 0.848.
- Engine cross-check: upstream cache measured 64.0% at 2048 MiB (about 1077 expert slots at 1.9 MB). Ledger (factory-ledger.html) states replay ~54% at ~12% capacity versus 64% measured and calls that "held". Slot-to-MiB mapping and prompt equivalence between trace and engine run were not audited here.
- D031 projection "2.4-3.4 t/s if a cache holds bytes once, Belady bound ~7.8 t/s at 2048 slots" is a model output. D032 shows the real cache did not cut NVMe reads (57-63 GiB in all arms), so that projection was NOT realized by the components tested; it remains an upper-bound estimate for a design not yet built.
- Caveats: one trace per workload (token counts not recorded in files I read); simulator assumes an expert-granular cache, the real system is a 4 KiB page cache; static-profile result is cross-workload only (leave-one-out), not same-workload.
- J-SPEC1 window-union replay: experiments/jetson/SPEC1/{SPEC1.md,results.json}; `expert_replay.py specwin` (Codex J-SPEC1; 14 tests + 5 subtests pass). 864 scenarios (4 workloads x W 1-8 x a 0.5/0.7/0.9 x LRU/Belady x 650/1300/2600 slots x optimistic/pessimistic). Best W>1 case 1.0011x (optimistic, LRU, code, 650 slots, a=0.9, W=2); a=0.7, W=4, LRU, 1300: 0.62-0.63x; all pessimistic cases slower. Reading: once a page/expert cache exists, consecutive-token expert overlap is already captured as hits at W=1, so verify batching adds no IO dedupe, while rejected drafts add misses. Simulation only; traces are natural text (no copy-heavy workload), so it does not cover n-gram on repeat text.

### E. Negative and null results (all kept)
1. Upstream moe-cache on UMA is a net loss: decode 0.91-0.98 vs 1.18-1.21 (-17 to -23%) at 64% hit (D030); also in D032 (0.97 vs 1.14-1.20).
2. Removing the page-cache copy (uma-dedupe, 22,437 madvise, 0 failures) made decode slower (0.89), so the D030 double-residency hypothesis is not supported (D032).
3. Direct-fill never activated (alignment); no performance claim exists for it (D032). Fix idea J-OM6b (aligned-window read) not done.
4. GPU expert cache does not reduce NVMe traffic on UMA: 56.7-62.7 GiB per run in every arm (D032).
5. Whole-file MADV_RANDOM: half speed, though 57% fewer bytes (D033).
6. Router-informed WILLNEED prefetch (J-OM7, om7-20261110-1555): -38% (0.72-0.76 vs 1.18-1.22), bytes -57%, IO still 5 KiB; next-layer prediction 3% precision.
7. Kernel screen does not predict graph-level fusion effects (KS-20261110-1620).
8. (Not negative, listed for contrast) n-gram self-spec on copy-heavy prompts is the only over-memory win: +63%/+97% (SPEC2).
9. Window-union replay predicts no disk-bound gain from speculative decoding on natural text (J-SPEC1).
6. `--numa distribute` test: invalid, switch inert (D032 addendum).
7. Static hot set from other workloads: leave-one-out 16.6% vs LRU 62.4% decode hits (simulation).
8. Q8_0 unfuse: -6.2% decode on Orin (D027).
9. Container `--memory` is not a CUDA allocation bound on Tegra (D028): 1500 MiB device allocation completed under a 1 GiB cap 5/5; two early runs were killed at 1000/750 MiB, unexplained.
10. J-E003: mmap double residency not supported; kernel launch-overhead hypothesis (ledger) refuted by J-P001.
11. Own-error negatives: D024 (nearly pushed 1.4 GB archive), D029 (watchdog kill command broken), mistyped build commit (jetson-up1108), flag-drift failures (6 + 4 + 1 attempts), 8.7 h idle GPU from a pipeline deadlock (5.6).

---------------------------------------------------------------------

## 4. Predictions versus observations

Source: handoff/factory-ledger.html prediction register ("4 held, 7 missed or partial", of 11; hand-written at commit 6ae2616, before D031-D033), extended by predictions written in D029-D033 and assignments. Verdicts are mine from the cited files.

### 4.1 Ledger register (11)
| # | prediction | observed | verdict |
|---|---|---|---|
| 1 | Jetson decode is launch-overhead bound | kernel bound, CUDA graphs already on (J-P001) | wrong |
| 2 | mmap doubles DRAM residency for a resident model | page cache reclaimable, no available-memory loss (J-E003) | wrong |
| 3 | kernel fusion is a win in general | depends on weight type and platform (D027: bf16/q4_K lose-fusion, q8_0 keeps) | partial (counted as a miss) |
| 4 | half the expert bytes (Q4_K) gives faster decode than Q8_0 | no until the dispatch fix; dequant bound | wrong |
| 5 | unfusing quantized gate/up keeps numerics identical | byte-identical ppl, KL and text (D027) | correct |
| 6 | a container memory limit caps GPU memory on Jetson | not charged to the cgroup (D028) | wrong |
| 7 | an expert cache with high hit rate speeds up decode | 64% hits, 17-23% slower on UMA (D030) | wrong |
| 8 | routing has exploitable locality | replay ~54% decode hits at ~12% capacity; 64% measured | correct |
| 9 | routing-frequency profile gives a good static hot set | leave-one-out 16.6% vs LRU 62% at 12.7% | wrong |
| 10 | llama.cpp default cannot load a model larger than RAM on Orin | confirmed from source and by the watchdog | correct |
| 11 | over-memory decode ~ NVMe bytes per token / fault-read bandwidth | ~1.3 t/s predicted, 1.26-1.37 measured (note: the model was partly fit to those runs, D031) | correct |

### 4.2 Predictions made after the ledger or not in it
| prediction | source | observed | verdict |
|---|---|---|---|
| double residency (page cache + device bank) is the cost of the cache on UMA; dedupe would help | D030 hypothesis | dedupe slowed decode (0.89 vs 0.97) | wrong |
| UMA cache that holds bytes once -> 2.4-3.4 t/s | D031 | components tested did not cut NVMe reads | untested for a correct design / not realized |
| readahead pollution is the bottleneck; removing it helps (H-RA) | D032 | 57% of bytes wasted (confirmed) but speed halves (drive IOPS-bound) | half right: bytes yes, speed no |
| `--numa distribute` approximates H-RA with no code | D032 | switch inert | invalid / inconclusive |
| router-informed exact prefetch gives up to ~2x decode (IO 46 s vs 140 s) | D033 | 0.72-0.76 vs 1.18-1.22 t/s; bytes -57% but IO stays 5 KiB (no lead time) | **wrong** (as implemented) |
| direct-fill removes copy cost | J-OM6 | never activated | untested |
| kernel screen (test-backend-ops perf, seconds) predicts the D027 per-type rule | NEEDS-MAP/pipeline4 | bf16/q8_0 A/B within A/A noise; q4_K arm failed (type-name bug) | **wrong** for graph-fusion changes |
| (see Addendum c2) | | | |
| D025/D027 transfer to sm_120 | x86 request R1 | no x86 answer | untested |
| n-gram speculation helps a disk-bound decoder (x86 saw 2.3-3.0x on repeat text) | research/x86-c2-digest.md | J-SPEC2: copy-heavy +63% (rewrite) / +97% (repeat), text identical, NVMe bytes -27..-35%; J-SPEC1 replay: no gain on natural text; chat arm invalid | **correct for copy-heavy text** (smaller than x86's 2.3-3.0x); natural text untested live (replay says ~0) |
| replay LRU hit rate matches the engine counter | ledger "good predictor" | ~54% (12% capacity) vs 64% measured: close, slot-to-MiB mapping not audited | inconclusive-to-correct |
| first-order model fits other drives/models | D031 | one drive, one model | inconclusive |

---------------------------------------------------------------------

## 5. Lab Manager / CC portfolio discovery

### 5.1 Tools and commands used
bin/run_op (recorded commands; intent/started/outcome, survivors), bin/cc_hook, bin/lab (image add/submit/run/reconcile, host profiles + admission), bin/perf_summary, bin/ledger_appendix.py, cc/build_engine (target jetson-orin, ccache, `--extra-target`, `--with-tests`), cc/resource_observe (tegrastats/proc/cgroup/sysfs/io + watchdog), cc/om_run/om_run.sh (+ pipeline2-6, sweep_*.sh, build_queue.sh, fio_ceiling.sh), cc/expert_replay, cc/routing_trace (generic patch), cc/model_manifest/gguf_manifest.py, llama-bench / llama-perplexity (`--kl-divergence`, `-ub 1`) / llama-completion / llama-quantize, nsys (`--force-export=true` needed when a stale .sqlite gives a 0-byte CSV), tegrastats, fio, ccache 4.5.1, cmake/ninja from pip in ~/.local, docker + buildx, `codex exec` workers, git (`filter-branch` rescue, D024).

### 5.2 Missing tools and repeated manual work
- No "wait for job" primitive: every chain is a hand-written bash pipeline polling `pgrep -f` (pipeline2-6, build_queue). It broke once (5.6).
- Per-run summary extraction written as ad-hoc Python inline in pipelines (prompt/eval t/s from stderr, NVMe GiB and KiB/IO from telemetry, MemAvailable min, output sha) - candidate "run summarizer" CC.
- Flag capability probing: llama-bench lacks `-c`, `-cmoe`, `-mmp`; 11 attempts failed on flags (J-E003 x4, base-0435 x6, feas-b1 x1). Needs a `--help` capability probe before a queued run.
- Switch-effect verification: the NUMA switch was inert; only a log/counter line proved patch advice active. Need a standard "switch active" assertion per env-gated component.
- No A/A-aware acceptance gate in perf_summary yet (Strata autotune rule noted: median per-round ratio > max(3%, 2 x A/A gap, 2 x drift)).
- No launch identity record of the effective engine config (argv is recorded, effective config is not).
- Hand ordering of ABBA/ABAB arms and manual overlap avoidance with builds.
- Telemetry `host_id` field is the hostname (`satay`), not the campaign host id (`jetson-arm`) (D029; fix pending).
- Result classifier labels `watchdog_killed` while exit_code is 0 (feas-b1-a2).

### 5.3 Build and dependency problems
- System numpy is broken on Jetson; heavy converters run from `~/venvs/convert` (torch/transformers). Python tools stay stdlib.
- ccache first attempt had 0/454 hits because staging paths were random; fixed in J-T1b (CCACHE_BASEDIR, NOHASHDIR, compilercheck=content): rebuild 348/360 hits, 141 s vs 3154 s (D026). Binaries still differ per attempt (embedded `staging-...` path); `-ffile-prefix-map` not attempted.
- Docker installed by owner (docker.io 29.1.3, nvidia-ctk runtime); user is in group docker only for new sessions, so old sessions use `sg docker -c` (env `DOCKER_CMD_PREFIX='sg docker -c'`). buildx is a user plugin (v0.29.1).
- Host-built binaries have RUNPATH to deleted staging dirs, need `LD_LIBRARY_PATH`; images need `--runtime nvidia` even for CPU tools.
- Each archive ~1.39 GB; disk 88 GB free at inventory, ~62 GB later with a 22 GB model.
- D024: `git add` of a directory a build was writing nearly published a 1.4 GB archive.
- Nix rejected (Strata flake x86_64 only).

### 5.4 Permissions
sudo needs a password the agent cannot type (owner runs sudo). `ncu` counters blocked (RmProfilingAdminOnly=1). `jetson_clocks`/nvpmodel changes never run, so GPU clocks float (306-1020 MHz); mitigated with warm-up discard and counterbalanced A/B. Docker group, nvidia runtime config and apt installs were owner-authorized (D022).

### 5.5 Platform and memory
- Shared DRAM: no separate VRAM; NVML/nvidia-smi memory is [N/A]; per-client nvmap not visible; never sum `system-shared-dram` with process/cgroup scopes.
- cgroup/`--memory` cap does not bound CUDA allocations (D028); admission budget must say `guarantee: none-for-device-allocations`.
- llama.cpp auto-fit sizes context from "available memory", which on Tegra is system DRAM: unpinned `-c` picked n_ctx 45568 and 6568 MiB high-water (D023). Pin `-c`.
- `NVIDIA_VISIBLE_DEVICES=all` (+ `NVIDIA_DRIVER_CAPABILITIES`) must be set explicitly for generic arm64 images; otherwise "CUDA driver version insufficient" (D022).
- `-lm auto` disables mmap on iGPU (llama-model.cpp:1590, ggml-cuda.cu:5473), so default over-memory runs die; baseline must say `-lm mmap`.
- Zram swap bursts trip naive swap-rate watchdogs (false positives at 33 and 286 MiB/s); limits settled at MemAvailable < 600 MiB, zram > 2.5 GiB, swap rate 1000 MiB/s (cmp sweep), two consecutive samples.
- Storage: 128 GB budget NVMe at ~570 MiB/s is a first-order design input; record drive model + fio matrix in every host inventory (D031).

### 5.6 Scheduling and logging
- One GPU job at a time; builds must not overlap timed runs (D029 unscored; D030/D031 contaminated by nice-19 replay).
- **pipeline4 deadlock**: the wait loop `while pgrep -f "python3 cc/build_engine/build_engine.py|...pipeline3.sh|...pipeline2.sh"` matched the launching shell's own command line (per the comment in pipeline5.sh), so the loop never ended. The GPU idled about 8.7 h (06:34 to 15:15 on 2026-11-10) and the kernel-screen stage never ran. Fix: wait on a PID (`while kill -0 <PID>`; pipeline6.sh does this); pipeline5.sh was rewritten without pgrep.
- Logging: run_op records intent/started/outcome with host_id; per-host evidence streams `evidence/jetson-arm/` (D021, host id from untracked ~/.config/siminfeng/host_id); run_op non-UTF-8 echo crash fixed (rc 0 reported as 1); telemetry JSONL ~3 MB per 3-minute trace run (bulk, not for git).
- Networking/transport: no SSH between hosts; exchange is git branches `jetson/*` pushed by the operator; evidence durability is sandbox-survivable only (host loss loses data). x86 has not answered round 2.

### 5.7 Recovery
Sandbox kill, runner SIGKILL, duplicate refusal, reconcile, retry-as-new-attempt all demonstrated on Jetson (3.x/2.3). Watchdog kill quoting bug (D029): `--kill-cmd "sg docker -c docker kill NAME"` was split by shlex so `docker` ran bare; fixed with single quotes and a live forced-trip test (watchdog-killtest). Exit codes must come from run_op records, not pipelines (D022: a `| tail` masked a status).

---------------------------------------------------------------------

## 6. Candidate CCs

| family | candidate | concrete example from this campaign | integration requirements | status |
|---|---|---|---|---|
| build | `build_engine` with `jetson-orin` profile, ccache keying, `--with-tests`, `--extra-target`, record schema 2 | 13 attempts above incl. 1 failure at verify_source and 2 incomplete | one CC two native targets; commit/patch sha/toolchain/base digest as identity (not binary hash); name digests by role (local_image_digest, config_digest, registry_digest); validate that the requested commit exists before queueing; record ccache before/after | qualified on jetson-arm |
| build | build queue with PID-based dependency | build_queue.sh, pipeline2-6 | job IDs not pgrep patterns; one heavy build at a time; no overlap with timed runs | missing |
| profiling/test | `resource_observe` (+io, watchdog) | om_run telemetry: NVMe GiB, avg KiB/IO, refaults, MemAvailable, zram | scope field, null-with-reason, baseline required, `host_id` = campaign id | qualified (33 tests; live kill validated after fix) |
| profiling/test | guarded run wrapper (`om_run.sh`) | 48 guarded runs (result.json count in OM/), 5 results labelled watchdog_killed (one of them a forced test) | kills only owned container; limits as data; results.json status must separate "tripped" from "killed" | in use |
| profiling/test | kernel screen (`test-backend-ops perf` per op x type) | ran (KS-20261110-1620): did NOT predict D027 | exact type names (q4_K, case-sensitive); A/A arm; add graph-level fused cases (MUL_MAT_ID+GLU chain) or screen with a 1-layer llama-bench graph | ran; useful for kernel-local changes only |
| (see Addendum c2) | | | | |
| profiling/test | decode-path quality gate (ub 1 KL vs saved base logits) | J-E005: ppl20 identical, ub1 KL 2.7e-4 | stores base logits per model; bounded chunk count for large models | in use |
| tool | `expert_replay` + `routing_trace` generic | replay sweep, leave-one-out profiles | trace schema + completeness footer (Strata reader silently dropped truncated tails); slot-to-bytes mapping | qualified (10 tests; trace 56 tests) |
| tool | `gguf_manifest` | bf16 file sha differs from x86 | x86 to run the same on its file | implemented |
| tool | run summarizer / A/A-aware perf_summary | inline Python in pipeline5 | steady-mean rule, A/A gap, output-class gate (exact / numerically different / lossy) | missing |
| methodology | counterbalanced ABBA, A/A, discard sample 1, same image + env-switch arms | J-E005/E007 | record n per arm; classify the claim (D012) | in use |
| methodology | verify the switch reaches the code path (log line or counter) | D032 addendum (numa inert), D033 advice_log | every env-gated component logs "active" | adopt |
| methodology | shared-memory hosts: pin `-c`, declare memory domain, unmeasured free-memory defaults are host policy | D023 | admission flags unpinned defaults | adopt (admission check not done) |
| methodology | fair over-memory baseline (`-lm mmap`, `-cmoe/-ncmoe`) and report the auto default failure separately | D030 | baseline recipe in the test spec | adopt |
| methodology | do not read contaminated timing; mark UNSCORED | D029, upv | concurrency guard in scheduler | adopt |
| component | `moe_dispatch_sm87` patch 03 | +19.3% bf16, +23.6% Q4_K, -6.2% Q8_0 | env-gated default = upstream; per-type x platform applicability table; qualify on the target with the 3-model A/B | qualified on Orin; x86 untested |
| component | `mmap_advice` (01 random, 02 router prefetch) | 01 measured (D033); 02 built + ABCCBA (om7): -38%, output identical | opt-in env; startup log naming policy and consumer hook; counters at exit; no extra GPU sync | tested, negative on this drive; retain as an instrumented hook (exact per-layer expert IDs on host) for an aligned-read/pinned-buffer loader |
| component | `moe_cache_profile` (profile pin, dedupe, direct-fill) | D030/D032 negative | only for discrete GPUs; UMA double-copy; needs KL check because text changes | tested, negative on UMA |

### Upstream and external reuse
- llama.cpp: `--moe-cache-mib` (#29887), CPU-share (#30966), ClassLRU (#31233), CPU does not wait on cached experts (#31888, in cf34d311). Useful on discrete GPUs (x86 D044: +30..76% over static); a loss on UMA. `test-backend-ops perf` is the cheap screen (needs perf cases added, J-T6). Watch/adopt candidates from the external survey: #31957 (iGPU lazy-mode fallback only when mmap off), #31268/#30858/#28285 (Orin MMVQ), #31469 (study only, qwen4exp-only).
- Strata (MIT, reference): replay policies (LFU-decay, adaptive swap), profile building from a varied corpus, `quant_quality` top-64 renormalised KL, autotune acceptance rule, benchmark protocol IDs, launch identity, management protocol patterns. Engine code does not transfer (own runtime).
- Candidate reuse for the 128 GB / 4 TB target: expert-sized aligned reads and router-informed exact prefetch are the mechanisms worth re-testing on a fast NVMe; the Jetson drive is IOPS-bound.

### Approaches to avoid (evidence-backed)
- `MADV_RANDOM` on the whole model file (D033: half speed).
- Static cross-workload hot sets (leave-one-out 16.6% vs LRU 62.4%).
- Cgroup/`--memory` caps as a GPU or shared-DRAM budget on Tegra (D028).
- `--numa distribute` on a 1-node host as a readahead switch (inert).
- Copying experts into a separate device bank on UMA (D030/D032).
- Treating hit rate as speed on unified memory, batch-512 ppl as a decode-kernel check, binary hash equality as "same build" (D026), raw t/s across hosts.
- Unfusing Q8_0 gate/up on Orin (-6.2%).
- `pgrep -f`/`pkill -f` patterns launched from a shell whose command line contains the pattern (pipeline4 deadlocked ~8.7 h; a closeout `pkill -f` also killed its own shell).

---------------------------------------------------------------------

## 7. Open items, gaps and placeholders

- KS q4_K arm not rerun (type-name bug in pipeline5; use `--types q4_K`).
  (see Addendum c2)
- SPEC2 chat arm needs a chat-templated prompt (`/v1/chat/completions`) and >=2 rounds; MTP not tried (J-SPEC1 predicts ~break-even; would need the 22.66 GB -MTP GGUF).
- Gaps found while drafting: Granite safetensors sha and the KL base file sha not recorded on Jetson; Qwen file sha not re-verified locally; trace token counts not recorded; J-OM6b not done; why pr31469 was dropped not recorded; early D028 kills unexplained; no x86 reply to R1-R6 (sm_120 transfer of D025/D027 is the highest-value open test); J-T2 Codex run in the artifact log ended with API 401 errors, how the resource_observe code was then completed is not documented in files I read; the factory-ledger register predates D031-D033 and needs regeneration.
- Honest status: two local full-engine gains (A1, A2) and one dependency gain (A3) exist; no over-memory *loader/cache* component of ours has beaten the baseline (`-lm mmap -cmoe`, ~1.2-1.36 t/s). The only over-memory speedup is upstream n-gram self-speculation on copy-heavy prompts (SPEC2: +63%/+97%, text identical), which is a decoding-strategy gain, workload-dependent.


## Addendum c2 (2026-11-11)

The completed c2 engine series, including J-OM10c, supersedes the earlier statement that no local over-memory loader beat baseline. These results use same-image switches, counterbalanced arm orders, measured A/A gates and identical output text. They apply to the tested Qwen prompt/model on this Jetson and drive. D034–D036 preserve the mechanism, qualifications and corrections.

| run | intervention | result | output / decision | source |
|---|---|---|---|---|
| J-OM8 | selected-expert pread fill | base 1.185; fill 1.33; fill_random 1.39 t/s; B/A 1.122 / 1.173; A/A gap 0.0928, threshold 0.186; null | identical; exploratory, D034 | `experiments/jetson/OM/J-OM8/REPORT.md`, `verdict.json` |
| J-OM8b | decode-only random + fill, frd16 | base 1.1875 → 1.3025 t/s; B/A 1.097; A/A gap 0.0471, threshold 0.0941; improved | identical; preserve prefill, D034 | `experiments/jetson/OM/J-OM8b/REPORT.md`, `verdict.json` |
| J-OM9 | reproduce fill; diagnose faults; POPULATE_READ | base 1.195; frd16 1.295; diagnostic 1.18; populate 1.2 t/s; A/A gap 0.0232; fill improved, populate null | identical; synchronous fill IO bottleneck, D034 | `experiments/jetson/OM/J-OM9/REPORT.md`, `verdict.json`, `003-frd16d/stderr.txt`, `004-frd16p/stderr.txt` |
| J-OM10 | host-router next-layer prediction | base 1.185; frd16 1.29; pg8 1.23; pg16 0.75 t/s; A/A gap 0.0163 | identical; prediction works but first scheduling loses against fill, D035 | `experiments/jetson/OM/J-OM10/REPORT.md`, `verdict.json` |
| J-OM10b | demand-first, cancel stale queue, cap async occupancy | base 1.165; frd16 1.285; pg8b 1.42; pg12b 1.285 t/s; pg8b B/A 1.219; A/A gap 0.0258; improved | identical; retain pg8b opt-in, D035 | `experiments/jetson/OM/J-OM10b/REPORT.md`, `verdict.json` |
| J-OM10c | occupancy sweep, completed confirmation | base 1.19; pg8b 1.385; pg8i8 1.39; pg8i12 1.38; pg10i8 1.4 t/s; A/A gap 0.0217; all improved vs base | identical; pg10i8 actually logs K=8, no established incremental occupancy win, D035 | `experiments/jetson/OM/J-OM10c/REPORT.md`, `verdict.json`, `005-pg10i8/summary.json` |
| KS-q4K-r3 | canonical type name + explicit job environment | 108 runs; A/B 0.91–1.06 within A/A 0.94–1.18 | success; micro-screen does not predict D027 graph unfuse, D036 | `experiments/jetson/KS-q4K-20261111-r3/q4_K/results.md`; `c2/worklog.jsonl` KS-q4K-r3 result |

Each abbreviated source after a full experiment path is relative to that same experiment directory. Per-arm sample counts and the exact source ratios are in D034/D035 and the cited verdict inputs. J-OM8b has four observations per arm; the other listed OM experiments have two. The OM output hash is `7a1541792364` throughout (cited reports). Gates compare each arm with that experiment's base; they do not prove differences between nearby candidate arms.

Harness components added:

- `bin/jobq`: durable job IDs, dependencies and resource queues replace hand-written pgrep chains; captured submission environment determines job behavior. H1 checkin passed after the lead moved full environment out of public job JSON into a private side file (`c2/worklog.jsonl`, H1 checkin; `bin/README-jobq.md`).
- `cc/abrun`: explicit counterbalanced arms, summary extraction, active-switch checks, A/A/drift-aware verdicts and output identity classification. H2 checkin reproduced the earlier negative prefetch verdict; J-OM8b's checkin notes that an OR switch regex required post-hoc decode-marker verification (`c2/worklog.jsonl`, H2 and J-OM8b checkins).
- `c2/shape_report.py`: regenerates `c2/SHAPE.md` and `c2/shape.json` from evidence and worklog. H4 checkin identified incomplete flag-error classification and idle-gap bounds; H2b-H4b follow-up is recorded. Treat accounting as measured evidence with warnings, not an exact reconstruction of all historical work (`c2/worklog.jsonl`; `c2/test_shape_report.py`).
- `c2/METHODS.md` and `c2/skills/`: campaign methods and reusable build, experiment, profiling, over-memory and delegation recipes. H3 checkin passed with the qualification that the historical flag-failure count was sourced only from CLOSEOUT (`c2/worklog.jsonl`, H3 checkin; `c2/skills/README.md`).

Corrections to the historical sections above:

- KS attribution in the results table and prediction/candidate rows is affected by the tool's case-folding bug: `q4_K` became `q4_k` inside kernel_screen, independently of pipeline upper-casing. The fabricated fixture repeated the same error. The canonical type map and corrected fixture address the tool defect; passing the correctly cased argument alone was insufficient. See D036, the current `cc/kernel_screen/kernel_screen.py` git diff and the KS-q4K failure event in `c2/worklog.jsonl`.
- The q4_K rerun open item is closed by r3. r1 was invalid from case folding; r2 was invalid because DOCKER_CMD_PREFIX was absent from the captured job environment; r3 succeeded. The graph-level screen gap remains (`c2/worklog.jsonl`, KS failure/result events).
- The old “no over-memory loader/cache component beat baseline” statement is superseded by fill and scheduled pregate results above. The original no-lead-time prefetch failure remains valid for that implementation. D033's projected gain was not established by this series.
- J-OM10c's `pg10i8` name describes a requested K=10, but the active value is K=8. The parser only recognizes the other supported overrides; use the active log, not the arm name, to identify the experiment (`cc/mmap_advice/patches/06-pregate.patch`, `07-pregate-sched.patch`; `experiments/jetson/OM/J-OM10c/005-pg10i8/summary.json`).
- POPULATE_READ removes many downstream faults but adds its own population cost; it is net zero against baseline here. Diagnostics add overhead, and process-wide fault counters include driver threads (D034; J-OM9 result/checkin events in `c2/worklog.jsonl`).
- Scheduler, summarizer and A/A-gate entries previously marked missing now have c2 implementations above. This does not imply every historical harness gap or cross-host qualification is resolved. The original tables remain a historical snapshot.
