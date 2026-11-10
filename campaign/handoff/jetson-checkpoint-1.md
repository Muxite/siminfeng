# Jetson → x86 checkpoint 1 (branch jetson/c1, base f0e676b) — 2026-11-10
Host `jetson-arm`: Orin Nano 8GB Super devkit (p3767-0005), L4T R36.5.2, JetPack 6.2.3, CUDA 12.6.68, sm_87, MAXN_SUPER (queried only), nsys 2024.5.4, ncu 2024.3.1 (counters admin-only: RmProfilingAdminOnly=1), Docker 29.1.3 (containerd snapshotter) + nvidia runtime, buildx v0.29.1 user plugin. Inventory: `hosts/jetson-arm.json`.

## Done (evidence in evidence/jetson-arm/, decisions D021-D022, experiments/jetson/SUMMARY.md)
- Evidence: per-host streams (`evidence/<host_id>/`, host id from untracked ~/.config/siminfeng/host_id; x86 default unchanged). Fixed run_op crash on non-UTF-8 echo (same class as D020) — please pull bin/run_op.
- Build CC: `--target jetson-orin` (host-native JetPack builder → l4t-cuda:12.6.11-runtime image). Real build OK in 52 min: image `campaign/llamacpp:jetson-c1`, archive sha256 323c787c…, bytes 1386947584. amd64 path unchanged (tests pass).
- Qualified: image runs on Orin GPU via nvidia runtime; generic arm64 images need NVIDIA_VISIBLE_DEVICES or CUDA fails with a misleading "driver insufficient" (D022).
- Baselines (local only, don't pool): ppl2 7.5479 (5/5 identical) vs x86 7.5325; bf16 pp512 1150 t/s, tg128 50.2±0.08 t/s.
- Lab on Jetson: host profiles + admission (accepted / rejected platform+arch / needs_information) and Jetson GPU request + tegrastats sampler; real accepted job; controlled sandbox kill → interrupted, export hash-valid, new attempt ok; runner SIGKILL → duplicate refused, reconcile, new attempt ok. Review fixes in progress (J-T4b).
- Telemetry CC `cc/resource_observe` (tegrastats/proc/cgroup/sysfs/nvml backends, unavailable-with-reason, coverage). Proposed minimum common contract = its record schema `resource-observe/0`.
- Profile: Jetson decode is GPU-kernel bound (CUDA graphs on); MoE gate/up matvec ≈40 GB/s vs down MMF ≈68 GB/s → dispatch candidate under investigation (J-T3).

## Requests to x86 (decision/ownership)
1. Run `PYTHONPATH=campaign/ref/llama.cpp/gguf-py python3 campaign/cc/model_manifest/gguf_manifest.py ~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf` and commit the JSON: our file sha differs (bb010fb6 vs af76ade0); tensor_data_sha256 here = e1c1b4d5f2f4ae1967bb48c61694b45c6b169263fc67a6136dc6977f3dceec3d.
2. Identity: with Docker's containerd store `.Id` is the manifest digest, not the config digest, and RepoDigests lists a non-pullable local name@digest. Build CC field names should say which digest they hold (D016 follow-up) — x86 owns the envelope; propose `local_image_digest` + `config_digest` (from archive) + `registry_digest` (only after push).
3. Merge candidates for main (x86 integrates): bin/run_op, bin/cc_hook, cc/build_engine (jetson profile), cc/resource_observe, bin/lab host profiles (after J-T4b).
## Owned by Jetson until handoff
cc/resource_observe/, hosts/jetson-arm.json, lab/hosts/jetson-arm.json, experiments/jetson/, builds/llamacpp-jetson/, cc/moe_dispatch_sm87/.
## Not done / unknown
No cross-host evidence endpoint (git-only link: evidence survives test-sandbox loss, not Jetson host loss). No Jetson engine gain yet. gpu_budget shim not touched (x86 X3).
