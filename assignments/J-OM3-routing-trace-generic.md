# Assignment J-OM3 — routing trace capture for any MoE (qwen35moe first) (host jetson-arm; parent over-memory track; worker Codex)
Base: jetson/c1. Existing x86 asset: cc/routing_trace/ (routing-trace.patch hooks `ffn_moe_topk-N` / `ffn_moe_weights*-N` from build_moe_ffn; hardcodes n_expert=32, k=8 → aborts on Qwen3.6 with 256 experts). Read its README, patch, trace_replay.py and research/own-repo-assets.md. Pinned llama.cpp ref/llama.cpp 31288c9c6 (read-only; `git apply --check` allowed).
Write `cc/routing_trace/patches/routing-trace-generic.patch` (standalone on 31288c9c6, may build on the existing patch's tool):
- Derive n_expert and k per layer from the hooked tensors' shapes (never hardcode); record arch, n_layer, n_expert, k, model path/sha (if cheap) in a header line.
- Emit JSONL compatible with cc/expert_replay input: `{"tok":i,"layer":l,"experts":[...],"weights":[...],"phase":"prefill|decode","seq":s}` plus a final `{"trace_complete":true,"tokens":N,"records":M}` footer written only on clean exit, so truncation is detectable. Flush per token; bounded buffer.
- Hybrid models (qwen35moe has DeltaNet + attention layers; all layers have MoE? check llama-model.cpp build for qwen35moe and state which layers emit routing) — layer index = model layer id, not hook counter.
- Shared expert(s) (always active) — note they are not routed; record presence in header.
- Keep the capture tool target name (check the existing patch; e.g. `llama-routing-trace`) so Build CC `--extra-target` can build it.
- Update cc/routing_trace/README.md (stale "not compile-linked" claim per own-repo-assets.md) and add a Jetson section.
- Tests: Python test validating the JSONL schema/footer logic with fixture lines; `git -C ref/llama.cpp apply --check` result.
No builds/GPU (Opus builds with ccache). Time 40 min. Final ≤ 12 lines: patch, tool name, layers emitting routing for qwen35moe, apply --check result, risks.
