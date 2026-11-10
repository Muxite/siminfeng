#!/usr/bin/env python3
import json
from pathlib import Path
O=Path(__file__).resolve().parent
r=json.loads((O/'analysis.json').read_text())
c=json.loads((O/'captures.json').read_text())
lines=['# Qwen3-30B-A3B routing skew and placement forecast (T-QRT)', '',
'## Result', '',
'The observer is shared across compatible MoE graphs, but the original tool was limited to 32 expert IDs and top-8. It could not trace this Qwen model. The additive patch reads `qwen3moe.expert_count` for Qwen3 only; Granite retains its original behavior. The graph, router, experts, backend and sampling math are unchanged. `model_inspect` already handles 128 experts and needed no edits. `trace_replay.py` still assumes Granite/32 and was not used.', '',
'## Capture and validation', '',
'Source: `31288c9c66e59377f7788d920678a565bb89dd37`, archived into `artifacts/codexQRT/tree`. CPU-only build, `GGML_CUDA=OFF`, `GGML_NATIVE=OFF`, `cmake --build ... --target llama-routing-trace -j4`, under `nice -n 10`. `configure.log`, `build.log`, `build-flags.txt`, `git-apply-check.log`, `patch-apply.log` and `build.sha256` retain build evidence. The final patch passes `git apply --check` against the pinned examples CMake file with new-file paths absent. `reference-status.txt` is empty (tracked reference tree clean).', '',
'Each capture uses mmap, no repacking, `-ngl 0 -t 8 -tb 8 -b 128 -ub 128 -c 4096 -n 64`, under nice. One model process ran at a time, after the compiler finished. Commands, prompt hashes, exit codes and elapsed times are in `captures.json`; `capture.py` and `capture.log` reproduce the serial capture. Raw prompts have no added chat template. Greedy generation evaluates exactly 64 tokens even if EOG occurs. All prompt logits are requested to prevent last-layer pruning. Observer synchronization and all-logits prefill perturb performance; capture elapsed times are not baseline throughput.', '',
'Model SHA256, independently computed from the local GGUF: `9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48`.', '',
'| Trace | Split | Prefill tokens | Decode tokens | Rows | Bytes | SHA256 |',
'|---|---|---:|---:|---:|---:|---|']
for item in c:
 v=r['validation'][item['name']]
 lines.append(f"| {item['name']} | {item['split']} | {v['tokens']['prefill']} | {v['tokens']['decode']} | {v['rows']} | {item['trace_bytes']} | `{item['trace_sha256']}` |")
lines+=['',
'All three exited zero. `analyze.py` checks eight distinct in-range IDs, one sequence, contiguous positions, exactly 64 decode positions, no duplicate token/layer record, and every position in all 48 layers. It cross-checks every phase/layer histogram against the official inspector. The required `python3 cc/model_inspect/inspect_model.py <gguf> --trace <trace>` ran for each capture, producing `prose-inspect.json`, `chat-inspect.json`, `code-inspect.json`, including complete per-layer coverage curves. `pytest.log`: 64 passed (existing tracer replay and model-inspection suites).', '',
'## Fitting and exact byte budgets', '',
'Hot experts are ranked separately for each layer using only prose TRAIN prefill request counts, with ascending expert ID on ties. Prose decode is diagnostic only. Chat/code prefill and decode never choose hot sets or allocations. Coverage counts unweighted (token, expert) requests, not mixture weight mass. Evaluation decode pools 64 chat and 64 code tokens with equal request weight. This is a small two-prompt evaluation, not a representative workload guarantee.', '',
'The GGUF has 16,740 MiB of expert weights and 951.34 MiB of other weights. Its 24 layers with 2.53125 MiB/expert and 24 with 2.91796875 MiB/expert average 2.724609375 MiB. Static K means the first K layers put expert tensors on the CPU while other weights stay on the GPU; thus the comparable expert budget is the sum of expert bytes in layers K..47. Runtime/KV/compute/staging buffers are excluded from both analytical weight budgets. D039 GPU deltas include them and are listed separately.', '',
'For each byte budget, start every layer at H=floor(budget / sum(per-layer expert sizes)), then spend the remaining bytes on the next TRAIN-ranked expert with highest request-count/byte benefit until no expert fits. This is a simple nearly uniform allocation, not a claim of global optimal placement. H ranges below are the actual per-layer counts. Exact IDs and byte slack are in `analysis.json`. Nominal equal-sized-expert counts would be 106.67, 80, 48, 16; actual quantized layer sizes change these.', '',
'| K | D039 GPU delta MiB | Expert budget MiB | Used expert MiB | H range | Remaining MiB |',
'|---:|---:|---:|---:|---:|---:|']
for row,gpu in zip(r['results'],[15684,12246,8162,4074]):
 lines.append(f"| {row['K']} | {gpu} | {row['expert_budget_MiB']:.2f} | {row['used_expert_MiB']:.2f} | {row['H_min']}-{row['H_max']} | {row['unused_bytes']/2**20:.3f} |")
lines+=['', (O/'table.md').read_text().strip(), '',
'All coverage columns above use the same TRAIN-fitted hot sets, including EVAL prefill. The full inspector coverage curves rank each individual trace for descriptive statistics; those EVAL-fitted curves are not used for the held-out table.', '',
'## Decode forecast and crossover', '',
f"D039 fit: c=(1/24.4 - 1/93.5)/40 = {r['c_fit_ms']:.6f} ms/CPU layer; t_gpu=1/93.5 - 8c = {r['t_gpu_ms']:.6f} ms/token. The requested 2.72 MiB expert / 30 decimal GB/s model costs {r['byte_layer_ms']:.6f} ms per fully cold top-8 layer.", '',
'For the main forecast, in milliseconds: `t = t_gpu + 48*(1-heldout_decode_coverage)*8*2.72*2^20/30e9*1000 + 48*t_handoff`. All 48 layers contain CPU-side cold banks, so the main table charges 48 handoffs even when a token selects only hot experts. An optional miss-aware implementation could avoid some boundaries; observed CPU-active layers and that optimistic 50 us variant are retained in `analysis.json`.', '',
'D031/T-PROF adds 371.13 us of exposed sync/copy time for 12 extra CPU boundaries: 30.93 us/boundary. Its split total is 812.10/24=33.84 us/boundary. Those motivate 30 us as a lower estimate. Residual host time (+214.40 us/12=17.87 us) cannot be separated from CPU work/scheduling, and Qwen activations are larger. Use 50 us centrally and 80 us as a sensitivity upper bound. This is an assumed engineering range, not a measured Qwen confidence interval. The old per-slot GPU dispatch is excluded: the forecast assumes the T-PROF batched skip-ID dispatch design preserves roughly the fitted GPU term. Any new GPU work consumes the same benefit margin.', '',
'D039 c includes aggregate CPU-layer overhead and only approximately identifies a pure bandwidth cost. Its intercept is not an independently measured all-GPU run (the whole model does not fit). Adding a separate handoff term to the byte estimate is conservative relative to the original aggregate fit, but does not remove this identification uncertainty. Actual Q4 type bandwidth, per-expert fixed CPU overhead, GPU hot work, overlap, host contention and memory allocation can change the forecast. No CUDA hybrid was implemented or measured.', '',
'| K | Held-out decode | Hybrid t/s at 50 us | Static measured t/s | Required coverage at 50 us | Max affordable handoff us | CPU-active layers observed |',
'|---:|---:|---:|---:|---:|---:|---:|']
for row in r['results']:
 lines.append(f"| {row['K']} | {row['eval_decode_coverage']:.2%} | {row['predicted_tps']['50']:.1f} | {row['static_tps']:.1f} | {row['coverage_crossover_at_50us']:.2%} | {row['critical_handoff_us']:.1f} | {row['observed_cpu_active_layers']:.2f} |")
lines+=['',
'The coverage crossover sets the requested latency equation equal to each measured static point. The maximum handoff is the residual latency margin divided by 48; dispatch overhead beyond t_gpu must fit inside that same margin. K=8 and K=18 already lose at zero handoff. K=30 crosses parity at 44.44% coverage for a 50 us handoff; measured held-out coverage is only 44.96%. Its entire remaining margin is about 0.193 ms/token. K=42 requires 20.39% coverage at 50 us but achieves 15.26%; it would need handoff below 11.02 us, outside the estimated range. Thus there is no robust crossover across the requested budgets.', '',
'Each domain is itself skewed: mean per-layer prefill entropy is 5.471 bits for prose, 5.488 for chat, and 5.410 for code, versus a 7-bit uniform maximum; separately fitted descriptive hot sets reach 90% coverage at mean H=42.92, 41.02, and 43.10 respectively. This does not mean prose-selected experts transfer to code. Low cross-domain coverage is evidence against this particular TRAIN policy, not proof that all expert placement policies fail.']
lines+=['', '## Recommendation', '', 'Do not prioritize a general qwen3moe CUDA hybrid on this TRAIN/EVAL evidence. At 30-80 us handoff, the pooled forecasts lose at K=8,18,42; K=30 spans 35.0-38.2 t/s versus static 36.6, with only a 0.7% central gain. The required batched skip-ID dispatch is a prerequisite, but its unknown incremental cost can erase that small gain. First validate hot sets trained on a larger, diverse and deployment-matched corpus against longer held-out continuations. A domain-specific prototype could be reconsidered if held-out coverage clears the listed thresholds with enough margin to pay dispatch, allocation and handoff costs. Preserve original selected-slot order/numerics, batch CUDA skip-ID expert dispatch, and measure actual Qwen handoffs before claiming speedup. This assignment measures routing and forecasts; it does not validate a hybrid implementation.', '',
'## Reproduction', '',
'After applying the patch and building the target as in `cc/routing_trace/README.md`, run:', '',
'```bash',
'PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 artifacts/codexQRT/capture.py',
'PYTHONDONTWRITEBYTECODE=1 python3 artifacts/codexQRT/analyze.py',
'PYTHONDONTWRITEBYTECODE=1 python3 artifacts/codexQRT/write_report.py',
'```', '',
'Writes for this assignment are confined to the additive `cc/routing_trace/{routing-trace.patch,README.md}` changes and `artifacts/codexQRT/**`. No Docker, GPU execution, commits or changes to the reference source were used.', '']
(O/'ANALYSIS.md').write_text('\n'.join(lines))
