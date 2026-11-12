# F13 — expert streaming: readiness observability (patch 10) + in-flight join (patch 11)

Repo: `/home/muk/projects/xs-dev/llama.cpp` (branch `xs-dev`, a private clone at `6e9d96cf4` = llama.cpp `cf34d3113` + patches 01–08 of the expert-streaming stack). Work ONLY in this clone. Do NOT touch `~/.jiuwenswarm-factory`, Docker, the GPU, or any other repo. No network.

Read first:
- `/tmp/claude-1000/-home-muk-projects-siminfeng/f788ee04-950c-45f8-ae6d-aa6e956451d3/scratchpad/expert-streaming-sourcemap.md` (source map with file:line anchors; sections 4–8 are the important ones).
- `src/llama-expert-fill.h` (ExpertFiller, demand `fill()`, async queue, pregate counters), `src/llama-context.cpp` (`expert_prefetch_cb`, `expert_config: effective` line, strict `LLAMA_EXPERT_*` env validation from patch 08), `src/llama-model.cpp` (`expert_file_info`).

Background (what evidence is missing today):
- Pregate counters tell us whether the right expert was *predicted* (`hits`, `recall_at_k`) and the async IO state at arrival (`hit_done/hit_inflight/hit_queued/hit_resident`), but not whether the selected expert's **bytes were actually resident at demand time**, split by predicted vs unpredicted. Without `DIAG=1` (which perturbs timing) there is no residency measure.
- In-flight async pages look non-resident to `mincore`, so demand `fill()` can issue **duplicate preads** for ranges an async worker is already reading (`hit_inflight` case), double-counting `bytes_read` and wasting IO bandwidth on the critical path.

## Patch 10 — `expert_readiness` (observability, switch `LLAMA_EXPERT_READINESS=0|1`, default 0)
When enabled (and fill active), at each layer arrival in the callback, **before** demand fill reads anything, classify each *selected* expert's page-rounded byte ranges as resident / non-resident using the residency probe the demand path already performs (reuse it; do not add a second full mincore pass if avoidable), and split by whether that expert was in the pregate prediction set for this layer (unpredicted if pregate off or layer 0). Also count: async bytes read for predicted-but-not-selected experts (wasted prefetch), and demand bytes that overlapped an in-flight async chunk (duplicate reads). Keep prompt (`n_tokens>1`) and decode (`n_tokens==1`) separate. Measure the classification's own cost (`probe_ns`). At context teardown, next to the existing stats lines, print exactly one line:
`expert_readiness: stats decode_tokens=%llu decode_selected=%llu decode_selected_predicted=%llu decode_ready_bytes_predicted=%llu decode_missing_bytes_predicted=%llu decode_ready_bytes_unpredicted=%llu decode_missing_bytes_unpredicted=%llu decode_wasted_async_bytes=%llu decode_duplicate_demand_bytes=%llu prompt_selected=%llu prompt_missing_bytes=%llu probe_ns_total=%llu`
and at init (when enabled) `expert_readiness: active`. Add `readiness=` to the `expert_config: effective` line and register the env name in the strict validator (unknown `LLAMA_EXPERT_*` names abort today — the new name must validate as `0|1`). Disabled ⇒ no extra work on the hot path beyond a branch.

## Patch 11 — `inflight_join` (performance, switch `LLAMA_EXPERT_PREGATE_JOIN=0|1`, default 0)
When enabled, demand `fill()` must not re-read pages covered by an async chunk that is **in flight**: wait for that chunk's completion (condition variable / per-chunk state; no spinning) and then treat it as read. A **queued, not started** chunk covering needed pages is claimed (removed from the queue) and read by demand as today. Errors in the async read ⇒ demand re-reads (fallback, never skip). Never write weights; never change which experts compute uses. Add `join=` to the effective-config line, register/validate the env, and add counters to the pregate stats line: `join_waits= join_wait_ns= join_bytes=` (appended at end of the existing `expert_pregate: stats` line so existing regexes keep matching). Patch 11 should apply on top of 10 (declare that dependency); if it can be made independent cleanly, say so.

## Deliverables
1. Two commits on `xs-dev` (10 then 11), messages `expert: readiness observability` / `expert: join in-flight async reads`.
2. `git format-patch` outputs saved as `/home/muk/projects/xs-dev/10-expert-readiness.patch` and `/home/muk/projects/xs-dev/11-inflight-join.patch`, each applying with `git apply --check` on top of the previous state (verify with a throwaway `git worktree` from `6e9d96cf4`, then remove the worktree).
3. CPU-only compile + functional check (no CUDA): `cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu -j4 --target llama-completion`. Then with `/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf` (small MoE) run `build-cpu/bin/llama-completion -m <gguf> -p "The capital of France is" -n 32 -c 512 --temp 0 -s 1 -no-cnv -cmoe -lm mmap -t 4` in these env configs and record stdout sha256 + the stats lines into `/home/muk/projects/xs-dev/F13-check.md`:
   a. all LLAMA_EXPERT_* unset; b. `FILL=1 FILL_RANDOM=decode FILL_MAX_TOKENS=1 PREGATE=1 PREGATE_K=8`; c. b + `READINESS=1`; d. c + `PREGATE_JOIN=1`; e. `READINESS=2` must abort with the validator's message. Stdout of a–d must be byte-identical. Note: the page cache may already hold the small model, so many counters may be near zero — that's fine; say so. If -cmoe/the callback is inactive on a CPU-only build, explain and still verify compile + identical output + that the lines print.
4. In `F13-check.md` also: the exact semantics of each new counter (one line each), hot-path cost when disabled, and known limitations.

## Acceptance (run verbatim by a checker)
- A1 `git -C /home/muk/projects/xs-dev/llama.cpp log --oneline -3` shows the two commits on top of `6e9d96cf4`.
- A2 both patch files exist and `F13-check.md` shows identical sha for a–d and the abort for e.
- A3 `grep -c 'expert_readiness: stats' F13-check.md` ≥ 1 and the pregate line contains `join_waits=`.
- A4 Nothing outside `/home/muk/projects/xs-dev` modified.
Final message: summary, counter semantics, risks.
