# Assignment H5 — fix measurement gaps in run summaries and gate (worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Read c2/MEASUREMENT-GAPS.md §1-2 first. Files you may edit: cc/abrun/*, cc/resource_observe/*, their tests/README, c2/METHODS.md (append only).
## Deliverables
1. run_summary.py:
   a. `eval_ms_per_token` and `prompt_ms_per_token` parsed from `common_perf_print` lines (`eval time = X ms / N runs ( Y ms per token`), plus `eval_ms_total`, `eval_runs`, `load_ms`. Keep eval_tps for display.
   b. Phase windows: derive decode window start/end on the telemetry clock. stderr lines are prefixed with a relative timestamp (e.g. `2.20.928.587 I ...`); if the absolute mapping is not derivable, use end-of-run anchored on the last telemetry sample before container exit and eval_ms_total (document the method + error bound). Report `nvme_gib_decode`, `nvme_gib_per_token_decode` (bytes read only inside the decode window / eval_runs), and `nvme_gib_load`.
   c. Summaries over the decode window: gpu_util median/p90 (tegrastats `gpu_util`), `device_util` median (io backend), `power_in_w_median`, `energy_j_per_token` (∫ VDD_IN instant mW over decode window / eval_runs), temp_tj max.
   d. gpu_freq_median: resource_observe must sample `/sys/class/devfreq/17000000.gpu/cur_freq` (Hz; discover the gpu devfreq node by name `*.gpu`, not hard-coded) as metric `gpu_freq` per sample. Record emc freq too if readable without root.
2. gate.py: primary metric may be declared `lower_is_better` (spec field `"metric": "eval_ms_per_token", "direction": "lower"`). Rule unchanged in ratio form (use A/B for lower-is-better). Default for new specs: eval_ms_per_token. regate.py must work on old run dirs (recompute from stderr).
3. abrun secondary table adds: ms/token, NVMe GiB/token (decode window), gpu_util med, J/token.
4. Tests: fixtures must be CAPTURED from real run dirs (copy trimmed real stderr.txt + telemetry.jsonl from experiments/jetson/OM/J-OM10c/003-pg8i8 into tests/fixtures; never hand-written numbers). Assert eval_ms_per_token == 716.54 for that fixture.
5. Validation: `python3 cc/abrun/regate.py experiments/jetson/OM/J-OM10c --dry` (or equivalent no-write mode) prints the J-OM10c table with ms/token; include it in the final message.
## Acceptance (checker runs)
- A1 all cc/abrun + cc/resource_observe tests pass (count reported).
- A2 regate on J-OM10c (no writes into experiments/) shows ms/token per arm: base ≈ 840, pg arms ≈ 712-732.
- A3 decode-window NVMe GiB/token < whole-run GiB/token for J-OM10c base (load excluded), with the method documented.
- A4 resource_observe emits gpu_freq in a 3 s dry sample (`python3 cc/resource_observe/resource_observe.py --help` to find the sampling mode; must run without docker/GPU job).
- A5 no edits under experiments/ or evidence/.
Constraints: no docker/GPU runs; no commits. Time 60 min. Final <= 12 lines.
