# Memory, quality and speed report

Run from any working directory (Python standard library only):

```sh
python /home/muk/projects/siminfeng/cc/memory_pareto/pareto.py
python cc/memory_pareto/pareto.py experiments/memory/configs.csv --budget-mib 700
```

The script prints a Markdown table sorted by ascending GPU MiB (unknown memory last), followed by a terminal ASCII scatter of GPU MiB versus KLD. It writes `pareto.md` and `pareto.json` beside the script by default. `--output-dir artifacts/codexI` selects another assignment-permitted destination. Reports are overwritten on reruns. Nothing is written to the input paths. No network, Docker, or third-party runtime packages are used.

## Inputs

The primary input is `experiments/memory/configs.csv`, resolved relative to the campaign root, with these required headers:

```csv
config,desc,class,gpu_mib,host_mib,tg_ts,pp_ts,kld,top1_pct,ppl_ratio,notes
```

`class` must be `transferable`, `discrete-only`, or `mixed`. Supply the orchestrator's classifications; the script does not validate scientific transferability. Memory is MiB; TG and PP are tokens per second. KLD is smaller-is-better. Top1 is a percentage; PPL ratio is supplied relative to the experiment's stated reference. Neither is converted into KLD. Empty cells, NA, N/A, null, none, NaN and nonfinite numbers become unavailable, never zero. Invalid numbers, negative values, invalid classes, and percentages above 100 are errors. Zero is valid. Notes should identify workloads, hardware, model, measurement method and quality reference. Rows are individual observations; repeated names or exact metric ties are preserved.

If the default CSV is absent, the script reads `lab/jobs/*/att-*/result.json`, the corresponding `job.json` (`spec.model`), and `lab/tests/*.json` (test-specific `model_file`, args and env). `--lab-root` overrides this fallback root. An explicitly supplied CSV path that is absent is an error; an invalid or empty existing CSV never triggers fallback.

Fallback accepts completed attempts and successful, nonpartial tests only. It emits a separate row per benchmark sample record or quality test; it never joins observations across tests, attempts, models or workloads. GPU MiB means `mem.gpu_peak_mib`, **not** baseline-subtracted delta; host MiB means `host_cgroup_peak_mib`. TG requires zero prompt tokens and positive generated tokens; PP requires zero generated tokens and positive prompt tokens. Combined PP+TG samples supply neither speed objective. `avg_ts` supplies speed; repeated `samples_ts` are not independently analyzed. Raw PPL and its stderr are recorded in notes, but no PPL ratio or KLD is invented. Fallback classification is explicitly `unclassified`, so these rows cannot support transferable claims. Job/test/sample identifiers and source result paths are retained. Attempts are not averaged or deduplicated.

## Frontier and budget

A row is Pareto-optimal if no other eligible row has GPU MiB no higher, TG tokens/s no lower, and KLD no higher, with at least one strict improvement. All three objectives must be available. Exact ties remain optimal together. Missing Top1 or PPL ratio is flagged but does not disqualify a row with KLD. Missing KLD, GPU memory or TG speed excludes it from every frontier. Missing-quality and missing-objective lists are explicit in both output formats.

The table and JSON mark the global frontier, the frontier among budget-fitting rows, and the frontier within each labelled class. The global comparison is a descriptive pooled view; it does not combine discrete-only measurements into transferable claims. Always consult the class column and class frontier. Comparing workloads or quality references that differ may be meaningless even within one class.

A row fits when its known GPU MiB is **at most** `--budget-mib` (inclusive); missing GPU memory has unknown fit. Fit does not require quality data or frontier membership. The budget defaults to 700 MiB, a rounded scaled analogue of a 500 GB to 128 GB target on a 2.67 GB model (2.67 × 128 / 500 ≈ 0.684 GB). This is a comparison line, not a prediction of unified-memory or production capacity; GB/MiB conventions and runtime overhead matter. Host memory is shown separately and is not added to the GPU budget. Since GPU memory is also an objective, a fitting row's dominator must also fit; the budget frontier is therefore the global frontier restricted to fitting rows.

The ASCII scatter is only a two-dimensional projection; its frontier symbols come from all three objectives, including TG speed. `+` indicates plotting-cell collisions. Point coordinates reference table row numbers. Unknown GPU/KLD points are omitted and an empty plot is explained.

## Limits and validation

This tool establishes **no statistical significance**, uncertainty bounds, speed stability, quality equivalence, or causal improvement. It uses exact input values without tolerance, confidence intervals or significance tests. A frontier is relative only to supplied observations and these objectives, not proof of general optimality. Host memory, PP, Top1 and PPL ratio are contextual columns, not optimization axes. Discrete-only rows remain labelled; neither they nor mixed/unclassified rows establish transferable savings. Fair claims require matched workloads, hardware, model/quality references and memory accounting, reviewed outside this script.

Run synthetic checks without writing outside the assignment locations:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=artifacts/codexI/pytest-tmp cc/memory_pareto/test_pareto.py
```

Tests cover domination, speed/quality trade-offs, inclusive and unknown budget fit, missing quality/objectives, ties, class frontiers, sorting and report output, zero values, invalid inputs, CSV precedence, and conservative lab fallback.
