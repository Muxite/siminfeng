# D036 — Correct q4_K screen closes the missing type, not the graph-level prediction gap (2026-11-11, jetson-arm)
The first KS-q4K attempt (`experiments/jetson/KS-q4K-20261111/`, r1) was invalid: kernel_screen lower-cased `q4_K` to `q4_k` in the parameter regex. ggml names are case-sensitive, so the intended cases were not selected. The fixture had been fabricated with the same incorrect spelling and passed tests. The existing `cc/kernel_screen/kernel_screen.py` git diff replaces `.lower()` with a canonical ggml type map; this documentation task does not edit that fix.

The second attempt (`experiments/jetson/KS-q4K-20261111-r2/`) was invalid because DOCKER_CMD_PREFIX was not exported to the jobq submission environment. jobq replays submit-time environment, so the job received docker permission denied. The third attempt supplied the environment explicitly and succeeded. Sources: `c2/worklog.jsonl`, KS-q4K tool_bug failure, KS-q4K-r2 operator_env failure, KS-q4K-r3 result; current `cc/kernel_screen/kernel_screen.py` git diff.

| r3 case (q4_K) | A median us | A2 median us | B median us | A/A2 | A/B |
|---|---:|---:|---:|---:|---:|
| MUL_MAT, down, decode | 15.89 | 15.74 | 15.89 | 1.0095 | 1.0000 |
| MUL_MAT_ID, down, decode, b=0 | 89.59 | 88.41 | 89.20 | 1.0133 | 1.0044 |
| MUL_MAT_ID, up, decode, b=1 | 65.06 | 55.07 | 66.97 | 1.1814 | 0.9715 |
| MUL_MAT_VEC_FUSION, down, decode, b=0 | 3.72 | 3.74 | 3.84 | 0.9947 | 0.9688 |

Source: `experiments/jetson/KS-q4K-20261111-r3/q4_K/results.md` (decode means n_tokens=1; direction labels match its case metadata). Ratios are A/arm time; higher is faster. A2 is the interleaved duplicate control.

Interpretation:
- The completed r3 screen has 108 runs, A/B 0.91–1.06 against A/A 0.94–1.18 (`c2/worklog.jsonl`, KS-q4K-r3 result). It does not reproduce D027's graph-level mixE4K gain of +23.6% (same worklog event). The new type agrees with the earlier bf16/q8_0 conclusion: isolated ops cannot predict the changed gate/up fusion graph. Use this harness for kernel-local screening; add a graph-level fused chain or a small engine graph before using it to screen graph rewrites.
- CLOSEOUT's original KS result row attributes the type failure only to pipeline upper-casing. That attribution is affected: the tool's own lower-casing was independently broken, and merely passing `--types q4_K` was insufficient before the fix. Adjacent “(see Addendum c2)” markers identify the original affected claims without rewriting historical text.
- The original open item “KS q4_K arm not rerun” is now closed by r3. The broader graph-level screen remains open. Capture fixtures from actual tool output; a fabricated fixture that shares the implementation's normalization error cannot validate external compatibility.
