Captured verbatim from `experiments/jetson/OM/J-OM10c/003-pg8i8` on 2026-11-11.

`stderr.txt` retains the first and last lines and all common_perf_print lines.
`telemetry.jsonl` retains every io and tegrastats record plus the first/last records
(to preserve the run clock endpoints). No retained line or numeric value was changed.
`result.json` and `stdout.txt` are exact copies. Tests verify provenance against the source.
