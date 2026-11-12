# F17 — page-cache state: measure it per run, control it by protocol

Repo: `/home/muk/projects/siminfeng` (branch `jetson/c1`). Touch only `cc/om_run/`, `cc/abrun/`, `cc/page_cache/` (new) and `c2/METHODS.md`. Do not touch `evidence/`, `experiments/`, `lab/jobq/`, or anything else. No Docker/GPU runs, no commits.

## Why
Expert streaming performance depends on which of the model file's 20.8 GiB is in the page cache when a run starts (7.4 GiB RAM). Today the lab neither records nor controls that: each run inherits whatever the previous run (other arm) left behind, a carryover effect that ABBA ordering only partly balances. No sudo is available (no drop_caches).

## Build
1. `cc/page_cache/page_cache.py` (stdlib + ctypes, Python 3.10, no root):
   - `resident(path) -> {file_bytes, resident_bytes, resident_pct, pages, elapsed_ms}` via mmap(PROT_READ) + `mincore` over the whole file (ctypes; chunk the vector, e.g. 1 GiB windows, to bound memory). Optionally `--ranges tensors.json` to report residency for given byte ranges (e.g. expert tensors) — accept a JSON list of `{name, offset, size}`.
   - `evict(path)` via `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` (works unprivileged for clean pages of a file you can read; pages mapped by another live process may stay — report resident before/after).
   - `warm(path, ranges|prefix_bytes)` sequential read of a fixed set for a reproducible warm start.
   - CLI: `python3 cc/page_cache/page_cache.py {resident,evict,warm} <file> [--json] [...]`.
   - Tests `cc/page_cache/test_page_cache.py` on a temp file (resident after read > 0; after evict decreases — tolerate filesystems where DONTNEED is a no-op by skipping with reason).
2. `cc/om_run/om_run.sh`: before launching the engine container, if `OM_CACHE_PROTOCOL` is set: `carryover` (default; do nothing but record), `cold` (evict model file), `warm` (evict then warm `OM_CACHE_WARM_BYTES` prefix, or ranges file). ALWAYS record `page_cache.start.json` (resident at start, after protocol action) and `page_cache.end.json` in the run dir, cheap. Keep the existing behaviour byte-for-byte otherwise (read the script carefully; it's used by running experiments).
3. `cc/abrun`: spec key `cache_protocol` (carryover|cold|warm, default carryover) passed to om_run as env; run summaries (`run_summary.py`) gain `cache_start_resident_pct` and `cache_end_resident_pct`; REPORT.md gets a column for them; gate.py unchanged except: add to the verdict a `cache_confound` note when the arms' mean start residency differs by > 10 percentage points (do NOT change the gate decision; just flag).
4. `c2/METHODS.md`: a short "Page-cache protocol" section: what each protocol measures (carryover = steady-state streaming as in production; cold = worst case after eviction; warm = controlled), when to use which, and that verdicts must state the protocol.
## Acceptance (run verbatim)
- P1 `python3 -m pytest cc/page_cache cc/abrun -q -p no:cacheprovider` passes (existing abrun tests still pass).
- P2 `python3 cc/page_cache/page_cache.py resident /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf --json` prints valid JSON.
- P3 `bash -n cc/om_run/om_run.sh` OK and `git diff cc/om_run/om_run.sh` shows only additive changes guarded by env/recording.
Final message: changes, how to use in a spec, caveats (e.g. eviction while another process maps the file).
