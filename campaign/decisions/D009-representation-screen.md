# D009 — Representation screen (REPRESENTATION CHANGE, not an engine change)
Jobs: job-23149e82ac (bf16), job-48de707fc0 (Q8_0), job-80598c0569 (Q4_K_M); 3 interleaved attempts each; same image, same data; raw samples in lab/jobs/*/att-*/*.stdout.
| repr | PPL(20ch) | tg128 t/s | pp512 t/s |
| bf16 | 10.015 +/-0.370 | 599.1 | 17710 |
| Q8_0 | 10.073 (+0.58%) | 793.9 (+32.5%) | 44638 (+152%) |
| Q4_K_M | 10.610 (+5.9%) | 710.3 (+18.6%) | 41680 (+135%) |
Spread across attempts <=1.1% on all arms (A/A noise bound ~2%). Prefill first-sample ratio 0.6 for quantized => warmup matters more there; means include it (all arms equally).
Interpretation: decode dominated by BF16 weight streaming + BF16 matvec; Q8_0 halves bytes and uses MMQ integer kernels. Q4_K_M slower than Q8_0 in decode: K-quant dequant cost outweighs bytes saved on this GPU/kernels.
Limits: PPL is a smoke metric on 20 chunks (stderr ~0.37 unpaired); paired token-level check (KL vs BF16) still required before calling Q8_0 quality-preserving. This is a weights-representation change; the engine and routing are unchanged.
