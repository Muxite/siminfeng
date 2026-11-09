# D035 — Placement-aware precision (PAP) + imatrix: results (protocol D032-A), job job-3d2fbbd061 (48/48 clean)
Conditions: llamacpp-b0, chat -n 160 -c 4096, 3 alternating rounds; host monitor: no compiler, BUT the 18.6 GB proxy-2 download ran throughout (dl=3) and round 3 was slower for several configs -> decode spread 10-15% within a config (vs 1-3% in D022/D029). Speed conclusions are therefore PROVISIONAL.
QUALITY (deterministic, 20x512 wikitext-test vs bf16 logits; solid):
  mixE4K KLD 0.040 top1 90.5 | mix4_imat (same types/bytes, imatrix from wikitext TRAIN) 0.027 / 91.9 (-33% KLD, free) |
  pap12_q3 0.073 / 87.3 | pap18_q3 0.086 / 85.8 | pap24_q3 0.117 / 83.6 vs pap24_q3_noimat 0.135 / 81.8 (imatrix -13%) |
  pap12_q2 0.193 | pap18_q2 0.258 | pap24_q2 0.359 (2-bit experts unacceptable even with imatrix and only in CPU layers).
SPEED (median decode t/s, same K vs mixE4K): pap12_q3 214.9 vs 195.7 (+9.8%), pap18_q3 +5.5%, pap24_q3 +8.7%, q2 variants +14..19%; mix4_imat vs mixE4K -3..+4% (identical bytes => file-level noise floor in this job ~±5%).
GPU MiB unchanged (PAP only touches CPU-resident layers); host cgroup peak unchanged (weights are mmap page cache, not charged) — PAP's memory effect is on file/page-cache size: pap12_q3 738 MB vs mixE4K 816 MB (-9.6%).
Verdict vs frozen rule (>= +5% AND KLD <= 0.08): pap12_q3 PROVISIONALLY ACCEPTED (speed needs a quiet-host 5-round rerun); all others rejected (quality). Imatrix: ACCEPTED as a factory default — same bytes, -33% KLD at 4-bit, -13% at 3-bit (calibration on wikitext-train; generalization beyond wikitext not tested).
Factory rules: (1) always quantize experts with an imatrix from a held-out calibration set; (2) lower-bit only for CPU-resident layers and only to 3-bit; (3) speed gain from PAP is bounded by the CPU share of decode time — it matters most where the CPU part dominates (big offloaded models).
