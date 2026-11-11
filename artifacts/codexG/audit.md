# T-G preparation audit

Input HEAD verified read-only: `ffa80d73133ea5c548aefe57889a16e8085a938f` in `/home/muk/projects/Strata`.

Deliverables:

- `builds/strata-tests/select.md`: eight candidates, five selected, file:line evidence, CPU coverage boundaries, dependency/fetch analysis and exact per-object expert recipe.
- `builds/strata-tests/run_cpu_tests.sh`: executable container runner; three CPU CMake targets, two standalone unchanged-source builds; partial results and toolchain versions in `/build/results.json`.
- `builds/strata-tests/README.md`: exact no-network/no-GPU Docker command, read-only mounts, campaign label, 8 CPU / 16 GiB limits, estimates and blockers.

Read the assigned prior inventory/map and Strata's root `AGENTS.md`. No applicable campaign `AGENTS.md` was present along the working-directory ancestry. No delegation was used. Wrote only inside `builds/strata-tests/` and `artifacts/codexG/`.

Validation actually performed:

- `bash -n builds/strata-tests/run_cpu_tests.sh`: passed after the final script edit.
- Extracted the embedded Python here-document and called `ast.parse`: passed; no runner statements were executed.
- Statically checked five selected entries (three CMake, two standalone), existence of all ten directly named test/support source files, and output-file trailing whitespace: passed.
- Static recursive quoted-include scan for the original test/base-library sources resolved 32 files without a missing quoted include or CUDA/ggml angle include. The added `q2_avx1.cpp` source includes the already resolved expert/layout headers and standard/intrinsic headers. This is dependency inspection, not a preprocessor, compiler or linker run.
- Inspected CMake gates and existing source declarations. Found an expected native-OFF link failure: `expert_layout.cpp:402-419` references AVX1 functions whose `q2_avx1.cpp` definitions are added only under native-ON. Adjusted the expert test to a standalone six-object build, preserving per-file ISA flags and using repository-provided native-disabled definitions. This finding and remedy are unconfirmed by compilation.

Not performed: Docker, CMake configure, compilation, linking, test execution, model access, GPU access, network access. No runtime `results.json` is fabricated here. The orchestrator must run the documented command to obtain actual verdicts. No source checkout files were modified.
