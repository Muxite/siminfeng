# Assignment T-C (parent: campaign c1 / orchestrator, task stage1; see decisions/D003)
Goal context: we are reverse-engineering the *methodology* behind Strata (MoE expert slots/residency/CPU-GPU dispatch) to decide what tools and components WE must build for something similar. Other engines are inspiration inputs.
Question: how does llama.cpp implement MoE (Granite MoE specifically) and expert placement, and what does it have / lack compared with Strata's expert-slot + residency + hit-hook model?
Input: /home/muk/projects/siminfeng/campaign/ref/llama.cpp @ 31288c9c6 (READ ONLY). Strata for comparison: /home/muk/projects/Strata @ ffa80d73133ea5c548aefe57889a16e8085a938f (READ ONLY).
Cover: Granite MoE graph build (src/models or llama-model.cpp: granitemoe), router/top-k (ggml_top_k / soft_max / weights norm), expert matmul (ggml_mul_mat_id and CUDA kernels in ggml/src/ggml-cuda), expert tensor layout in GGUF, convert_hf_to_gguf.py GraniteMoe handling, CPU/GPU placement flags (--n-cpu-moe, --cpu-moe, -ot, --override-tensor, -ngl), scheduler (ggml-backend sched: splits, copies, sync), KV cache/session reset, memory fit (llama-fit-params), built-in instrumentation (llama-bench, llama-perplexity, eval callback / cb_eval, GGML_SCHED_DEBUG, tensor-level tracing hooks usable to log expert ids).
Deliverables under /home/muk/projects/siminfeng/campaign/artifacts/codexC/:
 1. llamacpp-moe-map.md — file:line cited map.
 2. compare-vs-strata.md — table: mechanism | llama.cpp | Strata | is the difference a missing component we'd have to build? | tool/test needed.
 3. instrumentation-options.md — cheapest ways to log router top-k expert IDs per layer/token from llama.cpp WITHOUT modifying the image (cb_eval via a small out-of-tree harness?) vs with a patched build; cost and what each can/can't prove.
Write only under campaign/artifacts/codexC/. No network. Time limit: 40 min. Final message: files written + open questions.
