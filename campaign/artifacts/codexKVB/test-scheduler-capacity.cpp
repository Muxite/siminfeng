#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>

int main() {
    constexpr size_t capacity = 600;
    ggml_init_params params = { 2*capacity*ggml_tensor_overhead() + ggml_graph_overhead_custom(capacity, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, capacity, false);
    ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    for (int i = 0; i < 550; ++i) {
        result = ggml_add(ctx, result, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1));
    }
    ggml_build_forward_expand(graph, result);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, capacity, false, false);
    std::fprintf(stderr, "capacity=%zu nodes=%d leaves=%d\n", capacity, ggml_graph_n_nodes(graph), graph->n_leafs);
    // Use the same split-only path as the flash-attention capability probe.
    ggml_backend_sched_split_graph(sched, graph);
    bool ok = ggml_backend_sched_reserve(sched, graph);
    std::fprintf(stderr, "reserve=%s\n", ok ? "ok" : "failed");
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok ? 0 : 1;
}
