from pathlib import Path
p=Path('artifacts/codexR/tree/src/llama-graph.cpp')
s=p.read_text().replace('''    if (split_layer && split_layer->ffn_up_exps_cold) {
        if (moe_cache''','''    if (split_layer && split_layer->ffn_up_exps_cold) {
        if (cparams.op_offload) {
            throw std::runtime_error("split Granite requires --no-op-offload to keep cold operations on CPU");
        }
        if (moe_cache''')
# Verify tiny mapping arrays when building the graph, after model buffers are loaded.
s=s.replace('''        // Flatten first: GET_ROWS table batches must match the ID batches.''','''        const int64_t n_hot = up_exps->ne[2] - 1;
        std::vector<int32_t> map_hot(n_expert), map_cold(n_expert);
        ggml_backend_tensor_get(split_layer->ffn_expert_ids_hot, map_hot.data(), 0, n_expert * sizeof(int32_t));
        ggml_backend_tensor_get(split_layer->ffn_expert_ids_cold, map_cold.data(), 0, n_expert * sizeof(int32_t));
        for (int64_t e = 0; e < n_expert; ++e) {
            if (map_hot[e] != (e < n_hot ? e : n_hot) || map_cold[e] != (e < n_hot ? -1 : e - n_hot)) {
                throw std::runtime_error("invalid Granite split ID map");
            }
        }

        // Flatten first: GET_ROWS table batches must match the ID batches.''')
p.write_text(s)
p=Path('cc/hybrid_split/compare_logits.cpp');s=p.read_text().replace('        cp.n_ctx = 512;', '        cp.op_offload = false;\n        cp.n_ctx = 512;');p.write_text(s)
