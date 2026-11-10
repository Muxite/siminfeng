from pathlib import Path
p=Path('artifacts/codexR/tree/src/llama-model.h');s=p.read_text().replace('    struct ggml_tensor * ffn_expert_ids_cold = nullptr;','    struct ggml_tensor * ffn_expert_ids_cold = nullptr;\n    struct ggml_tensor * ffn_expert_ids_router = nullptr;');p.write_text(s)
p=Path('artifacts/codexR/tree/src/models/granite-moe.cpp');s=p.read_text();s=s.replace('''                if (layer.ffn_expert_ids_hot->type != GGML_TYPE_I32 || layer.ffn_expert_ids_cold->type != GGML_TYPE_I32) {''','''                layer.ffn_expert_ids_router = create_tensor(tn(LLM_TENSOR_FFN_EXPERT_IDS, "ids_router", i), {n_expert}, 0);
                if (layer.ffn_expert_ids_hot->type != GGML_TYPE_I32 || layer.ffn_expert_ids_cold->type != GGML_TYPE_I32 || layer.ffn_expert_ids_router->type != GGML_TYPE_I32) {''');p.write_text(s)
p=Path('artifacts/codexR/tree/src/llama-graph.cpp');s=p.read_text();start=s.index('    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4;',s.index('         ggml_tensor * gate_inp_b,\n', s.index('ggml_tensor * llm_graph_context::build_moe_ffn(')))
pos=s.index('\n    if (gate_inp_b)',start)
s=s[:pos]+'''
    if (split_layer && split_layer->ffn_up_exps_cold) {
        // Preserve the original reduction order and top-k tie break after permutation.
        auto * order = ggml_repeat_4d(ctx0, split_layer->ffn_expert_ids_router, n_expert, n_tokens, 1, 1);
        logits = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, logits, 1, n_expert, n_tokens), order), n_expert, n_tokens);
        cb(logits, "ffn_moe_logits_original_order", il);
    }
''' +s[pos:]
old='''        std::vector<int32_t> map_hot(n_expert), map_cold(n_expert);
        ggml_backend_tensor_get(split_layer->ffn_expert_ids_hot, map_hot.data(), 0, n_expert * sizeof(int32_t));
        ggml_backend_tensor_get(split_layer->ffn_expert_ids_cold, map_cold.data(), 0, n_expert * sizeof(int32_t));
        for (int64_t e = 0; e < n_expert; ++e) {
            if (map_hot[e] != (e < n_hot ? e : n_hot) || map_cold[e] != (e < n_hot ? -1 : e - n_hot)) {
                throw std::runtime_error("invalid Granite split ID map");
            }
        }'''
new='''        std::vector<int32_t> map_hot(n_expert), map_cold(n_expert), map_router(n_expert);
        std::vector<bool> seen(n_expert, false);
        ggml_backend_tensor_get(split_layer->ffn_expert_ids_hot, map_hot.data(), 0, n_expert * sizeof(int32_t));
        ggml_backend_tensor_get(split_layer->ffn_expert_ids_cold, map_cold.data(), 0, n_expert * sizeof(int32_t));
        ggml_backend_tensor_get(split_layer->ffn_expert_ids_router, map_router.data(), 0, n_expert * sizeof(int32_t));
        for (int64_t e = 0; e < n_expert; ++e) {
            const int32_t p = map_router[e];
            if (p < 0 || p >= n_expert || seen[p] || map_hot[e] != (p < n_hot ? p : n_hot) || map_cold[e] != (p < n_hot ? -1 : p - n_hot)) {
                throw std::runtime_error("invalid Granite split ID map");
            }
            seen[p] = true;
        }'''
assert old in s;s=s.replace(old,new)
s=s.replace('''        if (moe_cache || weight_before_ffn''','''        if (moe_cache || (loras && !loras->empty()) || hdmd || weight_before_ffn''')
p.write_text(s)
p=Path('cc/hybrid_split/split_gguf.py');s=p.read_text();s=s.replace("writer.add_tensor_info(name, (experts, 1), np.dtype('int32'), experts * 4, gguf.GGMLQuantizationType.I32)","writer.add_tensor_info(name, (experts,) if name.endswith('ids_router') else (experts, 1), np.dtype('int32'), experts * 4, gguf.GGMLQuantizationType.I32)")
s=s.replace("        add(f'blk.{il}.ffn_expert_ids.ids_cold', None, 'map', il)","        add(f'blk.{il}.ffn_expert_ids.ids_cold', None, 'map', il)\n        add(f'blk.{il}.ffn_expert_ids.ids_router', None, 'map', il)")
old="""            values = ([i if i < nh else nh for i in range(experts)] if name.endswith('ids_hot')
                      else [-1 if i < nh else i - nh for i in range(experts)])
            data = np.asarray(values, dtype=np.int32).reshape(experts, 1)"""
new="""            inverse = np.argsort(perms[il]).tolist()
            values = ([i if i < nh else nh for i in inverse] if name.endswith('ids_hot')
                      else [-1 if i < nh else i - nh for i in inverse] if name.endswith('ids_cold') else inverse)
            data = np.asarray(values, dtype=np.int32)
            if not name.endswith('ids_router'):
                data = data.reshape(experts, 1)"""
assert old in s;s=s.replace(old,new);p.write_text(s)
p=Path('cc/hybrid_split/test_split_gguf.py');s=p.read_text();s=s.replace("    ic = ts['blk.0.ffn_expert_ids.ids_cold'].data.ravel()", "    ic = ts['blk.0.ffn_expert_ids.ids_cold'].data.ravel()\n    inverse = ts['blk.0.ffn_expert_ids.ids_router'].data\n    np.testing.assert_array_equal(ts['blk.0.ffn_gate_inp.weight'].data[inverse], router)")
s=s.replace('h.data[ih[idx]] if ic[idx] == -1 else c.data[ic[idx]]','h.data[ih[old]] if ic[old] == -1 else c.data[ic[old]]');p.write_text(s)
