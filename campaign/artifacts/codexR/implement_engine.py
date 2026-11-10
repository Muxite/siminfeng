from pathlib import Path
root = Path('artifacts/codexR/tree')
def edit(name, old, new):
 p = root/name
 s=p.read_text()
 assert s.count(old)==1,(name,s.count(old),old[:100])
 p.write_text(s.replace(old,new))
edit('src/llama-arch.h', '    LLM_TENSOR_FFN_GATE_INP,\n', '    LLM_TENSOR_FFN_GATE_INP,\n    LLM_TENSOR_FFN_EXPERT_IDS,\n')
edit('src/llama-arch.cpp', '    { LLM_TENSOR_FFN_GATE_INP,                           "blk.%d.ffn_gate_inp" },', '    { LLM_TENSOR_FFN_GATE_INP,                           "blk.%d.ffn_gate_inp" },\n    { LLM_TENSOR_FFN_EXPERT_IDS,                         "blk.%d.ffn_expert_ids" },')
edit('src/llama-arch.cpp', '    {LLM_TENSOR_FFN_GATE_INP,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},', '    {LLM_TENSOR_FFN_GATE_INP,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},\n    {LLM_TENSOR_FFN_EXPERT_IDS,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_GET_ROWS}},')
edit('src/llama-model.h', '    // ff MoE per-expert scales (NVFP4 per-tensor scale2)', '''    // Optional Granite hot/cold banks; the normal expert fields hold the hot banks.
    struct ggml_tensor * ffn_gate_exps_cold = nullptr;
    struct ggml_tensor * ffn_up_exps_cold   = nullptr;
    struct ggml_tensor * ffn_down_exps_cold = nullptr;
    struct ggml_tensor * ffn_expert_ids_hot = nullptr;
    struct ggml_tensor * ffn_expert_ids_cold = nullptr;

    // ff MoE per-expert scales (NVFP4 per-tensor scale2)''')
edit('src/models/granite-moe.cpp', 'void llama_model_granite_moe::load_arch_tensors(llama_model_loader &) {', 'void llama_model_granite_moe::load_arch_tensors(llama_model_loader & ml) {')
old='''            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);'''
new='''            const auto hot_name = tn(LLM_TENSOR_FFN_UP_EXPS, "weight.hot", i);
            const auto * hot_meta = ml.get_tensor_meta(hot_name.str().c_str());
            if (hot_meta) {
                const int64_t n_hot = hot_meta->ne[2] - 1;
                if (n_hot < 0 || n_hot >= n_expert) {
                    throw std::runtime_error("invalid Granite split expert count");
                }
                for (auto tensor : {LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_UP_EXPS, LLM_TENSOR_FFN_DOWN_EXPS}) {
                    if (ml.get_tensor_meta(tn(tensor, "weight", i).str().c_str())) {
                        throw std::runtime_error("mixed split and unsplit Granite expert banks");
                    }
                }
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight.hot", i), {n_embd, n_ff, n_hot + 1}, 0);
                layer.ffn_up_exps   = create_tensor(hot_name, {n_embd, n_ff, n_hot + 1}, 0);
                layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight.hot", i), {n_ff, n_embd, n_hot + 1}, 0);
                layer.ffn_gate_exps_cold = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight.cold", i), {n_embd, n_ff, n_expert - n_hot}, 0);
                layer.ffn_up_exps_cold   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight.cold", i), {n_embd, n_ff, n_expert - n_hot}, 0);
                layer.ffn_down_exps_cold = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight.cold", i), {n_ff, n_embd, n_expert - n_hot}, 0);
                layer.ffn_expert_ids_hot  = create_tensor(tn(LLM_TENSOR_FFN_EXPERT_IDS, "ids_hot", i), {1, n_expert}, 0);
                layer.ffn_expert_ids_cold = create_tensor(tn(LLM_TENSOR_FFN_EXPERT_IDS, "ids_cold", i), {1, n_expert}, 0);
                if (layer.ffn_expert_ids_hot->type != GGML_TYPE_I32 || layer.ffn_expert_ids_cold->type != GGML_TYPE_I32) {
                    throw std::runtime_error("Granite split ID maps must be I32");
                }
            } else {
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff, n_expert}, TENSOR_NOT_REQUIRED);
                layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff, n_embd, n_expert}, 0);
                layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_embd, n_ff, n_expert}, 0);
            }'''
edit('src/models/granite-moe.cpp',old,new)
# Keep the ordinary FFN graph untouched. Split models return before cache binding.
anchor='''    // the experts of host-resident layers may be read from the MoE cache
    ggml_tensor * slots = build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, gate_up_exps, il);'''
new='''    const auto * split_layer = il >= 0 ? &model.layers[il] : nullptr;
    if (split_layer && split_layer->ffn_up_exps_cold) {
        if (moe_cache || weight_before_ffn || type_op != LLM_FFN_SILU || !gate_exps || gate_up_exps ||
                up_exps_b || gate_exps_b || down_exps_b || up_exps_s || gate_exps_s || down_exps_s || post_exp_norm) {
            throw std::runtime_error("unsupported split Granite FFN configuration");
        }
        for (auto * bank : {split_layer->ffn_up_exps_cold, split_layer->ffn_gate_exps_cold, split_layer->ffn_down_exps_cold}) {
            if (!ggml_backend_buffer_is_host(bank->buffer)) {
                throw std::runtime_error("split cold banks require -ot '\\\\.cold$=CPU'");
            }
        }

        // Flatten first: GET_ROWS table batches must match the ID batches.
        auto * ids_flat = ggml_reshape_1d(ctx0, ggml_cont(ctx0, selected_experts), n_expert_used * n_tokens);
        auto * ids_hot = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, split_layer->ffn_expert_ids_hot, ids_flat), n_expert_used, n_tokens);
        auto * ids_cold = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, split_layer->ffn_expert_ids_cold, ids_flat), n_expert_used, n_tokens);
        cb(ids_hot, "ffn_moe_ids_hot", il);
        cb(ids_cold, "ffn_moe_ids_cold", il);
        auto * inp = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        auto ffn = [&](ggml_tensor * u, ggml_tensor * g, ggml_tensor * d, ggml_tensor * ids) {
            auto * up = build_lora_mm_id(u, inp, ids);
            auto * gate = build_lora_mm_id(g, inp, ids);
            auto * act = ggml_swiglu_split(ctx0, gate, up);
            return build_lora_mm_id(d, act, ids);
        };

        // CUDA MMQ's compaction assumes unique IDs per token. Dispatch hot slots
        // separately so multiple misses can safely share one zero expert.
        ggml_tensor * hot = nullptr;
        for (int64_t slot = 0; slot < n_expert_used; ++slot) {
            auto * ids = ggml_view_2d(ctx0, ids_hot, 1, n_tokens, ids_hot->nb[1], slot * ids_hot->nb[0]);
            auto * out = ffn(up_exps, gate_exps, down_exps, ids);
            hot = hot ? ggml_concat(ctx0, hot, out, 1) : out;
        }
        auto * cold = ffn(split_layer->ffn_up_exps_cold, split_layer->ffn_gate_exps_cold, split_layer->ffn_down_exps_cold, ids_cold);
        cb(hot, "ffn_moe_hot", il);
        cb(cold, "ffn_moe_cold", il);
        auto * experts = ggml_mul(ctx0, ggml_add(ctx0, hot, cold), weights);
        cb(experts, "ffn_moe_weighted", il);
        ggml_tensor * out = nullptr;
        for (int64_t slot = 0; slot < n_expert_used; ++slot) {
            auto * value = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], slot * experts->nb[1]);
            out = out ? ggml_add(ctx0, out, value) : value;
            ggml_build_forward_expand(gf, out);
        }
        if (n_expert_used == 1) {
            out = ggml_cont(ctx0, out);
        }
        cb(out, "ffn_moe_out", il);
        return out;
    }

''' + anchor
edit('src/llama-graph.cpp',anchor,new)
edit('ggml/src/ggml-cpu/ggml-cpu.c','''                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW''','''                if (i02 < 0) {
                    memset((char *) dst->data + iid1 * dst->nb[2] + id * dst->nb[1], 0, ne0 * sizeof(float));
                    continue;
                }
                GGML_ASSERT(i02 < n_as);

                MMID_MATRIX_ROW''')
