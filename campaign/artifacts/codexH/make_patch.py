from pathlib import Path
import difflib

root = Path(__file__).resolve().parents[2]
work = root / 'artifacts/codexH/work/ggml/src/ggml-cuda'
ref = root / 'ref/llama.cpp/ggml/src/ggml-cuda'

def edit(name, old, new):
    p = work / name
    s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old), old[:80])
    p.write_text(s.replace(old, new))

# Start from the original files so the script is repeatable.
for name in ('common.cuh', 'mmvq.cu', 'ggml-cuda.cu'):
    (work / name).write_bytes((ref / name).read_bytes())

edit('common.cuh', 'struct ggml_backend_cuda_context {', '''struct ggml_cuda_q8_reuse {
    // Keep all sizes alive: older captured graphs can still use them.
    std::unordered_map<size_t, void *> buffers;
    const ggml_tensor * src = nullptr;
    ggml_tensor layout = {};
    cudaStream_t stream = nullptr;
    bool allowed = false;

    bool matches(const ggml_tensor * tensor, cudaStream_t current_stream) const {
        if (src != tensor || stream != current_stream || layout.data != tensor->data || layout.type != tensor->type) {
            return false;
        }
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (layout.ne[i] != tensor->ne[i] || layout.nb[i] != tensor->nb[i]) {
                return false;
            }
        }
        return true;
    }

    void remember(const ggml_tensor * tensor, cudaStream_t current_stream) {
        src = tensor;
        layout = *tensor;
        stream = current_stream;
    }
};

struct ggml_backend_cuda_context {''')
edit('common.cuh', '    int curr_stream_no = 0;', '''    int curr_stream_no = 0;
    ggml_cuda_q8_reuse q8_reuse;''')

edit('ggml-cuda.cu', '''    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {''', '''    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (!q8_reuse.buffers.empty()) {
        ggml_cuda_set_device(device);
#ifdef USE_CUDA_GRAPH
        cuda_graphs.clear();
#endif
        for (const auto & entry : q8_reuse.buffers) {
            CUDA_CHECK(cudaFree(entry.second));
        }
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {''')

edit('ggml-cuda.cu', 'static void ggml_cuda_graph_evaluate_and_capture(', '''static bool ggml_cuda_q8_reuse_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_Q8_REUSE");
        return !(value && value[0] == '0' && value[1] == '\\0');
    }();
    return enabled;
}

static bool ggml_cuda_q8_reuse_candidate(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1]) {
        return false;
    }
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * input  = node->src[1];
    return weight->type == GGML_TYPE_Q8_0 && input->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
        input->ne[0] > 0 && input->ne[1] == 1 && input->ne[2] == 1 && input->ne[3] == 1 &&
        ggml_is_contiguous(input) &&
        ggml_backend_buffer_get_usage(weight->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
}

static bool ggml_cuda_q8_reuse_overlaps(const ggml_tensor * node, const ggml_tensor & input) {
    const uintptr_t a = (uintptr_t) node->data;
    const uintptr_t b = (uintptr_t) input.data;
    // Subtract addresses instead of adding lengths, to avoid end-address overflow.
    return a <= b ? b - a < ggml_nbytes(node) : a - b < ggml_nbytes(&input);
}

static void ggml_cuda_graph_evaluate_and_capture(''')

edit('ggml-cuda.cu', '''                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {''', '''                auto & reuse = cuda_ctx->q8_reuse;
                const bool reuse_candidate = ggml_cuda_q8_reuse_enabled() && !should_launch_concurrent_events &&
                    ggml_cuda_q8_reuse_candidate(node);
                if ((!reuse_candidate && node->op != GGML_OP_ROPE) ||
                        (reuse.src && ggml_cuda_q8_reuse_overlaps(node, reuse.layout))) {
                    reuse.src = nullptr;
                }

                // Fused kernels use their original scratch allocation and cannot hit this cache.
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
                    reuse.src = nullptr;''')
edit('ggml-cuda.cu', '''                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {''', '''                reuse.allowed = reuse_candidate && cuda_ctx->curr_stream_no == 0 &&
                    !ggml_cuda_q8_reuse_overlaps(node, *node->src[1]);
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                reuse.allowed = false;
                if (!ok) {''')

edit('ggml-cuda.cu', '''    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;''', '''    cuda_ctx->q8_reuse.src = nullptr;
    cuda_ctx->q8_reuse.allowed = false;
    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;''')

edit('ggml-cuda.cu', '''    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture''', '''    // Allocate outside capture. Never replace storage referenced by an older graph.
    if ((!use_cuda_graph || cuda_graph_update_required) && ggml_cuda_q8_reuse_enabled() && cuda_ctx->stream_context().concurrent_events.empty()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (!ggml_cuda_q8_reuse_candidate(node)) {
                continue;
            }
            const int64_t n = node->src[1]->ne[0];
            GGML_ASSERT(n <= INT64_MAX - MATRIX_ROW_PADDING);
            const uint64_t blocks = GGML_PAD(n, MATRIX_ROW_PADDING) / QK8_1;
            GGML_ASSERT(blocks <= SIZE_MAX / sizeof(block_q8_1));
            const size_t bytes = blocks * sizeof(block_q8_1);
            if (cuda_ctx->q8_reuse.buffers.count(bytes) == 0) {
                void * ptr = nullptr;
                CUDA_CHECK(cudaMalloc(&ptr, bytes));
                cuda_ctx->q8_reuse.buffers.emplace(bytes, ptr);
            }
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture''')

edit('mmvq.cu', '''    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {''', '''    const size_t q8_bytes = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1;
    auto & reuse = ctx.q8_reuse;
    const auto cached_buffer = reuse.buffers.find(q8_bytes);
    const bool can_reuse = reuse.allowed && !ids && !fusion && src0->type == GGML_TYPE_Q8_0 &&
        ne11 == 1 && ne12 == 1 && ne13 == 1 && cached_buffer != reuse.buffers.end();
    void * src1_q8_data = can_reuse ? cached_buffer->second : src1_q8_1.alloc(ctx.pool(), q8_bytes);
    if (!can_reuse || !reuse.matches(src1, stream)) {''')
edit('mmvq.cu', '''        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01''', '''        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_data, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        if (can_reuse) {
            reuse.remember(src1, stream);
        }
    }

    const int64_t s01''')
edit('mmvq.cu', '''        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,''', '''        src0->data, src0->type, src1_q8_data, ids_d, fusion_local, dst_d, ne00,''')

patch = ''
for name in ('common.cuh', 'ggml-cuda.cu', 'mmvq.cu'):
    p = 'ggml/src/ggml-cuda/' + name
    patch += f'diff --git a/{p} b/{p}\n'
    patch += ''.join(difflib.unified_diff((ref/name).read_text().splitlines(True), (work/name).read_text().splitlines(True), fromfile='a/'+p, tofile='b/'+p))
(root/'cc/q8_reuse/q8-activation-reuse.patch').write_text(patch)
