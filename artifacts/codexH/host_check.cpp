
#include "ggml.h"
#include "ggml-backend.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <iostream>
using cudaStream_t = void *;
#define MATRIX_ROW_PADDING 512
#define QK8_1 32
#define CUDA_CHECK(x) assert((x) == 0)
#undef GGML_ASSERT
#define GGML_ASSERT(x) assert(x)
struct block_q8_1 { char data[36]; };
struct ggml_backend_buffer { ggml_backend_buffer_usage usage; };
ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t b) { return b->usage; }
size_t ggml_nbytes(const ggml_tensor * t) {
    size_t n = 4;
    for (int i=0; i<4; ++i) n += (t->ne[i]-1)*t->nb[i];
    return n;
}
bool ggml_is_contiguous(const ggml_tensor * t) {
    size_t stride=4;
    for (int i=0; i<4; ++i) { if(t->ne[i]>1 && t->nb[i]!=stride) return false; stride*=t->ne[i]; }
    return true;
}
int allocations=0;
int cudaMalloc(void ** p, size_t n) { ++allocations; *p=std::malloc(n); return *p ? 0 : 1; }
template<class T> struct ggml_cuda_pool_alloc {
    T * p=nullptr;
    T * alloc(int, size_t n) { return p=(T*)std::malloc(n); }
    ~ggml_cuda_pool_alloc() { std::free(p); }
};
struct ggml_cuda_q8_reuse {
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


struct ggml_cgraph { int n_nodes; ggml_tensor ** nodes; };
struct stream_context_t { std::unordered_map<int,int> concurrent_events; };
struct ggml_backend_cuda_context {
    ggml_cuda_q8_reuse q8_reuse;
    int curr_stream_no=0;
    stream_context_t sc;
    stream_context_t & stream_context() { return sc; }
    int pool() { return 0; }
    ~ggml_backend_cuda_context() { for(auto e:q8_reuse.buffers) std::free(e.second); }
};
static bool ggml_cuda_q8_reuse_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_Q8_REUSE");
        return !(value && value[0] == '0' && value[1] == '\0');
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


int quantizations=0;
uint64_t fingerprint(const float * p, int64_t n) {
    uint64_t h=0;
    for(int64_t i=0;i<n;++i) { uint32_t u; std::memcpy(&u,p+i,4); h=h*65599+u; }
    return h;
}
void quantize_row_q8_1_cuda(const float * p, const int32_t * ids, void * dst, ggml_type,
        int64_t n, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t) {
    assert(!ids); ++quantizations;
    uint64_t h=fingerprint(p,n); std::memcpy(dst,&h,8);
}
uint64_t quant_block(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        bool use_ids=false, bool use_fusion=false, cudaStream_t stream=(void*)1) {
    const void * ids=use_ids ? (void*)1 : nullptr;
    const void * fusion=use_fusion ? (void*)1 : nullptr;
    const int64_t ne10=src1->ne[0], ne11=src1->ne[1], ne12=src1->ne[2], ne13=src1->ne[3];
    const size_t ts_src1=4;
    const float * src1_d=(const float*)src1->data;
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t q8_bytes = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1;
    auto & reuse = ctx.q8_reuse;
    const auto cached_buffer = reuse.buffers.find(q8_bytes);
    const bool can_reuse = reuse.allowed && !ids && !fusion && src0->type == GGML_TYPE_Q8_0 &&
        ne11 == 1 && ne12 == 1 && ne13 == 1 && cached_buffer != reuse.buffers.end();
    void * src1_q8_data = can_reuse ? cached_buffer->second : src1_q8_1.alloc(ctx.pool(), q8_bytes);
    if (!can_reuse || !reuse.matches(src1, stream)) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_data, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        if (can_reuse) {
            reuse.remember(src1, stream);
        }
    }


    uint64_t h; std::memcpy(&h,src1_q8_data,8); return h;
}
uint64_t last_hash=0;
bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
    if (node->op == GGML_OP_MUL_MAT) last_hash=quant_block(ctx,node->src[0],node->src[1]);
    return true;
}
void prepare_buffers(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
        bool use_cuda_graph=false, bool cuda_graph_update_required=false) {
    // Allocate outside capture. Never replace storage referenced by an older graph.
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


}
void step(ggml_backend_cuda_context * cuda_ctx, ggml_tensor * node, bool fused=false, bool should_launch_concurrent_events=false) {
                auto & reuse = cuda_ctx->q8_reuse;
                const bool reuse_candidate = ggml_cuda_q8_reuse_enabled() && !should_launch_concurrent_events &&
                    ggml_cuda_q8_reuse_candidate(node);
                if ((!reuse_candidate && node->op != GGML_OP_ROPE) ||
                        (reuse.src && ggml_cuda_q8_reuse_overlaps(node, reuse.layout))) {
                    reuse.src = nullptr;
                }


    if(fused) { reuse.src=nullptr; assert(!reuse.allowed); return; }
                reuse.allowed = reuse_candidate && cuda_ctx->curr_stream_no == 0 &&
                    !ggml_cuda_q8_reuse_overlaps(node, *node->src[1]);
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                reuse.allowed = false;

    assert(ok); assert(!reuse.allowed);
}
ggml_tensor tensor(void * p, int64_t n=512) {
    ggml_tensor t={}; t.type=GGML_TYPE_F32; t.data=p; t.ne[0]=n; t.nb[0]=4;
    for(int i=1;i<4;++i) { t.ne[i]=1; t.nb[i]=n*4; }
    return t;
}
int main(int argc, char ** argv) {
    const bool enabled=argc==1 || std::atoi(argv[1]);
    assert(ggml_cuda_q8_reuse_enabled()==enabled);
    std::vector<float> storage(4096,1.0f);
    ggml_tensor input=tensor(storage.data()), other=input;
    ggml_tensor weight={}; weight.type=GGML_TYPE_Q8_0;
    ggml_backend_buffer buffer={GGML_BACKEND_BUFFER_USAGE_WEIGHTS}; weight.buffer=&buffer;
    ggml_tensor mm=tensor(storage.data()+2048); mm.op=GGML_OP_MUL_MAT; mm.src[0]=&weight; mm.src[1]=&input;
    ggml_tensor rope=tensor(storage.data()+3072); rope.op=GGML_OP_ROPE;
    ggml_backend_cuda_context ctx;
    ggml_tensor * nodes[]={&mm};
    // Real ggml_cgraph is internal; its minimal stub is supplied below.
    ggml_cgraph graph={1,nodes}; prepare_buffers(&ctx,&graph);
    assert(allocations==(enabled?1:0));
    const auto first_buffers=ctx.q8_reuse.buffers;
    prepare_buffers(&ctx,&graph,true,false); assert(allocations==(enabled?1:0));
    step(&ctx,&mm); step(&ctx,&rope); step(&ctx,&mm); step(&ctx,&mm);
    assert(quantizations==(enabled?1:3)); assert(last_hash==fingerprint(storage.data(),512));
    int before=quantizations;
    // Distinct tensor objects sharing an address must miss.
    mm.src[1]=&other; step(&ctx,&mm); assert(quantizations==before+1); mm.src[1]=&input;
    step(&ctx,&mm);
    if(enabled) {
        auto & r=ctx.q8_reuse;
        assert(r.matches(&input,(void*)1)); assert(!r.matches(&input,(void*)2));
        ++input.nb[2]; assert(!r.matches(&input,(void*)1)); --input.nb[2];
        ++input.ne[2]; assert(!r.matches(&input,(void*)1)); --input.ne[2];
        input.type=GGML_TYPE_F16; assert(!r.matches(&input,(void*)1)); input.type=GGML_TYPE_F32;
        input.data=storage.data()+1; assert(!r.matches(&input,(void*)1)); input.data=storage.data();
    }
    // A partial alias write through RoPE must invalidate, even with a different tensor address.
    rope.data=storage.data()+1; step(&ctx,&rope); storage[1]=2.0f;
    before=quantizations; step(&ctx,&mm); assert(quantizations==before+1);
    assert(last_hash==fingerprint(storage.data(),512));
    // Adjacent ranges do not overlap; one-float partial overlaps do.
    rope.data=storage.data()+512; assert(!ggml_cuda_q8_reuse_overlaps(&rope,input));
    rope.data=storage.data()+511; assert(ggml_cuda_q8_reuse_overlaps(&rope,input));
    rope.data=storage.data()-0; assert(ggml_cuda_q8_reuse_overlaps(&rope,input));
    // Unknown writes and successful fusions terminate the window.
    rope.op=GGML_OP_ADD; rope.data=storage.data()+3072; step(&ctx,&rope);
    before=quantizations; step(&ctx,&mm); assert(quantizations==before+1);
    step(&ctx,&mm,true); before=quantizations; step(&ctx,&mm); assert(quantizations==before+1);
    // Concurrent evaluation uses the pool and does not seed a cache entry.
    step(&ctx,&mm,false,true); assert(ctx.q8_reuse.src==nullptr);
    // New graph evaluation must not use the previous token's quantization.
    step(&ctx,&mm); ctx.q8_reuse.src=nullptr; ctx.q8_reuse.allowed=false; storage[2]=3.0f;
    before=quantizations; step(&ctx,&mm); assert(quantizations==before+1);
    assert(last_hash==fingerprint(storage.data(),512));
    // A larger graph keeps the previous captured pointer alive.
    input=tensor(storage.data(),1024); prepare_buffers(&ctx,&graph);
    if(enabled) { assert(allocations==2); for(auto e:first_buffers) assert(ctx.q8_reuse.buffers.at(e.first)==e.second); }
    // Eligibility exclusions: batch, weight type, temporary weight padding, and MoE.
    input.ne[1]=2; assert(!ggml_cuda_q8_reuse_candidate(&mm)); input.ne[1]=1;
    weight.type=GGML_TYPE_Q4_0; assert(!ggml_cuda_q8_reuse_candidate(&mm)); weight.type=GGML_TYPE_Q8_0;
    buffer.usage=GGML_BACKEND_BUFFER_USAGE_COMPUTE; assert(!ggml_cuda_q8_reuse_candidate(&mm)); buffer.usage=GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    mm.op=GGML_OP_MUL_MAT_ID; assert(!ggml_cuda_q8_reuse_candidate(&mm)); mm.op=GGML_OP_MUL_MAT;
    // Even if a future caller sets allowed, ids/fusion must still quantize afresh.
    ctx.q8_reuse.allowed=true;
    before=quantizations; quant_block(ctx,&weight,&input,true); quant_block(ctx,&weight,&input,false,true);
    assert(quantizations==before+2);
    std::cout << "PASS: host cache identity, overwrite, window, lifetime and fallback checks; enabled=" << enabled << "\n";
}
