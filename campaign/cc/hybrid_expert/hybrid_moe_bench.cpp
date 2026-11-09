// hybrid_moe_bench: batch-1 MoE expert FFN with expert-granular GPU-hit / CPU-miss dispatch, on real Granite layers.
// Real weights (Q4_K experts) from GGUF; real routing (top-8 ids + weights per layer/token) replayed from trace JSONL.
// One "step" = one decode position dispatched through every selected layer in sequence (--layers 12 => single layer;
// --layers 0-23 => all 24 real layers, which keeps the weight working set (24 x 27 MiB) far above CPU L3 / GPU L2 as in a real
// decode; a single repeated layer is cache-hot and therefore optimistic).
// Paths (per layer):
//   gpu_ref : all 32 experts resident on GPU, one mul_mat_id graph over the 8 selected experts (llama.cpp formulation)
//   cpu_all : same graph on the ggml CPU backend (weights in CPU_REPACK like llama.cpp --n-cpu-moe, or plain with --cpu-repack 0)
//   hybrid  : K GPU slots per layer (slot table expert->slot|-1); hits -> GPU graph over slots (launched async), misses -> CPU
//             graph computed concurrently; y = y_gpu + y_cpu. Each side multiplies its own experts by their routing weights,
//             so every selected expert's weight is applied exactly once.
//   upload  : llama.cpp --moe-cache-mib semantics: misses copied H2D into LRU slots before the layer, then all 8 on GPU.
// Slot policies: static = top-K by PREFILL-phase routing frequency of that layer (decode tokens are not used to choose);
// LRU = cold start, after each token every miss is admitted (evicting least recently used, never a slot used by this token);
// the admission H2D cost is timed separately (between tokens).
// Numerics: fp64 reference from dequantized Q4_K weights and unquantized activations; tolerance rule in README.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static const int NE = 32, NU = 8;   // experts per layer, experts used per token (checked against tensor shapes)
static int D = 1024, F = 512;

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

struct Token { int ids[NU]; float w[NU]; int xi; };

static bool parse_line(const std::string & l, int layer_want, const char * phase_want, Token & t) {
    auto p = l.find("\"layer\":"); if (p == std::string::npos || atoi(l.c_str() + p + 8) != layer_want) return false;
    auto pp = l.find("\"phase\":\""); if (pp == std::string::npos || l.compare(pp + 9, strlen(phase_want), phase_want) != 0) return false;
    auto pe = l.find("\"experts\":["), pw = l.find("\"weights\":["); CHECK(pe != std::string::npos && pw != std::string::npos);
    const char * s = l.c_str() + pe + 11; char * e;
    for (int i = 0; i < NU; i++) { t.ids[i] = (int) strtol(s, &e, 10); s = e + 1; }
    s = l.c_str() + pw + 11;
    for (int i = 0; i < NU; i++) { t.w[i] = strtof(s, &e); s = e + 1; }
    return true;
}

// ---------------------------------------------------------------- MoE graph (llama.cpp build_moe_ffn op sequence: SILU-gated, no bias)
struct MoeG {
    ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; ggml_gallocr_t ga = nullptr;
    ggml_tensor *x = nullptr, *ids = nullptr, *w = nullptr, *out = nullptr; int n = 0;
    void free_() { if (ga) ggml_gallocr_free(ga); if (ctx) ggml_free(ctx); ga = nullptr; ctx = nullptr; }
};
struct Weights { ggml_tensor *g = nullptr, *u = nullptr, *d = nullptr; };
static MoeG build_moe(ggml_backend_t be, const Weights & W, int n) {
    MoeG g; g.n = n;
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.x = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, D, 1, 1); ggml_set_input(g.x);
    g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n, 1); ggml_set_input(g.ids);
    g.w = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, n, 1); ggml_set_input(g.w);
    ggml_tensor * gate = ggml_mul_mat_id(g.ctx, W.g, g.x, g.ids);   // [F, n, 1]
    ggml_tensor * up   = ggml_mul_mat_id(g.ctx, W.u, g.x, g.ids);   // [F, n, 1]
    ggml_tensor * act  = ggml_swiglu_split(g.ctx, gate, up);       // silu(gate) * up
    ggml_tensor * dn   = ggml_mul_mat_id(g.ctx, W.d, act, g.ids);   // [D, n, 1]
    ggml_tensor * ex   = ggml_mul(g.ctx, dn, g.w);                 // routing weights applied here, once per expert
    ggml_tensor * out  = ggml_view_2d(g.ctx, ex, D, 1, ex->nb[2], 0);
    for (int i = 1; i < n; i++) out = ggml_add(g.ctx, out, ggml_view_2d(g.ctx, ex, D, 1, ex->nb[2], i * ex->nb[1]));
    if (n == 1) out = ggml_cont(g.ctx, out);
    ggml_set_output(out); g.out = out;
    g.gf = ggml_new_graph(g.ctx); ggml_build_forward_expand(g.gf, out);
    g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    CHECK(ggml_gallocr_alloc_graph(g.ga, g.gf));
    return g;
}

static ggml_backend_t be_gpu, be_cpu;
static float *h_x, *h_w, *h_y; static int32_t * h_ids;   // pinned staging
static std::vector<float> y_cpu_buf;

struct Sample { double total = 0, gpu = 0, cpu = 0, wait = 0, admit = 0; int hits = 0; };

static void run_gpu_launch(MoeG & g, const float * x, const int * ids, const float * w) {
    memcpy(h_x, x, D * 4); memcpy(h_ids, ids, g.n * 4); memcpy(h_w, w, g.n * 4);
    ggml_backend_tensor_set_async(be_gpu, g.x, h_x, 0, D * 4);
    ggml_backend_tensor_set_async(be_gpu, g.ids, h_ids, 0, g.n * 4);
    ggml_backend_tensor_set_async(be_gpu, g.w, h_w, 0, g.n * 4);
    CHECK(ggml_backend_graph_compute_async(be_gpu, g.gf) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get_async(be_gpu, g.out, h_y, 0, D * 4);
}
static void run_cpu(MoeG & g, const float * x, const int * ids, const float * w, float * y) {
    memcpy(g.x->data, x, D * 4); memcpy(g.ids->data, ids, g.n * 4); memcpy(g.w->data, w, g.n * 4);
    CHECK(ggml_backend_graph_compute(be_cpu, g.gf) == GGML_STATUS_SUCCESS);
    memcpy(y, g.out->data, D * 4);
}

// ---------------------------------------------------------------- slot cache (one per layer)
struct Slots {
    int K = 0; Weights W, Wh; ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
    std::vector<int> slot_of, expert_in; std::vector<long> last_use; long clock = 0;
    MoeG g[NU + 1];
    void init(int k, const Weights & host) {
        K = k; Wh = host; ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true }; ctx = ggml_init(ip);
        W.g = ggml_new_tensor_3d(ctx, Wh.g->type, Wh.g->ne[0], Wh.g->ne[1], K);
        W.u = ggml_new_tensor_3d(ctx, Wh.u->type, Wh.u->ne[0], Wh.u->ne[1], K);
        W.d = ggml_new_tensor_3d(ctx, Wh.d->type, Wh.d->ne[0], Wh.d->ne[1], K);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be_gpu); CHECK(buf);
        for (int n = 1; n <= NU; n++) g[n] = build_moe(be_gpu, W, n);
        reset();
    }
    void reset() { slot_of.assign(NE, -1); expert_in.assign(K, -1); last_use.assign(K, -1); clock = 0; }
    void copy_in(int e, int s) {   // async H2D from the pinned host copy; caller synchronizes
        ggml_tensor * dsts[3] = { W.g, W.u, W.d }, * srcs[3] = { Wh.g, Wh.u, Wh.d };
        for (int k = 0; k < 3; k++)
            ggml_backend_tensor_set_async(be_gpu, dsts[k], (char *) srcs[k]->data + e * srcs[k]->nb[2], s * dsts[k]->nb[2], dsts[k]->nb[2]);
        if (expert_in[s] >= 0) slot_of[expert_in[s]] = -1;
        expert_in[s] = e; slot_of[e] = s;
    }
    int victim(const std::vector<char> & protect) {
        int best = -1;
        for (int s = 0; s < K; s++) { if (protect[s]) continue; if (expert_in[s] < 0) return s; if (best < 0 || last_use[s] < last_use[best]) best = s; }
        return best;
    }
    void free_() { for (int n = 1; n <= NU; n++) g[n].free_(); ggml_backend_buffer_free(buf); ggml_free(ctx); }
};

// ---------------------------------------------------------------- per-layer state
struct Layer {
    int il = 0; Weights Wh, Wc, Wf;  // pinned host (H2D source), CPU compute copy, GPU full copy
    MoeG cpu_g[NU + 1], ref_g;
    std::vector<std::vector<float>> X;      // synthetic activations (rms_norm(N(0,1)) * real ffn_norm)
    std::vector<Token> stream, prefill; std::vector<int> rank;
    std::vector<std::vector<Token>> synth;  // [m] exactly m misses w.r.t. static top-16
    Slots S;
};
static std::vector<Layer> L;

static Sample hybrid_layer(Layer & ly, const Token & t, int mode, float * y) {
    Slots & S = ly.S; Sample sm; double t0 = now_us();
    int hid[NU], mid[NU]; float hw[NU], mw[NU]; int h = 0, m = 0;
    for (int i = 0; i < NU; i++) {
        int s = S.slot_of[t.ids[i]];
        if (s >= 0) { hid[h] = s; hw[h++] = t.w[i]; S.last_use[s] = S.clock; } else { mid[m] = t.ids[i]; mw[m++] = t.w[i]; }
    }
    S.clock++; sm.hits = h;
    const float * x = ly.X[t.xi].data();
    if (mode == 0) {   // overlap: GPU async, CPU concurrently, then wait
        if (h) run_gpu_launch(S.g[h], x, hid, hw);
        double c0 = now_us();
        if (m) run_cpu(ly.cpu_g[m], x, mid, mw, y_cpu_buf.data());
        double c1 = now_us();
        if (h) ggml_backend_synchronize(be_gpu);
        sm.cpu = c1 - c0; sm.wait = now_us() - c1;
    } else {           // serial: GPU (launch+sync), then CPU
        double g0 = now_us();
        if (h) { run_gpu_launch(S.g[h], x, hid, hw); ggml_backend_synchronize(be_gpu); }
        double g1 = now_us();
        if (m) run_cpu(ly.cpu_g[m], x, mid, mw, y_cpu_buf.data());
        sm.gpu = g1 - g0; sm.cpu = now_us() - g1;
    }
    for (int i = 0; i < D; i++) y[i] = (h ? h_y[i] : 0.f) + (m ? y_cpu_buf[i] : 0.f);
    sm.total = now_us() - t0;
    return sm;
}
static double lru_admit(Slots & S, const Token & t, int & n) {
    double t0 = now_us(); std::vector<char> prot(S.K, 0); n = 0;
    for (int i = 0; i < NU; i++) { int s = S.slot_of[t.ids[i]]; if (s >= 0) prot[s] = 1; }
    for (int i = 0; i < NU; i++) {
        int e = t.ids[i]; if (S.slot_of[e] >= 0) continue;
        int s = S.victim(prot); if (s < 0) break;
        S.copy_in(e, s); S.last_use[s] = S.clock - 1; prot[s] = 1; n++;
    }
    if (n) ggml_backend_synchronize(be_gpu);
    return n ? now_us() - t0 : 0.0;
}
static Sample upload_layer(Layer & ly, const Token & t, float * y) {
    Slots & S = ly.S; Sample sm; double t0 = now_us(); std::vector<char> prot(S.K, 0); int sid[NU];
    for (int i = 0; i < NU; i++) { int s = S.slot_of[t.ids[i]]; if (s >= 0) { prot[s] = 1; sm.hits++; } }
    for (int i = 0; i < NU; i++) {
        int e = t.ids[i]; if (S.slot_of[e] < 0) { int s = S.victim(prot); CHECK(s >= 0); S.copy_in(e, s); prot[s] = 1; }
        sid[i] = S.slot_of[e]; S.last_use[sid[i]] = S.clock;
    }
    S.clock++;
    double a = now_us();
    run_gpu_launch(S.g[NU], ly.X[t.xi].data(), sid, t.w); ggml_backend_synchronize(be_gpu);   // copies precede on the same stream
    memcpy(y, h_y, D * 4);
    sm.total = now_us() - t0; sm.gpu = now_us() - a; sm.admit = a - t0;
    return sm;
}

// ---------------------------------------------------------------- fp64 reference
struct Deq { std::vector<float> g, u, d; };
static void dequant(const Weights & W, Deq & q) {
    auto deq = [](ggml_tensor * t, std::vector<float> & out) {
        auto * tr = ggml_get_type_traits(t->type); int64_t ne0 = t->ne[0], rows = t->ne[1] * t->ne[2];
        out.resize(ne0 * rows);
        for (int64_t r = 0; r < rows; r++) tr->to_float((char *) t->data + r * t->nb[1], out.data() + r * ne0, ne0);
    };
    deq(W.g, q.g); deq(W.u, q.u); deq(W.d, q.d);
}
static void ref64(const Deq & q, const float * x, const Token & t, std::vector<double> & y) {
    y.assign(D, 0.0); std::vector<double> a(F);
    for (int k = 0; k < NU; k++) {
        int e = t.ids[k];
        for (int f = 0; f < F; f++) {
            const float * rg = &q.g[((size_t) e * F + f) * D], * ru = &q.u[((size_t) e * F + f) * D];
            double sg = 0, su = 0; for (int i = 0; i < D; i++) { sg += (double) rg[i] * x[i]; su += (double) ru[i] * x[i]; }
            a[f] = sg / (1.0 + std::exp(-sg)) * su;
        }
        for (int o = 0; o < D; o++) {
            const float * rd = &q.d[((size_t) e * D + o) * F]; double s = 0;
            for (int f = 0; f < F; f++) s += (double) rd[f] * a[f];
            y[o] += (double) t.w[k] * s;
        }
    }
}
static double rel_l2(const float * y, const std::vector<double> & r) {
    double n = 0, d = 0; for (int i = 0; i < D; i++) { double e = y[i] - r[i]; n += e * e; d += r[i] * r[i]; } return std::sqrt(n / d);
}
static double max_abs(const float * a, const float * b) { double m = 0; for (int i = 0; i < D; i++) m = std::max(m, (double) std::fabs(a[i] - b[i])); return m; }
static double pct(std::vector<double> v, double p) { if (v.empty()) return NAN; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5))]; }

struct Config {
    std::string name, stream, policy, mode; int K = -1, synth_m = -1;
    std::vector<Sample> s;                                   // per step (summed over layers)
    std::vector<double> admit_us; std::vector<int> admitted; // LRU between-token admission per step (summed over layers)
    std::vector<std::vector<std::vector<float>>> outs;       // [layer][check token] outputs of the first timed pass
};

int main(int argc, char ** argv) {
    std::string model = "/models/granite-3.1-1b-a400m-instruct-mixE4K.gguf", traces = "/traces", out = "/out/results.json", layers_arg = "12";
    int threads = 8, passes = 2, synth_n = 300, warm = 30, cpu_repack = 1, check_n = 1000000;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { CHECK(i + 1 < argc); return std::string(argv[++i]); };
        if (a == "--model") model = nx(); else if (a == "--traces") traces = nx(); else if (a == "--out") out = nx();
        else if (a == "--layers") layers_arg = nx(); else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--cpu-repack") cpu_repack = std::stoi(nx()); else if (a == "--check-n") check_n = std::stoi(nx());
        else if (a == "--passes") passes = std::stoi(nx()); else if (a == "--synth-n") synth_n = std::stoi(nx());
        else if (a == "--warmup") warm = std::stoi(nx());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    std::vector<int> lids;
    { auto dash = layers_arg.find('-'); if (dash == std::string::npos) lids.push_back(std::stoi(layers_arg));
      else for (int l = std::stoi(layers_arg.substr(0, dash)); l <= std::stoi(layers_arg.substr(dash + 1)); l++) lids.push_back(l); }
    const int NL = (int) lids.size();
    double T0 = now_us();

    be_cpu = ggml_backend_cpu_init(); CHECK(be_cpu);
    ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    ggml_threadpool_t tp = ggml_threadpool_new(&tpp); CHECK(tp);
    ggml_backend_cpu_set_threadpool(be_cpu, tp); ggml_backend_cpu_set_n_threads(be_cpu, threads);
    be_gpu = ggml_backend_cuda_init(0); CHECK(be_gpu);
    ggml_backend_buffer_type_t repack_buft = nullptr;
    if (cpu_repack) {
        // llama.cpp places CPU-overridden expert tensors (--n-cpu-moe) in the first CPU *extra* buffer type
        // (llama-model-loader.cpp: "when overriding to a CPU buffer, consider the extra buffer types") = CPU_REPACK;
        // for Q4_K on AVX2 that selects the interleaved q4_K_8x8_q8_K kernels.
        ggml_backend_reg_t reg = ggml_backend_cpu_reg(); ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        for (auto * b = fn ? fn(dev) : nullptr; b && *b; b++) if (std::string(ggml_backend_buft_name(*b)) == "CPU_REPACK") repack_buft = *b;
        CHECK(repack_buft);
    }
    const std::string cpu_buft_name = cpu_repack ? "CPU_REPACK" : "pinned-plain";

    // ---- load layers
    ggml_context * meta = nullptr; gguf_init_params gp = { true, &meta };
    gguf_context * gg = gguf_init_from_file(model.c_str(), gp); CHECK(gg);
    FILE * mf = fopen(model.c_str(), "rb"); CHECK(mf); const size_t base = gguf_get_data_offset(gg);
    auto read_tensor = [&](const char * name, void * dst, size_t n) {
        int64_t id = gguf_find_tensor(gg, name); CHECK(id >= 0);
        CHECK(fseeko(mf, base + gguf_get_tensor_offset(gg, id), SEEK_SET) == 0); CHECK(fread(dst, 1, n, mf) == n);
    };
    L.resize(NL); size_t expert_bytes = 0; std::string types;
    for (int li = 0; li < NL; li++) {
        Layer & ly = L[li]; ly.il = lids[li];
        char nm[3][64]; const char * kinds[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" }; ggml_tensor * mt[3];
        for (int k = 0; k < 3; k++) { snprintf(nm[k], 64, "blk.%d.%s.weight", ly.il, kinds[k]); mt[k] = ggml_get_tensor(meta, nm[k]); CHECK(mt[k]); }
        char nn[64]; snprintf(nn, 64, "blk.%d.ffn_norm.weight", ly.il); ggml_tensor * mnorm = ggml_get_tensor(meta, nn);
        CHECK(mnorm && mnorm->type == GGML_TYPE_F32);
        D = (int) mt[0]->ne[0]; F = (int) mt[0]->ne[1]; CHECK(mt[0]->ne[2] == NE && mt[2]->ne[0] == F && mt[2]->ne[1] == D);
        if (li == 0) { for (int k = 0; k < 3; k++) { fprintf(stderr, "%s type=%s ne=[%lld,%lld,%lld]\n", nm[k], ggml_type_name(mt[k]->type),
            (long long) mt[k]->ne[0], (long long) mt[k]->ne[1], (long long) mt[k]->ne[2]); types += std::string(k ? "," : "") + ggml_type_name(mt[k]->type); } }
        ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context * hctx = ggml_init(ip);
        ly.Wh = { ggml_dup_tensor(hctx, mt[0]), ggml_dup_tensor(hctx, mt[1]), ggml_dup_tensor(hctx, mt[2]) };
        CHECK(ggml_backend_alloc_ctx_tensors_from_buft(hctx, ggml_backend_cuda_host_buffer_type()));
        read_tensor(nm[0], ly.Wh.g->data, ggml_nbytes(ly.Wh.g)); read_tensor(nm[1], ly.Wh.u->data, ggml_nbytes(ly.Wh.u));
        read_tensor(nm[2], ly.Wh.d->data, ggml_nbytes(ly.Wh.d));
        std::vector<float> norm(D); read_tensor(nn, norm.data(), D * 4);
        std::mt19937 rng(1234 + ly.il); std::normal_distribution<float> nd(0, 1);
        for (int v = 0; v < 64; v++) {
            std::vector<float> x(D); double ss = 0; for (auto & z : x) { z = nd(rng); ss += z * z; }
            float sc = 1.0f / std::sqrt((float) (ss / D) + 1e-6f); for (int i = 0; i < D; i++) x[i] *= sc * norm[i];
            ly.X.push_back(x);
        }
        ly.Wc = ly.Wh;
        if (cpu_repack) {
            ggml_context * rctx = ggml_init(ip);
            ly.Wc = { ggml_dup_tensor(rctx, mt[0]), ggml_dup_tensor(rctx, mt[1]), ggml_dup_tensor(rctx, mt[2]) };
            CHECK(ggml_backend_alloc_ctx_tensors_from_buft(rctx, repack_buft));
            ggml_backend_tensor_set(ly.Wc.g, ly.Wh.g->data, 0, ggml_nbytes(ly.Wh.g));
            ggml_backend_tensor_set(ly.Wc.u, ly.Wh.u->data, 0, ggml_nbytes(ly.Wh.u));
            ggml_backend_tensor_set(ly.Wc.d, ly.Wh.d->data, 0, ggml_nbytes(ly.Wh.d));
        }
        ggml_context * fctx = ggml_init(ip);
        ly.Wf = { ggml_dup_tensor(fctx, mt[0]), ggml_dup_tensor(fctx, mt[1]), ggml_dup_tensor(fctx, mt[2]) };
        CHECK(ggml_backend_alloc_ctx_tensors(fctx, be_gpu));
        ggml_backend_tensor_set(ly.Wf.g, ly.Wh.g->data, 0, ggml_nbytes(ly.Wh.g));
        ggml_backend_tensor_set(ly.Wf.u, ly.Wh.u->data, 0, ggml_nbytes(ly.Wh.u));
        ggml_backend_tensor_set(ly.Wf.d, ly.Wh.d->data, 0, ggml_nbytes(ly.Wh.d));
        expert_bytes = ly.Wh.g->nb[2] + ly.Wh.u->nb[2] + ly.Wh.d->nb[2];
        for (int n = 1; n <= NU; n++) ly.cpu_g[n] = build_moe(be_cpu, ly.Wc, n);
        ly.ref_g = build_moe(be_gpu, ly.Wf, NU);
        // traces: decode tokens of prose, code, chat concatenated (the timed stream); prefill tokens -> static profile
        for (auto tag : { "prose", "code", "chat" }) {
            std::ifstream f(traces + "/" + tag + ".jsonl"); CHECK(f.good()); std::string l;
            while (std::getline(f, l)) { Token t; t.xi = 0;
                if (parse_line(l, ly.il, "decode", t)) ly.stream.push_back(t); else if (parse_line(l, ly.il, "prefill", t)) ly.prefill.push_back(t); }
        }
        for (size_t i = 0; i < ly.stream.size(); i++) ly.stream[i].xi = i % ly.X.size();
        std::vector<int> freq(NE, 0); for (auto & t : ly.prefill) for (int i = 0; i < NU; i++) freq[t.ids[i]]++;
        ly.rank.resize(NE); for (int i = 0; i < NE; i++) ly.rank[i] = i;
        std::stable_sort(ly.rank.begin(), ly.rank.end(), [&](int a, int b) { return freq[a] > freq[b]; });
        // synthetic: exactly m misses w.r.t. static top-16; real routing weights from decode tokens; ids drawn at random
        const int KS = 16; ly.synth.assign(NU + 1, {});
        std::mt19937 r2(42 + ly.il); std::vector<int> res(ly.rank.begin(), ly.rank.begin() + KS), non(ly.rank.begin() + KS, ly.rank.end());
        for (int m = 0; m <= NU; m++) for (int i = 0; i < synth_n; i++) {
            Token t = ly.stream[i % ly.stream.size()]; std::shuffle(res.begin(), res.end(), r2); std::shuffle(non.begin(), non.end(), r2);
            std::vector<int> pick; for (int k = 0; k < NU - m; k++) pick.push_back(res[k]); for (int k = 0; k < m; k++) pick.push_back(non[k]);
            std::shuffle(pick.begin(), pick.end(), r2); for (int k = 0; k < NU; k++) t.ids[k] = pick[k];
            t.xi = i % ly.X.size(); ly.synth[m].push_back(t);
        }
        CHECK(ly.stream.size() == L[0].stream.size());
    }
    fclose(mf);
    const size_t NT = L[0].stream.size();
    fprintf(stderr, "%d layer(s) [%d..%d], %zu decode positions per layer, CPU miss weights %s, expert bytes %zu (%.3f MiB)\n",
        NL, lids.front(), lids.back(), NT, cpu_buft_name.c_str(), expert_bytes, expert_bytes / 1048576.0);
    y_cpu_buf.resize(D);
    ggml_backend_buffer_t sbuf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 4 * (3 * 1024 + 64));
    char * sb = (char *) ggml_backend_buffer_get_base(sbuf);
    h_x = (float *) sb; h_y = (float *) (sb + 4096 * 2); h_w = (float *) (sb + 4096 * 3); h_ids = (int32_t *) (sb + 4096 * 3 + 128);

    // ---- fp64 references for the check subset (first check_n positions of each stream, per layer), parallel and untimed
    const size_t CN = std::min((size_t) check_n, NT), CS = std::min((size_t) check_n, (size_t) synth_n);
    std::vector<std::vector<std::vector<double>>> R_stream(NL);                // [layer][tok]
    std::vector<std::vector<std::vector<std::vector<double>>>> R_synth(NL);    // [layer][m][tok]
    {
        int nt = std::max(1, threads); Deq q;
        for (int li = 0; li < NL; li++) {
            R_stream[li].resize(CN); R_synth[li].assign(NU + 1, std::vector<std::vector<double>>(CS));
            dequant(L[li].Wh, q);
            std::vector<std::thread> th; const size_t jobs = CN + (NU + 1) * CS;
            for (int k = 0; k < nt; k++) th.emplace_back([&, k, li] {
                for (size_t j = k; j < jobs; j += nt) {
                    if (j < CN) ref64(q, L[li].X[L[li].stream[j].xi].data(), L[li].stream[j], R_stream[li][j]);
                    else { size_t jj = j - CN, m = jj / CS, i = jj % CS; ref64(q, L[li].X[L[li].synth[m][i].xi].data(), L[li].synth[m][i], R_synth[li][m][i]); }
                }
            });
            for (auto & t : th) t.join();
        }
    }
    fprintf(stderr, "fp64 reference done (%.1f s since start)\n", (now_us() - T0) / 1e6);

    std::vector<Config> cfgs; std::vector<float> y(D);
    // per-step function: dispatch one position through every layer, sum timings
    using LayerFn = std::function<Sample(Layer &, const Token &, float *)>;
    auto run = [&](Config & c, std::function<const Token &(Layer &, size_t)> tok, size_t n, LayerFn fn,
                   std::function<void(Layer &)> reset, bool lru_post, size_t keep) {
        auto step = [&](size_t i, bool rec, int p) {
            Sample tot; double adm = 0; int admn = 0;
            for (int li = 0; li < NL; li++) {
                Layer & ly = L[li]; const Token & t = tok(ly, i);
                Sample s = fn(ly, t, y.data());
                tot.total += s.total; tot.gpu += s.gpu; tot.cpu += s.cpu; tot.wait += s.wait; tot.admit += s.admit; tot.hits += s.hits;
                if (p == 0 && rec && i < keep) c.outs[li][i] = y;
                if (lru_post) { int k = 0; adm += lru_admit(ly.S, t, k); admn += k; }
            }
            if (rec) { c.s.push_back(tot); if (lru_post) { c.admit_us.push_back(adm); c.admitted.push_back(admn); } }
        };
        for (auto & ly : L) reset(ly);
        for (size_t i = 0; i < std::min(n, (size_t) warm); i++) step(i, false, -1);
        c.outs.assign(NL, std::vector<std::vector<float>>(keep, std::vector<float>(D)));
        for (int p = 0; p < passes; p++) { for (auto & ly : L) reset(ly); for (size_t i = 0; i < n; i++) step(i, true, p); }
    };
    auto tok_stream = [](Layer & ly, size_t i) -> const Token & { return ly.stream[i]; };
    auto gpu_ref_fn = [](Layer & ly, const Token & t, float * yy) {
        Sample s; double t0 = now_us(); run_gpu_launch(ly.ref_g, ly.X[t.xi].data(), t.ids, t.w); ggml_backend_synchronize(be_gpu);
        memcpy(yy, h_y, D * 4); s.total = s.gpu = now_us() - t0; s.hits = NU; return s;
    };
    auto cpu_all_fn = [](Layer & ly, const Token & t, float * yy) {
        Sample s; double t0 = now_us(); run_cpu(ly.cpu_g[NU], ly.X[t.xi].data(), t.ids, t.w, yy); s.total = s.cpu = now_us() - t0; return s;
    };
    auto noreset = [](Layer &) {};
    auto mk = [](std::string name, std::string stream, std::string pol, std::string mode, int K, int m) {
        Config c; c.name = name; c.stream = stream; c.policy = pol; c.mode = mode; c.K = K; c.synth_m = m; return c; };

    { auto c = mk("gpu_ref", "trace", "all_gpu", "gpu", 32, -1); run(c, tok_stream, NT, gpu_ref_fn, noreset, false, CN); cfgs.push_back(std::move(c)); }
    { auto c = mk("cpu_all", "trace", "all_cpu", "cpu", 0, -1); run(c, tok_stream, NT, cpu_all_fn, noreset, false, CN); cfgs.push_back(std::move(c)); }
    for (int K : { 8, 12, 16, 24, 32 }) {
        for (auto & ly : L) ly.S.init(K, ly.Wh);
        auto fill_static = [K](Layer & ly) { ly.S.reset(); for (int s = 0; s < K; s++) ly.S.copy_in(ly.rank[s], s); ggml_backend_synchronize(be_gpu); };
        for (int mode : { 0, 1 }) {
            auto c = mk(std::string("hybrid_static_") + (mode ? "serial" : "overlap") + "_K" + std::to_string(K), "trace", "static_prefill_profile", mode ? "serial" : "overlap", K, -1);
            run(c, tok_stream, NT, [mode](Layer & ly, const Token & t, float * yy) { return hybrid_layer(ly, t, mode, yy); }, fill_static, false, CN);
            cfgs.push_back(std::move(c));
        }
        if (K < 32) {
            auto c = mk("hybrid_lru_overlap_K" + std::to_string(K), "trace", "lru_admit_all_misses", "overlap", K, -1);
            run(c, tok_stream, NT, [](Layer & ly, const Token & t, float * yy) { return hybrid_layer(ly, t, 0, yy); }, [](Layer & ly) { ly.S.reset(); }, true, CN);
            cfgs.push_back(std::move(c));
            auto u = mk("upload_lru_K" + std::to_string(K), "trace", "lru_upload_on_miss", "gpu_upload", K, -1);
            run(u, tok_stream, NT, upload_layer, [](Layer & ly) { ly.S.reset(); }, false, CN);
            cfgs.push_back(std::move(u));
        }
        for (auto & ly : L) ly.S.free_();
    }
    // synthetic controlled miss counts with K=16 static, plus pure components
    {
        const int KS = 16;
        for (auto & ly : L) ly.S.init(KS, ly.Wh);
        auto fill_static = [](Layer & ly) { ly.S.reset(); for (int s = 0; s < 16; s++) ly.S.copy_in(ly.rank[s], s); ggml_backend_synchronize(be_gpu); };
        for (int m = 0; m <= NU; m++) {
            auto tok_m = [m](Layer & ly, size_t i) -> const Token & { return ly.synth[m][i]; };
            for (int mode : { 0, 1 }) {
                auto c = mk(std::string("synth_") + (mode ? "serial" : "overlap") + "_m" + std::to_string(m), "synth", "static_prefill_profile", mode ? "serial" : "overlap", KS, m);
                run(c, tok_m, synth_n, [mode](Layer & ly, const Token & t, float * yy) { return hybrid_layer(ly, t, mode, yy); }, fill_static, false, CS);
                cfgs.push_back(std::move(c));
            }
            if (m == 0) continue;
            auto c = mk("cpu_only_n" + std::to_string(m), "synth", "component", "cpu", 0, m);
            run(c, tok_m, synth_n, [m](Layer & ly, const Token & t, float * yy) {
                Sample s; double t0 = now_us(); run_cpu(ly.cpu_g[m], ly.X[t.xi].data(), t.ids, t.w, yy); s.total = s.cpu = now_us() - t0; return s; }, noreset, false, 0);
            cfgs.push_back(std::move(c));
            std::vector<MoeG> gg2; for (auto & ly : L) gg2.push_back(build_moe(be_gpu, ly.Wf, m));
            auto g = mk("gpu_full_only_n" + std::to_string(m), "synth", "component", "gpu", 32, m);
            run(g, tok_m, synth_n, [&](Layer & ly, const Token & t, float * yy) {
                MoeG & G2 = gg2[&ly - L.data()]; Sample s; double t0 = now_us(); run_gpu_launch(G2, ly.X[t.xi].data(), t.ids, t.w);
                ggml_backend_synchronize(be_gpu); memcpy(yy, h_y, D * 4); s.total = s.gpu = now_us() - t0; return s; }, noreset, false, 0);
            for (auto & q : gg2) q.free_();
            cfgs.push_back(std::move(g));
        }
        for (auto & ly : L) ly.S.free_();
    }
    // micro: H2D admission of one expert (pinned vs pageable source), and the bare GPU round trip (4 KB H2D + 4 KB D2H + sync)
    std::vector<double> h2d_pinned, h2d_pageable, rtt;
    {
        Slots S; S.init(8, L[0].Wh); std::vector<char> pageable(expert_bytes);
        for (int r = 0; r < 330; r++) { double t0 = now_us(); S.copy_in(r % NE, r % 8); ggml_backend_synchronize(be_gpu); if (r >= 30) h2d_pinned.push_back(now_us() - t0); }
        for (int r = 0; r < 330; r++) {
            int s = r % 8; double t0 = now_us(); size_t off = 0;
            for (auto dst : { S.W.g, S.W.u, S.W.d }) { ggml_backend_tensor_set(dst, pageable.data() + off, s * dst->nb[2], dst->nb[2]); off += dst->nb[2]; }
            if (r >= 30) h2d_pageable.push_back(now_us() - t0);
        }
        MoeG & g = S.g[1];
        for (int r = 0; r < 330; r++) {
            double t0 = now_us(); ggml_backend_tensor_set_async(be_gpu, g.x, h_x, 0, D * 4); ggml_backend_tensor_get_async(be_gpu, g.x, h_y, 0, D * 4);
            ggml_backend_synchronize(be_gpu); if (r >= 30) rtt.push_back(now_us() - t0);
        }
        S.free_();
    }

    // ---- correctness
    // Rule per (layer, token): e = rel-L2(out, fp64). PASS iff e_hybrid <= 1.5 * max(e_gpu_ref, e_cpu_all) + 1e-6, where the
    // endpoints are the same-token outputs of the all-GPU and all-CPU graphs. Plus bitwise: all-hit hybrid == gpu_ref and
    // zero-hit hybrid == cpu_all (same kernels, same order).
    std::vector<std::vector<std::vector<float>>> G_s(NL * (NU + 1)), C_s(NL * (NU + 1));
    for (int li = 0; li < NL; li++) for (int m = 0; m <= NU; m++) for (size_t i = 0; i < CS; i++) {
        std::vector<float> a(D), b(D); gpu_ref_fn(L[li], L[li].synth[m][i], a.data()); cpu_all_fn(L[li], L[li].synth[m][i], b.data());
        G_s[li * (NU + 1) + m].push_back(a); C_s[li * (NU + 1) + m].push_back(b);
    }
    auto & G = cfgs[0]; auto & C = cfgs[1];
    std::vector<double> eg, ec;
    for (int li = 0; li < NL; li++) for (size_t i = 0; i < CN; i++) { eg.push_back(rel_l2(G.outs[li][i].data(), R_stream[li][i])); ec.push_back(rel_l2(C.outs[li][i].data(), R_stream[li][i])); }
    std::ostringstream J; J.precision(4); J << std::fixed;
    J << "{\n\"meta\":{\"model\":\"" << model << "\",\"layers\":[";
    for (int li = 0; li < NL; li++) J << (li ? "," : "") << lids[li];
    J << "],\"threads\":" << threads << ",\"passes\":" << passes << ",\"cpu_weights\":\"" << cpu_buft_name << "\",\"warmup_steps\":" << warm
      << ",\"stream_positions\":" << NT << ",\"synth_n\":" << synth_n << ",\"check_positions\":" << CN << ",\"expert_bytes\":" << expert_bytes
      << ",\"types\":\"" << types << "\",\"timing\":\"host steady_clock around each layer dispatch (incl. staging, launch, sync, combine); per-step = sum over layers\""
      << ",\"label\":\"UNSCORED/exploratory\"},\n";
    auto arrd = [&](const std::vector<double> & v) { J << "["; for (size_t i = 0; i < v.size(); i++) J << (i ? "," : "") << v[i]; J << "]"; };
    J << "\"micro\":{\"h2d_one_expert_pinned_us_median\":" << pct(h2d_pinned, .5) << ",\"h2d_one_expert_pinned_us_p95\":" << pct(h2d_pinned, .95)
      << ",\"h2d_one_expert_pageable_us_median\":" << pct(h2d_pageable, .5) << ",\"gpu_roundtrip_4KB_us_median\":" << pct(rtt, .5)
      << ",\"h2d_pinned\":"; arrd(h2d_pinned); J << ",\"h2d_pageable\":"; arrd(h2d_pageable); J << ",\"roundtrip\":"; arrd(rtt); J << "},\n\"configs\":[\n";
    bool all_ok = true;
    fprintf(stderr, "\n(per step = sum over %d layer(s); us)\n%-26s %6s %9s %9s %9s %9s %9s %8s %9s %s\n", NL, "config", "hit%", "med", "p95", "gpu_med", "cpu_med", "wait_med", "admit", "maxrel", "check");
    for (size_t ci = 0; ci < cfgs.size(); ci++) {
        auto & c = cfgs[ci]; std::vector<double> tot, gpu, cpu, wait, ain; double hits = 0;
        for (auto & s : c.s) { tot.push_back(s.total); gpu.push_back(s.gpu); cpu.push_back(s.cpu); wait.push_back(s.wait); ain.push_back(s.admit); hits += s.hits; }
        double hit = hits / (c.s.size() * (double) NU * NL);
        std::string chk = "n/a"; double maxrel = -1, maxabs_ref = -1; int fails = 0, n = 0, zb = 0;
        if (!c.outs.empty() && !c.outs[0].empty()) {
            maxrel = 0; maxabs_ref = 0; const bool tr = c.stream == "trace";
            for (int li = 0; li < NL; li++) for (size_t i = 0; i < c.outs[li].size(); i++) {
                const auto & R = tr ? R_stream[li][i] : R_synth[li][c.synth_m][i];
                const float * gref = tr ? G.outs[li][i].data() : G_s[li * (NU + 1) + c.synth_m][i].data();
                const float * cref = tr ? C.outs[li][i].data() : C_s[li * (NU + 1) + c.synth_m][i].data();
                double e = rel_l2(c.outs[li][i].data(), R), egi = rel_l2(gref, R), eci = rel_l2(cref, R);
                maxrel = std::max(maxrel, e); n++;
                maxabs_ref = std::max(maxabs_ref, max_abs(c.outs[li][i].data(), gref));
                if (e > 1.5 * std::max(egi, eci) + 1e-6) fails++;
                if (c.synth_m == NU && max_abs(c.outs[li][i].data(), cref) != 0.0) zb++;
            }
            chk = fails ? "FAIL" : "ok";
            if (c.synth_m == NU) chk += zb ? ",FAIL(zero-hit not bit-identical to cpu_all)" : ",bitwise==cpu_all";
            if (c.name == "hybrid_static_overlap_K32" || c.name == "hybrid_static_serial_K32" || c.synth_m == 0)
                chk += maxabs_ref != 0.0 ? ",FAIL(all-hit not bit-identical to gpu_ref)" : ",bitwise==gpu_ref";
            if (chk.find("FAIL") != std::string::npos) all_ok = false;
        }
        double admit_mean = 0, adm_n = 0;
        for (size_t i = 0; i < c.admit_us.size(); i++) { admit_mean += c.admit_us[i]; adm_n += c.admitted[i]; }
        if (!c.admit_us.empty()) { admit_mean /= c.admit_us.size(); adm_n /= c.admit_us.size(); }
        fprintf(stderr, "%-26s %6.1f %9.1f %9.1f %9.1f %9.1f %9.1f %8.1f %9.2e %s\n", c.name.c_str(), hit * 100, pct(tot, .5), pct(tot, .95),
            pct(gpu, .5), pct(cpu, .5), pct(wait, .5), admit_mean, maxrel, chk.c_str());
        J << (ci ? ",\n" : "") << "{\"name\":\"" << c.name << "\",\"stream\":\"" << c.stream << "\",\"policy\":\"" << c.policy << "\",\"mode\":\"" << c.mode
          << "\",\"K\":" << c.K << ",\"synth_misses\":" << c.synth_m << ",\"n\":" << c.s.size() << ",\"hit_rate\":" << hit
          << ",\"total_us_median\":" << pct(tot, .5) << ",\"total_us_p95\":" << pct(tot, .95) << ",\"gpu_us_median\":" << pct(gpu, .5)
          << ",\"cpu_us_median\":" << pct(cpu, .5) << ",\"wait_us_median\":" << pct(wait, .5) << ",\"admit_in_step_us_median\":" << pct(ain, .5)
          << ",\"admit_between_steps_us_mean\":" << admit_mean << ",\"admitted_per_step_mean\":" << adm_n
          << ",\"check\":\"" << chk << "\",\"check_items\":" << n << ",\"check_fails\":" << fails
          << ",\"max_rel_l2_vs_fp64\":" << std::scientific << maxrel << ",\"max_abs_vs_gpu_ref\":" << maxabs_ref << std::fixed;
        J << ",\"total_us\":"; arrd(tot); J << ",\"gpu_us\":"; arrd(gpu); J << ",\"cpu_us\":"; arrd(cpu); J << ",\"wait_us\":"; arrd(wait);
        J << ",\"hits\":["; for (size_t i = 0; i < c.s.size(); i++) J << (i ? "," : "") << c.s[i].hits; J << "]";
        if (!c.admit_us.empty()) { J << ",\"admit_between_steps_us\":"; arrd(c.admit_us); }
        J << "}";
    }
    J << "\n],\n\"endpoint_rel_l2_vs_fp64\":{\"gpu_ref_median\":" << std::scientific << pct(eg, .5) << ",\"gpu_ref_max\":" << pct(eg, 1) << ",\"cpu_all_median\":" << pct(ec, .5)
      << ",\"cpu_all_max\":" << pct(ec, 1) << "},\n\"all_checks_ok\":" << (all_ok ? "true" : "false") << "\n}\n";
    std::ofstream(out) << J.str();
    fprintf(stderr, "endpoint rel-L2 vs fp64: gpu_ref median %.2e max %.2e | cpu_all median %.2e max %.2e\n", pct(eg, .5), pct(eg, 1), pct(ec, .5), pct(ec, 1));
    fprintf(stderr, "H2D one expert: pinned median %.1f us p95 %.1f | pageable median %.1f | GPU 4KB round trip median %.1f us\n",
        pct(h2d_pinned, .5), pct(h2d_pinned, .95), pct(h2d_pageable, .5), pct(rtt, .5));
    fprintf(stderr, "ALL CHECKS %s; wrote %s; wall %.1f s\n", all_ok ? "OK" : "FAILED", out.c_str(), (now_us() - T0) / 1e6);
    return all_ok ? 0 : 2;
}
