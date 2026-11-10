#include "llama.h"
#include "ggml-backend.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * what) {
    if (!ok) { throw std::runtime_error(what); }
}

static bool debug_cb(ggml_tensor * t, bool ask, void * user) {
    std::string name = t->name;
    bool use = name.find("ffn_moe_out-") == 0 || name.find("ffn_moe_topk-") == 0 || name.find("ffn_norm-") == 0;
    if (ask) { return use; }
    if (!use) { return true; }
    auto path = std::string("artifacts/codexR/debug/") + (const char *) user + "-" + name;
    std::vector<char> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, data.size());
    std::ofstream f(path + ".bin", std::ios::binary);
    f.write(data.data(), data.size());
    std::ofstream meta(path + ".txt");
    meta << t->type << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3];
    return true;
}
struct runner {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch_ext * batch = nullptr;
    runner(const char * path) {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_extra_bufts = false;
        model = llama_model_load_from_file(path, mp);
        check(model, "load model");
        auto cp = llama_context_default_params();
        cp.op_offload = false;
        cp.cb_eval = debug_cb;
        cp.cb_eval_user_data = (void *) (std::strstr(path, "split") ? "split" : "original");
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.n_ubatch = 128;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        ctx = llama_init_from_model(model, cp);
        check(ctx, "create CPU context");
        batch = llama_batch_ext_init(ctx);
        check(batch, "create batch");
    }
    ~runner() {
        if (batch) { llama_batch_ext_free(batch); }
        if (ctx) { llama_free(ctx); }
        if (model) { llama_model_free(model); }
    }
    void decode(const std::vector<llama_token> & tokens, int pos) {
        llama_batch_ext_clear(batch);
        for (size_t i = 0; i < tokens.size(); ++i) {
            int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            llama_pos p = pos + i;
            llama_batch_ext_set_pos(batch, idx, &p);
        }
        llama_batch_ext_set_output_logits(batch, tokens.size() - 1, true);
        check(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "CPU decode");
    }
};

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: compare-logits original.gguf split.gguf prompt.txt report.json\n");
        return 2;
    }
    try {
        llama_backend_init();
        std::ifstream input(argv[3]);
        check(input.good(), "open prompt");
        const std::string prompt((std::istreambuf_iterator<char>(input)), {});
        runner a(argv[1]), b(argv[2]);
        const auto * vocab = llama_model_get_vocab(a.model);
        const int nv = llama_vocab_n_tokens(vocab);
        check(nv == llama_vocab_n_tokens(llama_model_get_vocab(b.model)), "vocab mismatch");
        const int np = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        check(np > 0 && np + 64 < 512, "prompt length");
        std::vector<llama_token> tokens(np);
        check(llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) == np, "tokenize");
        a.decode(tokens, 0);
        b.decode(tokens, 0);
        double max_diff = 0, squared = 0;
        int differing = 0;
        std::vector<llama_token> ta, tb;
        std::vector<double> step_diff;
        for (int step = 0; step < 1; ++step) {
            const float * la = llama_get_logits_ith(a.ctx, -1);
            const float * lb = llama_get_logits_ith(b.ctx, -1);
            check(la && lb, "missing logits");
            double local_max = 0;
            for (int i = 0; i < nv; ++i) {
                check(std::isfinite(la[i]) && std::isfinite(lb[i]), "nonfinite logits");
                const double delta = std::abs(double(la[i]) - lb[i]);
                local_max = std::max(local_max, delta);
                squared += delta * delta;
            }
            max_diff = std::max(max_diff, local_max);
            step_diff.push_back(local_max);
            ta.push_back(std::max_element(la, la + nv) - la);
            tb.push_back(std::max_element(lb, lb + nv) - lb);
            differing += ta.back() != tb.back();
            std::fprintf(stderr, "step=%d original=%d split=%d max_abs_diff=%.9g\n", step, ta.back(), tb.back(), local_max);
            if (step != 0) {
                a.decode({ta.back()}, np + step);
                b.decode({tb.back()}, np + step);
            }
        }
        std::ofstream out(argv[4]);
        check(out.good(), "open report");
        out.precision(12);
        out << "{\n  \"backend\": \"CPU\",\n  \"threads\": 4,\n  \"prompt_tokens\": " << np
            << ",\n  \"continuation_tokens\": 64,\n  \"different_top1_tokens\": " << differing
            << ",\n  \"max_abs_logit_difference\": " << max_diff
            << ",\n  \"rms_logit_difference\": " << std::sqrt(squared / (64.0 * nv));
        auto array = [&](const char * key, const auto & values) {
            out << ",\n  \"" << key << "\": [";
            for (size_t i = 0; i < values.size(); ++i) { out << (i ? ", " : "") << values[i]; }
            out << ']';
        };
        array("original_tokens", ta);
        array("split_tokens", tb);
        array("step_max_abs_logit_difference", step_diff);
        out << "\n}\n";
        return differing ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
