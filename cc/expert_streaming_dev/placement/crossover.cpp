// Q4_K MUL_MAT_ID curve with pageable host weights and selected-expert uploads.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * what) {
    if (!ok) { throw std::runtime_error(what); }
}
struct tensors {
    ggml_context * ctx;
    ggml_backend_buffer_t buf = nullptr;
    tensors() {
        ggml_init_params p = {ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true};
        ctx = ggml_init(p);
        require(ctx != nullptr, "ggml_init failed");
    }
    void alloc(ggml_backend_t backend) {
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buf != nullptr, "allocation failed");
    }
    ~tensors() {
        if (buf) { ggml_backend_buffer_free(buf); }
        ggml_free(ctx);
    }
    tensors(const tensors &) = delete;
    tensors & operator=(const tensors &) = delete;
};
struct op_graph : tensors {
    ggml_tensor * b;
    ggml_tensor * ids;
    ggml_tensor * out;
    ggml_cgraph * graph;
    op_graph(ggml_backend_t backend, ggml_tensor * a, int channels, int n) {
        b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a->ne[0], channels, n);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, n);
        out = ggml_mul_mat_id(ctx, a, b, ids);
        require(ggml_backend_supports_op(backend, out), "backend does not support this MUL_MAT_ID");
        graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, out);
        alloc(backend);
    }
};
static void compute(ggml_backend_t backend, ggml_cgraph * graph) {
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
    ggml_backend_synchronize(backend);
}
template<class F> static double median_ms(F f) {
    std::array<double, 3> times;
    for (double & t : times) {
        const auto start = std::chrono::steady_clock::now();
        f();
        t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    std::sort(times.begin(), times.end());
    return times[1];
}
struct upload_range { size_t offset, bytes; };
static std::vector<upload_range> ranges(const ggml_tensor * a, const std::vector<int32_t> & ids, int & selected) {
    std::array<bool, 256> used{};
    for (int id : ids) { require(id >= 0 && id < 256, "invalid id"); used[id] = true; }
    selected = std::count(used.begin(), used.end(), true);
    std::vector<upload_range> result;
    for (int first = 0; first < 256;) {
        if (!used[first]) { ++first; continue; }
        int last = first;
        while (last + 1 < 256 && used[last + 1]) { ++last; }
        const size_t padding = last < 255 ? std::min<size_t>(a->nb[2], 512) : 0;
        result.push_back({first * a->nb[2], (last + 1 - first) * a->nb[2] + padding});
        first = last + 1;
    }
    return result;
}
static void run_shape(ggml_backend_t cpu, ggml_backend_t gpu, const char * shape,
                      int k, int m, int channels, bool smoke) {
    tensors weights_cpu;
    auto * a_cpu = ggml_new_tensor_3d(weights_cpu.ctx, GGML_TYPE_Q4_K, k, m, 256);
    weights_cpu.alloc(cpu);
    ggml_backend_buffer_set_usage(weights_cpu.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);
    std::vector<float> one_expert(size_t(k) * m);
    for (auto & x : one_expert) { x = uniform(rng); }
    const size_t quantized = ggml_quantize_chunk(GGML_TYPE_Q4_K, one_expert.data(), a_cpu->data, 0, m, k, nullptr);
    require(quantized == a_cpu->nb[2], "unexpected Q4_K expert size");
    for (int e = 1; e < 256; ++e) {
        std::memcpy(static_cast<char *>(a_cpu->data) + e * a_cpu->nb[2], a_cpu->data, a_cpu->nb[2]);
    }
    tensors weights_gpu;
    ggml_tensor * a_gpu = nullptr;
    if (gpu) {
        a_gpu = ggml_new_tensor_3d(weights_gpu.ctx, GGML_TYPE_Q4_K, k, m, 256);
        weights_gpu.alloc(gpu);
        ggml_backend_buffer_set_usage(weights_gpu.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_memset(a_gpu, 0, 0, ggml_nbytes(a_gpu));
    }
    std::array<int32_t, 256> perm;
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<int32_t> router_stream(8 * 512);
    for (int t = 0; t < 512; ++t) {
        std::shuffle(perm.begin(), perm.end(), rng);
        std::copy_n(perm.begin(), 8, router_stream.begin() + 8 * t);
    }
    for (int n : {1, 2, 4, 8, 16, 32, 64, 128, 512}) {
        if (smoke && n != 1) { continue; }
        op_graph c(cpu, a_cpu, channels, n);
        std::vector<int32_t> ids(router_stream.begin(), router_stream.begin() + 8 * n);
        std::vector<float> activation(ggml_nelements(c.b));
        for (auto & x : activation) { x = uniform(rng); }
        ggml_backend_tensor_set(c.b, activation.data(), 0, ggml_nbytes(c.b));
        ggml_backend_tensor_set(c.ids, ids.data(), 0, ggml_nbytes(c.ids));
        compute(cpu, c.graph);
        const double cpu_ms = median_ms([&] { compute(cpu, c.graph); });
        int selected = 0;
        const auto uploads = ranges(a_cpu, ids, selected);
        size_t upload_bytes = 0;
        for (auto r : uploads) { upload_bytes += r.bytes; }
        double gpu_ms = -1, staged_ms = -1, cpu_boundary_ms = -1, nmse = -1;
        if (gpu) {
            op_graph g(gpu, a_gpu, channels, n);
            ggml_backend_tensor_set(g.b, activation.data(), 0, ggml_nbytes(g.b));
            ggml_backend_tensor_set(g.ids, ids.data(), 0, ggml_nbytes(g.ids));
            auto upload = [&] {
                for (auto r : uploads) {
                    ggml_backend_tensor_set_async(gpu, a_gpu, static_cast<const char *>(a_cpu->data) + r.offset, r.offset, r.bytes);
                }
            };
            upload();
            compute(gpu, g.graph);
            const size_t out_bytes = ggml_nbytes(c.out);
            std::vector<float> reference(ggml_nelements(c.out)), actual(reference.size());
            ggml_backend_tensor_get(c.out, reference.data(), 0, out_bytes);
            ggml_backend_tensor_get(g.out, actual.data(), 0, out_bytes);
            double err = 0, norm = 0;
            for (size_t i = 0; i < reference.size(); ++i) {
                err += std::pow(double(actual[i]) - reference[i], 2);
                norm += double(reference[i]) * reference[i];
            }
            nmse = err / std::max(norm, 1e-30);
            require(std::isfinite(nmse) && nmse < 5e-4, "CPU/GPU NMSE exceeds 5e-4");
            gpu_ms = median_ms([&] { compute(gpu, g.graph); });
            std::vector<int32_t> router(ids.size());
            // This includes the callback's GPU router readback, dedup, upload and compute.
            staged_ms = median_ms([&] {
                ggml_backend_tensor_get_async(gpu, g.ids, router.data(), 0, ggml_nbytes(g.ids));
                ggml_backend_synchronize(gpu);
                int count;
                const auto current = ranges(a_cpu, router, count);
                for (auto r : current) {
                    ggml_backend_tensor_set_async(gpu, a_gpu, static_cast<const char *>(a_cpu->data) + r.offset, r.offset, r.bytes);
                }
                compute(gpu, g.graph);
            });
            // CPU expert boundary: GPU activations/ids in, GPU result out.
            cpu_boundary_ms = median_ms([&] {
                ggml_backend_tensor_get_async(gpu, g.b, c.b->data, 0, ggml_nbytes(c.b));
                ggml_backend_tensor_get_async(gpu, g.ids, c.ids->data, 0, ggml_nbytes(c.ids));
                ggml_backend_synchronize(gpu);
                compute(cpu, c.graph);
                ggml_backend_tensor_set_async(gpu, g.out, c.out->data, 0, out_bytes);
                ggml_backend_synchronize(gpu);
            });
        }
        std::printf("%s,%d,%d,%d,%zu,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.8g\n",
                    shape, k, m, n, size_t(channels), selected, uploads.size(), upload_bytes,
                    cpu_ms, cpu_boundary_ms, gpu_ms, staged_ms, nmse);
        std::fflush(stdout);
    }
}
int main(int argc, char ** argv) {
    ggml_backend_t cpu = nullptr, gpu = nullptr;
    try {
        bool cpu_only = false, smoke = false;
        int threads = 6;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--cpu-only") { cpu_only = true; }
            else if (arg == "--smoke") { smoke = true; }
            else if (arg == "--threads" && i + 1 < argc) { threads = std::stoi(argv[++i]); }
            else { throw std::runtime_error("usage: expert-crossover [--cpu-only] [--smoke] [--threads 6]"); }
        }
        require(threads > 0 && threads <= 64, "invalid thread count");
        ggml_backend_load_all();
        cpu = ggml_backend_cpu_init();
        require(cpu != nullptr, "CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(cpu, threads);
        if (!cpu_only) {
            gpu = ggml_backend_init_by_name("CUDA0", nullptr);
            require(gpu != nullptr, "CUDA0 unavailable (use --cpu-only for local validation)");
        }
        std::fprintf(stderr, "expert-crossover threads=%d ids=seed19_uniform_top8 reps=3 warm_pageable_weights=1\n", threads);
        std::puts("shape,k,m,n_tokens,activation_channels,selected,copies,upload_bytes,cpu_ms,cpu_boundary_ms,gpu_resident_ms,gpu_staged_ms,nmse");
        run_shape(cpu, gpu, "up_gate", 2048, 512, 1, smoke);
        run_shape(cpu, gpu, "down", 512, 2048, 8, smoke);
        if (gpu) { ggml_backend_free(gpu); }
        ggml_backend_free(cpu);
        ggml_quantize_free();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "expert-crossover: %s\n", e.what());
        if (gpu) { ggml_backend_free(gpu); }
        if (cpu) { ggml_backend_free(cpu); }
        return 1;
    }
}
