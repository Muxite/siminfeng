#include "expert_fill.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <sys/wait.h>
#include <csignal>

// Freeze speculative dispatch at the same boundary used while demand builds
// jobs. This makes cancel-before-start deterministic, independent of IO speed.
struct ExpertFillerTestAccess {
    static std::vector<size_t> queued_layers(ExpertFiller & f) {
        std::lock_guard<std::mutex> lock(f.mutex_);
        std::vector<size_t> layers;
        for (const auto & job : f.async_queue_) { layers.push_back(job.layer); }
        return layers;
    }
    static std::vector<size_t> queued_distances(ExpertFiller & f) {
        std::lock_guard<std::mutex> lock(f.mutex_);
        std::vector<size_t> distances;
        for (const auto & job : f.async_queue_) { distances.push_back(job.distance); }
        return distances;
    }
    static void demand_priority(ExpertFiller & f, bool active) {
        { std::lock_guard<std::mutex> lock(f.mutex_); f.demand_active_ = active; }
        f.ready_.notify_all();
    }
};

static void test_cancel_before_start(const char * path, void * map, size_t size, size_t page) {
    setenv("LLAMA_EXPERT_FILL_THREADS", "8", 1);
    setenv("LLAMA_EXPERT_FILL_CHUNK_KIB", "64", 1);
    ExpertFiller filler(path, map, size);
    ExpertFillerTestAccess::demand_priority(filler, true);
    auto older = filler.enqueue_async({{0, 3 * page}}, 2);
    auto current = filler.enqueue_async({{8 * page, 2 * page}}, 3);
    auto future = filler.enqueue_async({{16 * page, 2 * page}}, 4);
    assert(older && current && future);
    assert(older->pending == 3 && current->pending == 2 && future->pending == 2);
    filler.cancel_async_through(3);
    assert(older->pending == 0 && current->pending == 0 && future->pending == 2);
    assert(older->running == 0 && current->running == 0);
    assert(older->cancelled_jobs == 3 && older->cancelled_bytes == 3 * page);
    assert(current->cancelled_jobs == 2 && current->cancelled_bytes == 2 * page);
    assert(older->bytes_read == 0 && current->bytes_read == 0);
    auto c = filler.async_counters();
    assert(c.cancelled_jobs == 5 && c.cancelled_bytes == 5 * page && c.bytes_read == 0);
    filler.cancel_async_through(3); // idempotent; future batch untouched
    assert(filler.async_counters().cancelled_jobs == 5);
    filler.cancel_async_through(SIZE_MAX); // token boundary retires all layers
    filler.wait_async(); // cancellation must wake/complete the drain condition
    assert(future->pending == 0 && future->cancelled_bytes == 2 * page);
    // A fresh batch after cancellation must dispatch and finish normally.
    auto fresh = filler.enqueue_async({{32 * page, page}}, 1);
    ExpertFillerTestAccess::demand_priority(filler, false);
    filler.wait_async();
    assert(fresh && fresh->pending == 0 && fresh->errors == 0);
    assert(filler.async_counters().cancelled_jobs == 7);
    puts("PASS test_cancel_before_start (older/current/future batches, bytes, idempotence, drain)");
}

static void test_async_inflight_cap(const char * path, void * map, size_t size) {
    setenv("LLAMA_EXPERT_FILL_THREADS", "8", 1);
    setenv("LLAMA_EXPERT_FILL_CHUNK_KIB", "64", 1);
    for (const char * cap : {"1", "2", "4", "64"}) {
        if (std::string(cap) == "4") { unsetenv("LLAMA_EXPERT_PREGATE_MAX_INFLIGHT"); }
        else { setenv("LLAMA_EXPERT_PREGATE_MAX_INFLIGHT", cap, 1); }
        const uint64_t expected = std::min<uint64_t>(8, std::stoul(cap));
        ExpertFiller filler(path, map, size);
        ExpertFillerTestAccess::demand_priority(filler, true);
        auto a = filler.enqueue_async({{0, size / 2}}, 1);
        auto b = filler.enqueue_async({{size / 2, size / 2}}, 2);
        assert(a && b);
        assert(filler.async_counters().max_inflight == 0);
        ExpertFillerTestAccess::demand_priority(filler, false);
        // Demand must complete even with two async batches and a saturated cap.
        filler.fill({{0, size}}, true);
        filler.wait_async();
        const auto c = filler.async_counters();
        assert(c.max_inflight >= 1 && c.max_inflight <= expected);
        assert(a->pending == 0 && b->pending == 0 && a->running == 0 && b->running == 0);
        assert(a->errors == 0 && b->errors == 0 && c.cancelled_jobs == 0);
        // Completion releases every cap slot, allowing subsequent submissions.
        auto next = filler.enqueue_async({{0, size / 2}}, 3);
        filler.wait_async();
        assert(next && next->pending == 0 && next->errors == 0);
        assert(filler.async_counters().max_inflight <= expected);
    }
    unsetenv("LLAMA_EXPERT_PREGATE_MAX_INFLIGHT");
    unsetenv("LLAMA_EXPERT_FILL_THREADS");
    unsetenv("LLAMA_EXPERT_FILL_CHUNK_KIB");
    puts("PASS test_async_inflight_cap (1, 2, default 4, cap 64 limited by 8 workers, demand progress, slot reuse)");
}

static void expect_config_abort(const char * name, const char * value, const char * range) {
    int output[2]; assert(pipe(output) == 0);
    const pid_t child = fork(); assert(child >= 0);
    if (!child) {
        close(output[0]); assert(dup2(output[1], STDERR_FILENO) >= 0); close(output[1]);
        struct rlimit limit {0, 0}; setrlimit(RLIMIT_CORE, &limit);
        setenv(name, value, 1);
        (void) ExpertConfig::read();
        _exit(0);
    }
    close(output[1]);
    std::string error; char buffer[1024]; ssize_t n;
    while ((n = read(output[0], buffer, sizeof(buffer))) > 0) { error.append(buffer, size_t(n)); }
    close(output[0]); int status = 0; assert(waitpid(child, &status, 0) == child);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    assert(error == std::string("expert_config: invalid ") + name + "=" + value + " (range " + range + ")\n");
}

static void test_strict_parser() {
    struct IntegerCase { const char * name; size_t lo, hi; };
    const IntegerCase integers[] = {
        {"LLAMA_EXPERT_FILL", 0, 1}, {"LLAMA_EXPERT_FILL_THREADS", 1, 64},
        {"LLAMA_EXPERT_FILL_CHUNK_KIB", 64, 65536},
        {"LLAMA_EXPERT_FILL_MIN_TOKENS", 0, 2147483647}, {"LLAMA_EXPERT_FILL_MAX_TOKENS", 0, 2147483647},
        {"LLAMA_EXPERT_FILL_DIAG", 0, 1}, {"LLAMA_EXPERT_FILL_POPULATE", 0, 1},
        {"LLAMA_EXPERT_PREGATE", 0, 1}, {"LLAMA_EXPERT_PREGATE_K", 1, 32},
        {"LLAMA_EXPERT_PREGATE_MAX_INFLIGHT", 1, 64}, {"LLAMA_EXPERT_PREGATE_LOOKAHEAD", 1, 2},
        {"LLAMA_EXPERT_PREFETCH", 0, 1}, {"LLAMA_EXPERT_PREFETCH_NEXT", 0, 1}};
    for (const auto & c : integers) {
        const std::string range = std::to_string(c.lo) + ".." + std::to_string(c.hi);
        for (auto v : {c.lo, c.hi, (c.lo + c.hi) / 2}) {
            setenv(c.name, std::to_string(v).c_str(), 1);
            (void) ExpertConfig::read();
            assert(ExpertConfig::integer(c.name, 999, c.lo, c.hi) == v);
        }
        unsetenv(c.name);
        for (const std::string & v : {std::string(""), std::string("abc"), std::string("-1"), std::string("+1"),
                std::string(" 1"), std::string("1 "), std::string("1x"), std::string("9999999999999999999999999999"),
                std::to_string(c.hi + 1)}) { expect_config_abort(c.name, v.c_str(), range.c_str()); }
        if (c.lo) { expect_config_abort(c.name, std::to_string(c.lo - 1).c_str(), range.c_str()); }
    }
    for (const char * k : {"1", "8", "10", "12", "16", "32"}) {
        setenv("LLAMA_EXPERT_PREGATE_K", k, 1);
        assert(ExpertConfig::read().k == std::stoul(k));
    }
    unsetenv("LLAMA_EXPERT_PREGATE_K");
    for (const char * k : {"0", "33", "abc"}) { expect_config_abort("LLAMA_EXPERT_PREGATE_K", k, "1..32"); }
    struct ChoiceCase { const char * name; const char * range; std::vector<const char *> values; };
    const ChoiceCase choices[] = {
        {"LLAMA_EXPERT_FILL_RANDOM", "0|1|decode", {"0", "1", "decode"}},
        {"LLAMA_EXPERT_PREGATE_ORDER", "after|before", {"after", "before"}}};
    for (const auto & c : choices) {
        for (auto value : c.values) { setenv(c.name, value, 1); (void) ExpertConfig::read(); }
        unsetenv(c.name);
        for (const char * value : {"", "typo", "AFTER", "2"}) { expect_config_abort(c.name, value, c.range); }
    }
    expect_config_abort("LLAMA_EXPERT_FILL_DIAG_CSV", "", "nonempty path");
    setenv("LLAMA_EXPERT_FILL_DIAG_CSV", "/tmp/test with spaces.csv", 1);
    assert(ExpertConfig::read().diag_csv == "/tmp/test with spaces.csv");
    unsetenv("LLAMA_EXPERT_FILL_DIAG_CSV");
    expect_config_abort("LLAMA_EXPERT_PREGATE_KK", "10", "known LLAMA_EXPERT_* name");
    expect_config_abort("LLAMA_EXPERT_FILL_THREAD", "16", "known LLAMA_EXPERT_* name");
    const auto c = ExpertConfig::read();
    assert(!c.fill && c.random == "0" && !c.max_tokens && !c.min_tokens && c.threads == 8 && c.chunk_kib == 1024);
    assert(!c.pregate && c.k == 8 && c.order == "after" && c.max_inflight == 4 && c.lookahead == 1);
    puts("PASS test_strict_parser (all vars, boundaries, K=10 accepted; K=0/33/abc abort; unknown names)");
}

static void test_lookahead_order(const char * path, void * map, size_t size) {
    ExpertFiller filler(path, map, size);
    ExpertFillerTestAccess::demand_priority(filler, true);
    // A previous L+2 batch must not stand in front of newly submitted L+1 work.
    auto far = filler.enqueue_async({{0, 65536}}, 4, 2);
    auto near = filler.enqueue_async({{65536, 65536}}, 3);
    auto far_next = filler.enqueue_async({{131072, 65536}}, 4, 2);
    assert((ExpertFillerTestAccess::queued_layers(filler) == std::vector<size_t>{3, 4, 4}));
    filler.cancel_async_through(3);
    assert(near->pending == 0 && near->cancelled_jobs == 1 && far->pending == 1 && far_next->pending == 1);
    assert((ExpertFillerTestAccess::queued_layers(filler) == std::vector<size_t>{4, 4}));
    auto fresh_near = filler.enqueue_async({{196608, 65536}}, 4, 1);
    assert((ExpertFillerTestAccess::queued_distances(filler) == std::vector<size_t>{1, 2, 2}));
    filler.cancel_async_through(4);
    assert(fresh_near->cancelled_jobs == 1);
    filler.wait_async();
    puts("PASS test_lookahead_order (L+1 before L+2, natural arrival cancellation)");
}

static void test_classification_counters(const char * path, void * map, size_t size) {
    using F = ExpertFiller;
    F filler(path, map, size);
    ExpertFillerTestAccess::demand_priority(filler, true);
    const std::vector<F::Range> ranges {{0, 65536}};
    F::Prediction resident, queued, done;
    resident.enqueue(filler, ranges, 2, [](const auto &) { return true; });
    queued.enqueue(filler, ranges, 2, [](const auto &) { return false; });
    done.enqueue(filler, ranges, 3, [](const auto &) { return false; });
    assert(resident.batches.empty() && resident.arrive() == F::Arrival::resident);
    assert(queued.arrive() == F::Arrival::queued);
    const auto snapshot = filler.arrive_async_through(2, {queued.batches.front().second});
    assert(snapshot.size() == 1 && snapshot[0].pending == 1 && snapshot[0].running == 0);
    assert(F::classify(false, snapshot[0]) == F::Arrival::queued);
    assert(queued.batches.front().second->pending == 0 && queued.batches.front().second->cancelled_jobs == 1);
    assert(queued.arrive() == F::Arrival::queued); // cancelled must never turn into done
    ExpertFillerTestAccess::demand_priority(filler, false);
    filler.wait_async();
    assert(done.arrive() == F::Arrival::done);
    // Synthetic coherent worker snapshots avoid races with actual disk speed.
    assert(F::classify(false, {1, 1, 0, 0}) == F::Arrival::inflight);
    assert(F::classify(false, {2, 1, 0, 0}) == F::Arrival::queued);
    assert(F::classify(false, {0, 0, 0, 1}) == F::Arrival::queued); // failed IO is not done
    F::Timeliness t;
    for (bool hit : {false, true}) {
        t.record(resident.arrive(), hit); t.record(queued.arrive(), hit);
        t.record(done.arrive(), hit); t.record(F::classify(false, {1, 1, 0, 0}), hit);
    }
    assert(t.pred_done == 2 && t.pred_inflight == 2 && t.pred_queued == 2 && t.pred_resident_at_predict == 2);
    assert(t.hit_done == 1 && t.hit_inflight == 1 && t.hit_queued == 1 && t.hit_resident == 1);
    puts("PASS test_classification_counters (fake residency oracle, done/inflight/queued/cancelled/resident, hits)");
}

int main() {
    test_strict_parser();
    char path[] = "/tmp/expert-fill-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    const size_t size = 64 * 1024 * 1024, page = size_t(sysconf(_SC_PAGESIZE));
    std::vector<unsigned char> block(1024 * 1024);
    for (size_t i = 0; i < block.size(); ++i) { block[i] = (i * 17 + i / 251) % 256; }
    for (size_t pos = 0; pos < size; pos += block.size()) {
        assert(pwrite(fd, block.data(), block.size(), pos) == ssize_t(block.size()));
    }
    assert(fsync(fd) == 0);
    auto * map = static_cast<unsigned char *>(mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0));
    assert(map != MAP_FAILED);
    assert(posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED) == 0);
    std::vector<unsigned char> residency(size / page);
    assert(mincore(map, size, residency.data()) == 0);
    size_t before = 0;
    for (auto v : residency) { before += (v & 1) != 0; }
    assert(before == 0);
    // Three unaligned, overlapping ranges form one page-aligned union.
    const std::vector<ExpertFiller::Range> ranges = {
        {4 * 1024 * 1024 + 17, 3 * 1024 * 1024},
        {6 * 1024 * 1024 + 33, 4 * 1024 * 1024},
        {9 * 1024 * 1024 + 55, 2 * 1024 * 1024}};
    const size_t first = ranges.front().offset / page * page;
    const size_t last = (ranges.back().offset + ranges.back().len + page - 1) / page * page;
    {
        ExpertFiller filler(path, map, size);
        filler.fill(ranges);
        assert(mincore(map, size, residency.data()) == 0);
        size_t extra = 0;
        for (size_t i = 0; i < residency.size(); ++i) {
            if (i * page >= first && i * page < last) { assert(residency[i] & 1); }
            else { extra += (residency[i] & 1) != 0; }
        }
        printf("measured readahead extra pages: %zu (%zu bytes)\n", extra, extra * page);
        const auto a = filler.counters();
        assert(a.calls == 1 && a.ranges == 3 && a.bytes_requested == last - first);
        assert(a.bytes_read > 0 && a.bytes_read <= last - first);
        assert(a.bytes_read + a.bytes_skipped_resident == last - first);
        assert(a.read_syscalls > 0 && a.total_fill_wall_ns >= a.max_single_fill_ns);
        assert(a.max_single_fill_ns > 0);
        filler.fill(ranges, true);
        const auto b = filler.counters();
        assert(a.prompt_calls == 1 && a.decode_calls == 0);
        assert(a.prompt_bytes_read == a.bytes_read && a.decode_bytes_read == 0);
        assert(a.prompt_fill_wall_ns == a.total_fill_wall_ns && a.decode_fill_wall_ns == 0);
        assert(b.prompt_calls == 1 && b.decode_calls == 1);
        assert(b.prompt_bytes_read == a.bytes_read && b.decode_bytes_read == 0);
        assert(b.prompt_fill_wall_ns == a.total_fill_wall_ns);
        assert(b.prompt_fill_wall_ns + b.decode_fill_wall_ns == b.total_fill_wall_ns);
        assert(b.calls == 2 && b.ranges == 6 && b.bytes_requested == 2 * (last - first));
        assert(b.bytes_read == a.bytes_read && b.read_syscalls == a.read_syscalls);
        assert(b.bytes_skipped_resident - a.bytes_skipped_resident == last - first);
        for (size_t i = first; i < last; ++i) { assert(map[i] == block[i % block.size()]); }
        filler.fill({{32 * 1024 * 1024, 1024 * 1024}}, true);
        const auto c = filler.counters();
        assert(c.prompt_calls == 1 && c.decode_calls == 2);
        assert(c.decode_bytes_read > 0 && c.prompt_bytes_read == a.bytes_read);
        assert(c.prompt_bytes_read + c.decode_bytes_read == c.bytes_read);
        assert(c.prompt_fill_wall_ns + c.decode_fill_wall_ns == c.total_fill_wall_ns);
        // A partial last page must not cause pread beyond EOF.
        bool rejected = false;
        try { filler.fill({{size - 1, 2}}); } catch (const std::out_of_range &) { rejected = true; }
        assert(rejected);
        const auto d = filler.counters();
        assert(d.prompt_calls + d.decode_calls == d.calls);
        assert(d.prompt_bytes_read + d.decode_bytes_read == d.bytes_read);
        assert(d.prompt_fill_wall_ns + d.decode_fill_wall_ns == d.total_fill_wall_ns);
    }
    setenv("LLAMA_EXPERT_FILL_CHUNK_KIB", "64", 1); // minimum supported chunk
    // A cold asynchronous submission populates cache through the shared worker pool;
    // only teardown/test code waits. The subsequent synchronous fill must read zero.
    assert(madvise(map, size, MADV_DONTNEED) == 0);
    assert(posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED) == 0);
    {
        ExpertFiller filler(path, map, size);
        const std::vector<ExpertFiller::Range> async_ranges = {
            {16 * 1024 * 1024 + 17, 2 * 1024 * 1024},
            {18 * 1024 * 1024 + 17, 2 * 1024 * 1024}};
        assert(filler.nonresident(async_ranges) > 0);
        auto batch = filler.enqueue_async(async_ranges);
        assert(batch);
        filler.wait_async();
        assert(batch->pending == 0 && batch->running == 0 && batch->errors == 0);
        assert(batch->bytes_read > 0 && batch->read_syscalls > 0);
        const auto async_stats = filler.async_counters();
        assert(async_stats.bytes_read == batch->bytes_read);
        assert(async_stats.read_syscalls == batch->read_syscalls && async_stats.errors == 0);
        assert(filler.nonresident(async_ranges) == 0);
        assert(filler.counters().calls == 0 && filler.counters().bytes_read == 0);
        assert(filler.fill(async_ranges, true) == 0);
        assert(filler.counters().bytes_read == 0 && filler.counters().decode_calls == 1);
        // Exercise sync/async concurrency on disjoint cold ranges; both must complete.
        auto concurrent = filler.enqueue_async({{24 * 1024 * 1024, 8 * 1024 * 1024}});
        filler.fill({{40 * 1024 * 1024, 2 * 1024 * 1024}}, true);
        filler.wait_async();
        assert(concurrent && concurrent->pending == 0 && concurrent->errors == 0);
        assert(filler.nonresident({{24 * 1024 * 1024, 8 * 1024 * 1024}}) == 0);
        assert(filler.nonresident({{40 * 1024 * 1024, 2 * 1024 * 1024}}) == 0);
        bool rejected = false;
        try { filler.enqueue_async({{size - 1, 2}}); } catch (const std::out_of_range &) { rejected = true; }
        assert(rejected);
    }
    unsetenv("LLAMA_EXPERT_FILL_CHUNK_KIB");
    test_cancel_before_start(path, map, size, 64 * 1024);
    test_async_inflight_cap(path, map, size);
    test_lookahead_order(path, map, size);
    test_classification_counters(path, map, size);
    setenv("LLAMA_EXPERT_FILL_POPULATE", "1", 1);
    setenv("LLAMA_EXPERT_FILL_DIAG", "1", 1);
    {
        ExpertFiller filler(path, map, size);
        const size_t begin = 48 * 1024 * 1024, len = 4 * 1024 * 1024;
        filler.fill({{begin + 17, len - 34}}, true);
        assert(filler.nonresident({{begin, len}}) == 0);
        const auto stats = filler.counters();
        assert(stats.populate_calls > 0 && stats.populate_ns > 0);
        // This test requires a Linux >=5.14 kernel implementing POPULATE_READ.
        assert(stats.populate_errors == 0);
        volatile unsigned char touched = 0;
        for (size_t i = begin; i < begin + len; i += page) { touched = map[i]; }
        struct rusage a {}, b {};
        assert(getrusage(RUSAGE_SELF, &a) == 0);
        for (size_t i = begin; i < begin + len; i += page) { touched = map[i]; }
        assert(getrusage(RUSAGE_SELF, &b) == 0);
        assert(b.ru_majflt == a.ru_majflt);
        (void)touched;
    }
    unsetenv("LLAMA_EXPERT_FILL_POPULATE"); unsetenv("LLAMA_EXPERT_FILL_DIAG");
    assert(munmap(map, size) == 0); close(fd); unlink(path);
    puts("PASS expert_fill: resident union, counters, warm skip, mmap contents, bounds, async completion and zero-read sync repeat, populate residency and zero-major-fault retouch");
}
