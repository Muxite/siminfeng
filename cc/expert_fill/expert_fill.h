#pragma once
// Linux buffered reads populate the same page cache as the model mmap.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <sys/resource.h>
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <list>
#include <memory>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <thread>
#include <unistd.h>
#include <vector>

// All LLAMA_EXPERT_* settings are validated even when fill/pregate is off.
// Decimal digits only (no signs/whitespace); 0 token limits mean unlimited.
extern char ** environ;
struct ExpertConfig {
    size_t fill = 0, threads = 8, chunk_kib = 1024, min_tokens = 0, max_tokens = 0;
    size_t diag = 0, populate = 0, pregate = 0, k = 8, max_inflight = 4, lookahead = 1;
    size_t prefetch = 0, prefetch_next = 0;
    std::string random = "0", order = "after", diag_csv;
    [[noreturn]] static void invalid(const char * name, const char * value, const std::string & range) {
        fprintf(stderr, "expert_config: invalid %s=%s (range %s)\n", name, value, range.c_str());
        fflush(stderr);
        std::abort(); // Deliberately outside the recoverable fill/predict exception paths.
    }
    static size_t integer(const char * name, size_t fallback, size_t lo, size_t hi) {
        const char * value = getenv(name);
        if (!value) { return fallback; }
        size_t n = 0;
        bool valid = *value != 0;
        for (const char * p = value; *p; ++p) {
            if (*p < '0' || *p > '9' || n > hi / 10 ||
                    (n == hi / 10 && size_t(*p - '0') > hi % 10)) { valid = false; break; }
            n = n * 10 + size_t(*p - '0');
        }
        if (!valid || n < lo || n > hi) {
            invalid(name, value, std::to_string(lo) + ".." + std::to_string(hi));
        }
        return n;
    }
    static std::string choice(const char * name, const char * fallback, std::initializer_list<const char *> choices) {
        const char * value = getenv(name);
        if (!value) { return fallback; }
        std::string range;
        for (const char * c : choices) {
            if (std::string(value) == c) { return value; }
            if (!range.empty()) { range += "|"; }
            range += c;
        }
        invalid(name, value, range);
    }
    static ExpertConfig read() {
        const std::vector<std::string> known = {
            "LLAMA_EXPERT_FILL", "LLAMA_EXPERT_FILL_RANDOM", "LLAMA_EXPERT_FILL_THREADS",
            "LLAMA_EXPERT_FILL_CHUNK_KIB", "LLAMA_EXPERT_FILL_MIN_TOKENS", "LLAMA_EXPERT_FILL_MAX_TOKENS",
            "LLAMA_EXPERT_FILL_DIAG", "LLAMA_EXPERT_FILL_DIAG_CSV", "LLAMA_EXPERT_FILL_POPULATE",
            "LLAMA_EXPERT_PREGATE", "LLAMA_EXPERT_PREGATE_K", "LLAMA_EXPERT_PREGATE_ORDER",
            "LLAMA_EXPERT_PREGATE_MAX_INFLIGHT", "LLAMA_EXPERT_PREGATE_LOOKAHEAD",
            "LLAMA_EXPERT_PREFETCH", "LLAMA_EXPERT_PREFETCH_NEXT"};
        for (char ** e = environ; e && *e; ++e) {
            const std::string entry(*e);
            const size_t eq = entry.find('=');
            const std::string name = entry.substr(0, eq);
            if (name.compare(0, 13, "LLAMA_EXPERT_") == 0 &&
                    std::find(known.begin(), known.end(), name) == known.end()) {
                invalid(name.c_str(), eq == std::string::npos ? "" : entry.c_str() + eq + 1, "known LLAMA_EXPERT_* name");
            }
        }
        ExpertConfig c;
        c.fill = integer("LLAMA_EXPERT_FILL", 0, 0, 1);
        c.random = choice("LLAMA_EXPERT_FILL_RANDOM", "0", {"0", "1", "decode"});
        c.threads = integer("LLAMA_EXPERT_FILL_THREADS", 8, 1, 64);
        c.chunk_kib = integer("LLAMA_EXPERT_FILL_CHUNK_KIB", 1024, 64, 65536);
        c.min_tokens = integer("LLAMA_EXPERT_FILL_MIN_TOKENS", 0, 0, 2147483647);
        c.max_tokens = integer("LLAMA_EXPERT_FILL_MAX_TOKENS", 0, 0, 2147483647);
        c.diag = integer("LLAMA_EXPERT_FILL_DIAG", 0, 0, 1);
        c.populate = integer("LLAMA_EXPERT_FILL_POPULATE", 0, 0, 1);
        if (const char * csv = getenv("LLAMA_EXPERT_FILL_DIAG_CSV")) {
            if (!*csv) { invalid("LLAMA_EXPERT_FILL_DIAG_CSV", csv, "nonempty path"); }
            c.diag_csv = csv;
        }
        c.pregate = integer("LLAMA_EXPERT_PREGATE", 0, 0, 1);
        c.k = integer("LLAMA_EXPERT_PREGATE_K", 8, 1, 32);
        c.order = choice("LLAMA_EXPERT_PREGATE_ORDER", "after", {"after", "before"});
        c.max_inflight = integer("LLAMA_EXPERT_PREGATE_MAX_INFLIGHT", 4, 1, 64);
        c.lookahead = integer("LLAMA_EXPERT_PREGATE_LOOKAHEAD", 1, 1, 2);
        c.prefetch = integer("LLAMA_EXPERT_PREFETCH", 0, 0, 1);
        c.prefetch_next = integer("LLAMA_EXPERT_PREFETCH_NEXT", 0, 0, 1);
        return c;
    }
    void print() const {
        fprintf(stderr, "expert_config: effective fill=%zu random=%s max_tokens=%zu threads=%zu chunk_kib=%zu "
            "pregate=%zu k=%zu order=%s max_inflight=%zu lookahead=%zu min_tokens=%zu diag=%zu populate=%zu "
            "diag_csv=%s prefetch=%zu prefetch_next=%zu\n",
            fill, random.c_str(), max_tokens, threads, chunk_kib, pregate, k, order.c_str(), max_inflight,
            lookahead, min_tokens, diag, populate, diag_csv.empty() ? "(unset)" : diag_csv.c_str(), prefetch, prefetch_next);
    }
};

class ExpertFiller {
public:
    struct Range { uint64_t offset; size_t len; };
    struct Counters {
        uint64_t populate_calls = 0, populate_ns = 0, populate_errors = 0;
        uint64_t thread_majflt = 0, thread_minflt = 0;
        uint64_t calls = 0, ranges = 0, bytes_requested = 0;
        uint64_t bytes_skipped_resident = 0, bytes_read = 0, read_syscalls = 0;
        uint64_t total_fill_wall_ns = 0, max_single_fill_ns = 0;
        uint64_t prompt_calls = 0, prompt_bytes_read = 0, prompt_fill_wall_ns = 0;
        uint64_t decode_calls = 0, decode_bytes_read = 0, decode_fill_wall_ns = 0;
    };
    struct AsyncBatch {
        std::atomic<uint64_t> pending {0}, running {0}, bytes_read {0}, read_syscalls {0}, errors {0};
        std::atomic<uint64_t> cancelled_jobs {0}, cancelled_bytes {0};
    };
    using Async = std::shared_ptr<AsyncBatch>;
    struct AsyncCounters {
        uint64_t bytes_read, read_syscalls, errors, dropped, cancelled_jobs, cancelled_bytes;
        uint64_t max_inflight;
    };
    struct AsyncState {
        uint64_t pending = 0, running = 0, cancelled = 0, errors = 0;
        AsyncState & operator+=(const AsyncState & b) {
            pending += b.pending; running += b.running; cancelled += b.cancelled; errors += b.errors;
            return *this;
        }
    };
    AsyncState async_state(const Async & batch) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!batch) { return {0, 0, 1, 0}; } // rejected/unstarted, never resident
        return {batch->pending.load(), batch->running.load(), batch->cancelled_jobs.load(), batch->errors.load()};
    }
    enum class Arrival { done, inflight, queued, resident };
    // A prediction covers every tensor/file/chunk for one expert. Conservatively
    // classify queued if ANY chunk missed dispatch; errors never count as done.
    static Arrival classify(bool resident_at_predict, const AsyncState & state) {
        if (resident_at_predict) { return Arrival::resident; }
        if (state.cancelled || state.pending > state.running || state.errors) { return Arrival::queued; }
        if (state.running) { return Arrival::inflight; }
        return Arrival::done;
    }
    struct Timeliness {
        uint64_t pred_done=0, pred_inflight=0, pred_queued=0, pred_resident_at_predict=0;
        uint64_t hit_done=0, hit_inflight=0, hit_queued=0, hit_resident=0;
        void record(Arrival state, bool hit) {
            switch (state) {
                case Arrival::done: ++pred_done; hit_done += hit; break;
                case Arrival::inflight: ++pred_inflight; hit_inflight += hit; break;
                case Arrival::queued: ++pred_queued; hit_queued += hit; break;
                case Arrival::resident: ++pred_resident_at_predict; hit_resident += hit; break;
            }
        }
    };
    struct Prediction {
        int32_t id = -1;
        bool resident_at_predict = true;
        std::vector<std::pair<ExpertFiller *, Async>> batches;
        // The oracle is mincore in production, a deterministic fake in tests.
        template<class ResidencyOracle>
        void enqueue(ExpertFiller & filler, const std::vector<Range> & ranges, size_t layer, ResidencyOracle resident, size_t distance = 1) {
            if (resident(ranges)) { return; }
            resident_at_predict = false;
            batches.emplace_back(&filler, filler.enqueue_async(ranges, layer, distance));
        }
        AsyncState state() const {
            AsyncState sum;
            for (const auto & b : batches) { sum += b.first->async_state(b.second); }
            return sum;
        }
        Arrival arrive() const { return classify(resident_at_predict, state()); }
    };
    // mapping must be a live, offset-zero file mmap for the lifetime of this object.
    ExpertFiller(const std::string & path, void * mapping, size_t mapping_size,
                 const ExpertConfig & config = ExpertConfig::read())
        : mapping_(static_cast<char *>(mapping)), size_(mapping_size),
          threads_(config.threads), chunk_(config.chunk_kib * 1024) {
        async_limit_ = std::min(threads_, config.max_inflight);
        diag_ = config.diag;
        populate_ = config.populate;
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || !mapping_ || reinterpret_cast<uintptr_t>(mapping_) % page || !size_) {
            throw std::runtime_error("invalid expert mmap");
        }
        page_ = size_t(page);
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) { throw std::runtime_error("open expert file failed"); }
        struct stat st {};
        if (fstat(fd_, &st) || st.st_size < 0 || uint64_t(st.st_size) != size_) {
            close(fd_); fd_ = -1;
            throw std::runtime_error("expert file size differs from mmap");
        }
        try {
            // Allocate before starting workers, so allocation failures cannot kill a worker.
            scratch_.resize(threads_); residency_.resize(threads_);
            for (auto & b : scratch_) { b.resize(std::max(page_, chunk_)); }
            for (auto & b : residency_) { b.resize((std::max(page_, chunk_) + page_ - 1) / page_); }
            for (size_t i = 0; i < threads_; ++i) { workers_.emplace_back([this, i] { worker(i); }); }
        } catch (...) {
            stop(); close(fd_); throw;
        }
    }
    ~ExpertFiller() { wait_async(); stop(); if (fd_ >= 0) { close(fd_); } }
    ExpertFiller(const ExpertFiller &) = delete;
    ExpertFiller & operator=(const ExpertFiller &) = delete;
    size_t threads() const { return threads_; }
    size_t chunk_bytes() const { return chunk_; }
    Counters counters() const { std::lock_guard<std::mutex> lock(fill_mutex_); return counters_; }

    // Enqueue only: no residency query, file IO, or wait for worker completion.
    // Jobs use the existing pool, after all available synchronous jobs.
    Async enqueue_async(const std::vector<Range> & ranges, size_t layer = 0, size_t distance = 1) {
        std::vector<Range> jobs;
        for (const auto & r : ranges) {
            if (r.offset > size_ || r.len > size_ - r.offset) { throw std::out_of_range("async range outside mmap"); }
            if (!r.len) { continue; }
            const uint64_t first = r.offset / page_ * page_;
            const uint64_t end = r.offset + r.len;
            const uint64_t last = end + std::min<uint64_t>((page_ - end % page_) % page_, size_ - end);
            const size_t step = std::max(page_, chunk_ / page_ * page_);
            for (uint64_t at = first; at < last;) {
                const size_t len = std::min<uint64_t>(step, last - at);
                jobs.push_back({at, len}); at += len;
            }
        }
        auto batch = std::make_shared<AsyncBatch>();
        // Construct off-lock; splice publishes the whole submission without allocation.
        std::list<AsyncJob> submission;
        for (auto r : jobs) { submission.push_back({r, batch, layer, distance}); }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Bound speculative backlog; drop a whole submission, never block on IO.
            if (async_queue_.size() + jobs.size() > 4096) {
                ++async_dropped_;
                batch->cancelled_jobs = jobs.size();
                for (const auto & r : jobs) { batch->cancelled_bytes += r.len; }
                async_cancelled_jobs_ += jobs.size(); async_cancelled_bytes_ += batch->cancelled_bytes.load();
                return batch;
            }
            // Earliest layer deadline first; at equal deadlines prefer distance 1,
            // then FIFO. New L+1 work precedes old distance-2 work too.
            auto at = std::find_if(async_queue_.begin(), async_queue_.end(),
                [layer, distance](const AsyncJob & job) {
                    return job.layer > layer || (job.layer == layer && job.distance > distance);
                });
            async_queue_.splice(at, submission);
            batch->pending = jobs.size(); async_pending_ += jobs.size();
        }
        ready_.notify_all();
        return batch;
    }
    // Arrival deadline: retire queued chunks for this layer and older layers.
    // The queue mutex makes cancellation atomic with worker dispatch. Running
    // chunks retain their batch and finish normally; pending includes both kinds.
    void cancel_async_through(size_t layer) {
        std::lock_guard<std::mutex> lock(mutex_);
        cancel_async_locked(layer);
    }
    // Sample all observed experts and cancel their queued work under the SAME
    // dispatch lock: a chunk classified queued cannot start before cancellation.
    std::vector<AsyncState> arrive_async_through(size_t layer, const std::vector<Async> & observed) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<AsyncState> states;
        states.reserve(observed.size());
        for (const auto & b : observed) {
            states.push_back(b ? AsyncState {b->pending.load(), b->running.load(), b->cancelled_jobs.load(), b->errors.load()}
                               : AsyncState {0, 0, 1, 0});
        }
        cancel_async_locked(layer);
        return states;
    }
    // Teardown/tests only. The CPU observer never calls this.
    void wait_async() {
        std::unique_lock<std::mutex> lock(mutex_);
        async_done_.wait(lock, [this] { return async_pending_ == 0; });
    }
    AsyncCounters async_counters() const {
        return {async_bytes_.load(), async_syscalls_.load(), async_errors_.load(), async_dropped_.load(),
                async_cancelled_jobs_.load(), async_cancelled_bytes_.load(), async_max_inflight_.load()};
    }

    bool diagnostic() const { return diag_; }
    // Used by prediction-time residency and opt-in diagnostics. Counts page cache
    // residency, not installed PTEs; errors are explicit rather than resident.
    uint64_t nonresident(const std::vector<Range> & ranges) const {
        uint64_t count = 0;
        std::vector<Range> sorted = ranges;
        std::sort(sorted.begin(), sorted.end(), [](const Range & a, const Range & b) { return a.offset < b.offset; });
        uint64_t checked_end = 0;
        for (const auto & r : sorted) {
            if (!r.len) { continue; }
            const size_t first = std::max<uint64_t>(r.offset / page_ * page_, checked_end);
            if (r.offset + r.len <= first) { continue; }
            const size_t len = r.offset + r.len - first;
            checked_end = first + ((len + page_ - 1) / page_) * page_;
            std::vector<unsigned char> v((len + page_ - 1) / page_);
            if (mincore(mapping_ + first, len, v.data())) { throw std::runtime_error("expert residency mincore failed"); }
            for (auto b : v) { count += !(b & 1); }
        }
        return count;
    }
    uint64_t fill(const std::vector<Range> & ranges, bool decode = false) {
        std::lock_guard<std::mutex> serial(fill_mutex_);
        // Reserve scheduling priority before residency queries/job construction,
        // not just after demand jobs are published. In-flight async IO may finish.
        DemandScope demand(*this);
        const auto start = std::chrono::steady_clock::now();
        batch_diag_ = diag_ && decode;
        ++counters_.calls;
        ++(decode ? counters_.decode_calls : counters_.prompt_calls);
        const uint64_t bytes_before = counters_.bytes_read;
        counters_.ranges += ranges.size();
        auto finish = [&] {
            const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            counters_.total_fill_wall_ns += ns;
            (decode ? counters_.decode_fill_wall_ns : counters_.prompt_fill_wall_ns) += ns;
            (decode ? counters_.decode_bytes_read : counters_.prompt_bytes_read) += counters_.bytes_read - bytes_before;
            counters_.max_single_fill_ns = std::max(counters_.max_single_fill_ns, ns);
        };
        try {
            std::vector<Range> aligned;
            for (const auto & r : ranges) {
                if (r.offset > size_ || r.len > size_ - r.offset) { throw std::out_of_range("expert range outside mmap"); }
                if (!r.len) { continue; }
                const uint64_t first = r.offset - r.offset % page_;
                const uint64_t end = r.offset + r.len;
                const size_t pad = (page_ - end % page_) % page_;
                const uint64_t last = end + std::min<uint64_t>(pad, size_ - end);
                aligned.push_back({first, size_t(last - first)});
            }
            std::sort(aligned.begin(), aligned.end(), [](const Range & a, const Range & b) { return a.offset < b.offset; });
            std::vector<Range> merged;
            for (const auto & r : aligned) {
                if (!merged.empty() && r.offset <= merged.back().offset + merged.back().len) {
                    auto & b = merged.back();
                    b.len = size_t(std::max(b.offset + b.len, r.offset + r.len) - b.offset);
                } else { merged.push_back(r); }
            }
            std::vector<Range> jobs;
            for (const auto & r : merged) {
                counters_.bytes_requested += r.len; // page-aligned union, once per fill
                const size_t pages = r.len / page_ + (r.len % page_ != 0);
                std::vector<unsigned char> resident(pages);
                if (mincore(mapping_ + r.offset, r.len, resident.data())) { throw std::runtime_error("expert mincore failed"); }
                size_t run = 0;
                while (run < pages) {
                    if (resident[run] & 1) {
                        counters_.bytes_skipped_resident += std::min(page_, r.len - run * page_);
                        ++run; continue;
                    }
                    const size_t first = run++;
                    while (run < pages && !(resident[run] & 1)) { ++run; }
                    const size_t end = std::min(r.len, run * page_);
                    for (size_t pos = first * page_; pos < end;) {
                        const size_t len = std::min(chunk_, end - pos);
                        jobs.push_back({r.offset + pos, len}); pos += len;
                    }
                }
            }
            {
                std::unique_lock<std::mutex> lock(mutex_);
                populating_ = false;
                jobs_ = std::move(jobs); next_ = 0; pending_ = jobs_.size(); error_ = 0;
                ready_.notify_all();
                done_.wait(lock, [this] { return pending_ == 0; });
                counters_.bytes_read += batch_bytes_; counters_.read_syscalls += batch_syscalls_;
                batch_bytes_ = batch_syscalls_ = 0;
                if (error_) { throw std::runtime_error("expert pread failed: errno=" + std::to_string(error_)); }
                // All preads finish before populate, including ranges already resident.
                if (populate_.load() && !populate_unsupported().load()) {
                    jobs_.clear();
                    for (const auto & r : merged) {
                        for (size_t pos = 0; pos < r.len;) {
                            const size_t populate_chunk = std::max(page_, chunk_ / page_ * page_);
                            const size_t len = std::min(populate_chunk, r.len - pos);
                            jobs_.push_back({r.offset + pos, len}); pos += len;
                        }
                    }
                    populating_ = true; next_ = 0; pending_ = jobs_.size();
                    ready_.notify_all();
                    done_.wait(lock, [this] { return pending_ == 0; });
                    if (error_) { throw std::runtime_error("expert populate failed: errno=" + std::to_string(error_)); }
                }
            }
        } catch (...) { finish(); throw; }
        finish();
        return counters_.bytes_read - bytes_before;
    }
private:
    friend struct ExpertFillerTestAccess;
    struct DemandScope {
        ExpertFiller & owner;
        explicit DemandScope(ExpertFiller & f) : owner(f) {
            std::lock_guard<std::mutex> lock(owner.mutex_); owner.demand_active_ = true;
        }
        ~DemandScope() {
            { std::lock_guard<std::mutex> lock(owner.mutex_); owner.demand_active_ = false; }
            owner.ready_.notify_all();
        }
    };
    static std::atomic<bool> & populate_unsupported() {
        static std::atomic<bool> value {false}; return value;
    }
    // mutex_ must be held, including while taking arrival snapshots.
    void cancel_async_locked(size_t layer) {
        for (auto it = async_queue_.begin(); it != async_queue_.end();) {
            if (it->layer > layer) { ++it; continue; }
            ++it->batch->cancelled_jobs; it->batch->cancelled_bytes += it->range.len;
            ++async_cancelled_jobs_; async_cancelled_bytes_ += it->range.len;
            --it->batch->pending; --async_pending_;
            it = async_queue_.erase(it);
        }
        if (!async_pending_) { async_done_.notify_all(); }
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto & t : workers_) { t.join(); }
    }
    void worker(size_t index) {
#if defined(__linux__)
        char name[16];
        snprintf(name, sizeof(name), "xfill-%zu", index);
        (void) pthread_setname_np(pthread_self(), name);
#endif
        for (;;) {
            Range job {};
            Async async;
            bool diagnostic = false, populating = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || next_ < jobs_.size() ||
                    (!demand_active_ && !async_queue_.empty() && async_running_ < async_limit_); });
                if (stopping_) { return; }
                if (next_ < jobs_.size()) {
                    job = jobs_[next_++]; diagnostic = batch_diag_; populating = populating_;
                } else {
                    job = async_queue_.front().range; async = async_queue_.front().batch;
                    async_queue_.pop_front(); ++async->running;
                    ++async_running_;
                    async_max_inflight_.store(std::max<uint64_t>(async_max_inflight_.load(), async_running_));
                }
            }
            size_t bytes = 0; uint64_t syscalls = 0; int error = 0;
            struct rusage before {}, after {};
            if (diagnostic) { getrusage(RUSAGE_THREAD, &before); }
            uint64_t populate_ns = 0, populate_calls = 0, populate_errors = 0;
            if (populating) {
                if (populate_.load() && !populate_unsupported().load()) {
                    const auto start = std::chrono::steady_clock::now();
                    ++populate_calls;
                    if (madvise(mapping_ + job.offset, job.len, MADV_POPULATE_READ)) {
                        ++populate_errors;
                        if (errno == EINVAL) {
                            populate_.store(false);
                            if (!populate_unsupported().exchange(true)) { fprintf(stderr, "expert_fill: populate unsupported\n"); }
                        } else { error = errno; }
                    }
                    populate_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start).count();
                }
            } else if (async) {
                // Query at consumption time, including when sync filled these pages meanwhile.
                auto & resident = residency_[index];
                if (mincore(mapping_ + job.offset, job.len, resident.data())) { error = errno; }
                size_t pos = 0;
                while (!error && pos < job.len) {
                    if (resident[pos / page_] & 1) { pos = std::min(job.len, pos + page_); continue; }
                    const size_t first = pos;
                    do { pos = std::min(job.len, pos + page_); }
                    while (pos < job.len && !(resident[pos / page_] & 1));
                    size_t at = first;
                    while (at < pos) {
                        ++syscalls;
                        const ssize_t n = pread(fd_, scratch_[index].data() + at, pos - at, off_t(job.offset + at));
                        if (n < 0 && errno == EINTR) { continue; }
                        if (n <= 0) { error = n < 0 ? errno : EIO; break; }
                        at += size_t(n); bytes += size_t(n);
                    }
                }
            } else while (bytes < job.len) {
                ++syscalls;
                const ssize_t n = pread(fd_, scratch_[index].data() + bytes, job.len - bytes, off_t(job.offset + bytes));
                if (n < 0 && errno == EINTR) { continue; }
                if (n <= 0) { error = n < 0 ? errno : EIO; break; }
                bytes += size_t(n);
            }
            if (diagnostic) { getrusage(RUSAGE_THREAD, &after); }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (async) {
                    async->bytes_read += bytes; async->read_syscalls += syscalls;
                    async->errors += error != 0;
                    async_bytes_ += bytes; async_syscalls_ += syscalls; async_errors_ += error != 0;
                    --async->running; --async->pending; --async_running_;
                    ready_.notify_all(); // a capped async slot is now available
                    if (--async_pending_ == 0) { async_done_.notify_all(); }
                    continue;
                }
                counters_.populate_calls += populate_calls; counters_.populate_ns += populate_ns;
                counters_.populate_errors += populate_errors;
                if (diagnostic) {
                    counters_.thread_majflt += after.ru_majflt - before.ru_majflt;
                    counters_.thread_minflt += after.ru_minflt - before.ru_minflt;
                }
                batch_bytes_ += bytes; batch_syscalls_ += syscalls;
                if (error && !error_) { error_ = error; }
                if (--pending_ == 0) { done_.notify_one(); }
            }
        }
    }
    int fd_ = -1;
    char * mapping_;
    size_t size_, threads_, chunk_, page_ = 0;
    mutable std::mutex fill_mutex_;
    std::mutex mutex_;
    std::condition_variable ready_, done_, async_done_;
    struct AsyncJob { Range range; Async batch; size_t layer, distance; };
    std::list<AsyncJob> async_queue_;
    size_t async_pending_ = 0, async_running_ = 0, async_limit_ = 4;
    bool demand_active_ = false;
    std::atomic<uint64_t> async_bytes_ {0}, async_syscalls_ {0}, async_errors_ {0}, async_dropped_ {0};
    std::atomic<uint64_t> async_cancelled_jobs_ {0}, async_cancelled_bytes_ {0}, async_max_inflight_ {0};
    std::vector<std::thread> workers_;
    std::vector<std::vector<char>> scratch_;
    std::vector<std::vector<unsigned char>> residency_;
    std::vector<Range> jobs_;
    size_t next_ = 0, pending_ = 0;
    bool stopping_ = false, diag_ = false, batch_diag_ = false, populating_ = false;
    std::atomic<bool> populate_ {false};
    int error_ = 0;
    uint64_t batch_bytes_ = 0, batch_syscalls_ = 0;
    Counters counters_;
};
