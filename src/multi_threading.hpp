#pragma once
#include "platform.hpp"
#include "image_features.hpp"
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <queue>
#include <any>
#include <memory>
#include <chrono>
#include <string>
#include <unordered_map>
#include <numeric>
#include <cassert>

namespace simd_engine::mt {

using simd_platform::SystemTopology;
using simd_platform::NumaNodeInfo;
using simd_platform::detect_topology;
using simd_platform::set_thread_affinity;
using simd_platform::set_worker_priority;
using simd_platform::cpu_pause;
using simd_platform::hires_now_ns;

const SystemTopology& get_topology();

// ── Chase-Lev work-stealing deque ─────────────────────────────────────────
template<typename T>
class WorkStealingDeque {
    struct Array {
        std::vector<std::atomic<T*>> buf;
        size_t cap;
        explicit Array(size_t c) : buf(c), cap(c) {}
        void put(long i, T* v) noexcept { buf[i & (cap-1)].store(v, std::memory_order_relaxed); }
        T*   get(long i)       noexcept { return buf[i & (cap-1)].load(std::memory_order_relaxed); }
    };
public:
    WorkStealingDeque() { arr_.store(new Array(1024), std::memory_order_relaxed); }
    ~WorkStealingDeque() { delete arr_.load(); }

    void push(std::function<void()> fn) {
        auto* t = new std::function<void()>(std::move(fn));
        long b = bot_.load(std::memory_order_relaxed);
        long top = top_.load(std::memory_order_acquire);
        Array* a = arr_.load(std::memory_order_relaxed);
        if (b - top >= (long)a->cap) { grow(b, top, a); a = arr_.load(std::memory_order_relaxed); }
        a->put(b, t);
        std::atomic_thread_fence(std::memory_order_release);
        bot_.store(b + 1, std::memory_order_relaxed);
    }

    bool pop(std::function<void()>& out) {
        long b = bot_.load(std::memory_order_relaxed) - 1;
        Array* a = arr_.load(std::memory_order_relaxed);
        bot_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        long t = top_.load(std::memory_order_relaxed);
        if (t <= b) {
            auto* task = a->get(b);
            if (t == b) {
                if (!top_.compare_exchange_strong(t, t+1, std::memory_order_seq_cst, std::memory_order_relaxed))
                    { bot_.store(b+1, std::memory_order_relaxed); return false; }
                bot_.store(b+1, std::memory_order_relaxed);
            }
            out = std::move(*task); delete task; return true;
        }
        bot_.store(b+1, std::memory_order_relaxed); return false;
    }

    bool steal(std::function<void()>& out) {
        long t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        long b = bot_.load(std::memory_order_acquire);
        if (t < b) {
            Array* a = arr_.load(std::memory_order_consume);
            auto* task = a->get(t);
            if (!top_.compare_exchange_strong(t, t+1, std::memory_order_seq_cst, std::memory_order_relaxed))
                return false;
            out = std::move(*task); delete task; return true;
        }
        return false;
    }

    long size() const noexcept {
        return std::max(0L, bot_.load(std::memory_order_relaxed) - top_.load(std::memory_order_relaxed));
    }

private:
    void grow(long b, long t, Array* a) {
        auto* na = new Array(a->cap * 2);
        for (long i = t; i < b; ++i) na->put(i, a->get(i));
        arr_.store(na, std::memory_order_release);
    }
    std::atomic<long>   top_{0}, bot_{0};
    std::atomic<Array*> arr_{nullptr};
};

// ── Worker stats ──────────────────────────────────────────────────────────
struct alignas(64) WorkerStats {
    std::atomic<uint64_t> tasks_executed{0};
    std::atomic<uint64_t> tasks_stolen{0};
    std::atomic<uint64_t> steal_failures{0};
    std::atomic<uint64_t> ns_busy{0};
    uint32_t cpu_id = 0, node_id = 0;
};

// ── NUMA-Aware Thread Pool ─────────────────────────────────────────────────
class NumaThreadPool {
public:
    explicit NumaThreadPool(int num_threads = -1, bool pin_cores = true, bool numa_aware = true);
    ~NumaThreadPool();
    NumaThreadPool(const NumaThreadPool&)            = delete;
    NumaThreadPool& operator=(const NumaThreadPool&) = delete;

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using Ret = std::invoke_result_t<F, Args...>;
        auto pkg  = std::make_shared<std::packaged_task<Ret()>>(
            [f = std::forward<F>(f), ...a = std::forward<Args>(args)]() mutable { return f(std::forward<Args>(a)...); });
        std::future<Ret> fut = pkg->get_future();
        enqueue([pkg]{ (*pkg)(); });
        return fut;
    }

    template<typename F>
    void parallel_for(int count, F&& body, int grain = 1) {
        if (count <= 0) return;
        const int nw = num_threads();
        grain = std::max(grain, (count + nw - 1) / nw);
        std::atomic<int> done{0};
        int chunks = 0;
        for (int i = 0; i < count; i += grain, ++chunks) {
            int end = std::min(i + grain, count), lo = i;
            enqueue([lo, end, &body, &done]{ body(lo, end); done.fetch_add(1, std::memory_order_release); });
        }
        while (done.load(std::memory_order_acquire) < chunks) cpu_pause();
    }

    void wait_all();
    int  num_threads() const noexcept { return (int)workers_.size(); }
    bool is_numa_aware() const noexcept { return numa_aware_; }
    std::vector<WorkerStats> snapshot_stats() const;
    std::string topology_report() const;

private:
    struct Worker {
        std::thread thread;
        WorkStealingDeque<std::function<void()>> deque;
        std::unique_ptr<WorkerStats> stats;
        uint32_t cpu_id = 0, node_id = 0;
    };
    void worker_loop(int idx);
    bool try_steal(int thief, std::function<void()>& task);
    void enqueue(std::function<void()> task);

    std::vector<std::unique_ptr<Worker>> workers_;
    std::queue<std::function<void()>>    global_queue_;
    std::mutex                           global_mtx_;
    std::condition_variable              global_cv_;
    std::atomic<bool>                    shutdown_{false};
    std::atomic<int>                     pending_{0};
    std::mutex                           drain_mtx_;
    std::condition_variable              drain_cv_;
    SystemTopology                       topo_;
    bool                                 numa_aware_, pin_cores_;
};

NumaThreadPool& global_pool();

void parallel_rows(const ImageBuffer& src, ImageBuffer& dst,
    std::function<void(const float*, float*, int, int, int)> row_fn, int grain = 8);

void parallel_tiles(const ImageBuffer& src, ImageBuffer& dst,
    std::function<void(int, int, int, int)> tile_fn, int tw = 128, int th = 128);

ImageBuffer numa_scatter_gather(const ImageBuffer& src,
    std::function<ImageBuffer(const ImageBuffer&)> fn);

// ── Task Graph ────────────────────────────────────────────────────────────
struct TaskNode {
    std::function<void()> fn;
    std::vector<size_t>   deps;
    size_t id = 0;
    int preferred_node = -1, priority = 0;
    std::string name;
};

class TaskGraph {
public:
    size_t add_task(std::function<void()> fn, std::string name = "",
                    std::vector<size_t> deps = {}, int numa_node = -1, int prio = 0);
    void execute(int num_threads = -1);
    struct Timing { std::string name; size_t id; double start_ms, end_ms; int thread_id; };
    const std::vector<Timing>& timings() const noexcept { return timings_; }
private:
    std::vector<TaskNode> nodes_;
    std::vector<Timing>   timings_;
    size_t next_id_ = 0;
};

// ── Scoped Profiler ───────────────────────────────────────────────────────
class ScopedTimer {
public:
    explicit ScopedTimer(const char* name) noexcept;
    ~ScopedTimer();
    static std::unordered_map<std::string, double> snapshot();
    static void reset();
private:
    const char* name_;
    int64_t     t0_;
    static std::mutex                             mtx_;
    static std::unordered_map<std::string,double> registry_;
};

#define SIMD_PROFILE(name) ::simd_engine::mt::ScopedTimer _st_##__LINE__(name)

// ── Progress Tracker ──────────────────────────────────────────────────────
class ProgressTracker {
public:
    explicit ProgressTracker(size_t total, const char* label = "") noexcept;
    void   update(size_t delta = 1) noexcept;
    float  fraction() const noexcept;
    double elapsed_ms() const noexcept;
    std::string status() const;
private:
    std::atomic<size_t> done_{0};
    size_t total_;
    const char* label_;
    int64_t start_ns_;
};

// ── Streaming Pipeline ────────────────────────────────────────────────────
template<typename Input, typename Output>
class StreamingPipeline {
public:
    using StageFn = std::function<Output(Input)>;
    explicit StreamingPipeline(std::vector<StageFn> stages, size_t depth = 4);
    void push(Input item);
    bool pop(Output& out, std::chrono::milliseconds timeout = std::chrono::milliseconds(100));
    void shutdown();
    struct Stats { std::vector<double> stage_ms; double throughput_fps; size_t total; };
    Stats get_stats() const;
private:
    std::vector<StageFn>   stages_;
    std::vector<std::thread> threads_;
    std::atomic<bool>      running_{true};
    std::atomic<size_t>    processed_{0};
    struct BoundedQueue {
        std::queue<std::any> q;
        std::mutex m;
        std::condition_variable cv;
        size_t cap = 4;
    };
    std::vector<BoundedQueue> queues_;
};

} // namespace simd_engine::mt
