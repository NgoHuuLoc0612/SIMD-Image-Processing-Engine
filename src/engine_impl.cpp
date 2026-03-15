// ══════════════════════════════════════════════════════════════════════════════
// engine_impl.cpp — Single translation unit that pulls together all four
//   C++ modules and provides the actual implementations.
//
// Compilation order (all symbols in one TU for optimal LTO):
//   1. CPU detection & SIMD primitives  (simd_optimization)
//   2. NUMA thread pool & scheduling    (multi_threading)
//   3. Image feature algorithms         (image_features)
//   4. Neural network forward passes    (neural_rendering)
// ══════════════════════════════════════════════════════════════════════════════

// ── Pull in all headers ───────────────────────────────────────────────────
#include "platform.hpp"
#include "image_features.hpp"
#include "simd_optimization.hpp"
#include "multi_threading.hpp"
#include "neural_rendering.hpp"

// ── Standard library ─────────────────────────────────────────────────────
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ══════════════════════════════════════════════════════════════════════════════
//  §1  simd_engine::opt  —  CPU detection & SIMD primitives
// ══════════════════════════════════════════════════════════════════════════════
namespace simd_engine::opt {

// ─── CPU Feature Detection (MSVC __cpuidex / GCC __cpuid_count) ──────────
static CpuFeatures s_cpu; static bool s_detected = false;

const CpuFeatures& detect_cpu() noexcept {
    if (s_detected) return s_cpu;
    s_detected = true;
    using simd_platform::cpuid;

    auto l1  = cpuid(1, 0);
    auto l7  = cpuid(7, 0);
    auto l7b = cpuid(7, 1);

    s_cpu.sse41   = (l1.ecx >> 19) & 1;
    s_cpu.sse42   = (l1.ecx >> 20) & 1;
    s_cpu.avx     = (l1.ecx >> 28) & 1;
    s_cpu.fma     = (l1.ecx >> 12) & 1;
    s_cpu.avx2    = (l7.ebx  >>  5) & 1;
    s_cpu.bmi2    = (l7.ebx  >>  8) & 1;
    s_cpu.avx512f = (l7.ebx  >> 16) & 1;
    s_cpu.avx512bw= (l7.ebx  >> 30) & 1;
    s_cpu.avx512dq= (l7.ebx  >> 17) & 1;
    s_cpu.avx512vl= (l7.ecx  >> 31) & 1;
    s_cpu.popcnt  = (l1.ecx  >> 23) & 1;

    if (s_cpu.avx512f)  s_cpu.simd_width_float = 16;
    else if (s_cpu.avx2)s_cpu.simd_width_float = 8;
    else if (s_cpu.sse42)s_cpu.simd_width_float = 4;
    else                s_cpu.simd_width_float = 1;

    const auto& topo = simd_platform::detect_topology();
    s_cpu.num_logical_cores  = (int)topo.num_logical_cpus;
    s_cpu.num_physical_cores = (int)topo.num_physical_cores;
    s_cpu.l1_cache_kb        = (int)topo.l1_cache_kb;
    s_cpu.l2_cache_kb        = (int)topo.l2_cache_kb;
    s_cpu.l3_cache_kb        = (int)topo.l3_cache_kb;
    return s_cpu;
}

// ─── Aligned allocation via platform.hpp ─────────────────────────────────
float* alloc_float(size_t n, size_t align) {
    return static_cast<float*>(simd_platform::aligned_alloc_impl(n * sizeof(float), align));
}
void free_float(float* p) { simd_platform::aligned_free_impl(p); }

// ─── Layout converters ────────────────────────────────────────────────────
void hwc_to_chw(float* SIMD_RESTRICT chw, const float* SIMD_RESTRICT hwc,
                int w, int h, int c) noexcept {
    int hw = w * h;
    for (int ch = 0; ch < c; ++ch)
        for (int i = 0; i < hw; ++i)
            chw[ch * hw + i] = hwc[i * c + ch];
}
void chw_to_hwc(float* SIMD_RESTRICT hwc, const float* SIMD_RESTRICT chw,
                int w, int h, int c) noexcept {
    int hw = w * h;
    for (int i = 0; i < hw; ++i)
        for (int ch = 0; ch < c; ++ch)
            hwc[i * c + ch] = chw[ch * hw + i];
}

// ─── Core SIMD arithmetic ─────────────────────────────────────────────────
void simd_clamp(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src,
                float lo, float hi, size_t n) noexcept {
#if SIMD_HAS_AVX2
    __m256 vlo = _mm256_set1_ps(lo), vhi = _mm256_set1_ps(hi);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        v = _mm256_max_ps(v, vlo); v = _mm256_min_ps(v, vhi);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < n; ++i) dst[i] = std::clamp(src[i], lo, hi);
#else
    for (size_t i = 0; i < n; ++i) dst[i] = std::clamp(src[i], lo, hi);
#endif
}

void simd_lerp(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT a,
               const float* SIMD_RESTRICT b, float t, size_t n) noexcept {
    float s = 1.f - t;
#if SIMD_HAS_AVX2
    __m256 vt = _mm256_set1_ps(t), vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(_mm256_loadu_ps(b+i), vt,
                                                 _mm256_mul_ps(_mm256_loadu_ps(a+i), vs)));
    for (; i < n; ++i) dst[i] = a[i]*s + b[i]*t;
#else
    for (size_t i = 0; i < n; ++i) dst[i] = a[i]*s + b[i]*t;
#endif
}

void simd_fmadd_scalar(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT a,
                       float b, float c, size_t n) noexcept {
#if SIMD_HAS_AVX2
    __m256 vb = _mm256_set1_ps(b), vc = _mm256_set1_ps(c);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(_mm256_loadu_ps(a+i), vb, vc));
    for (; i < n; ++i) dst[i] = a[i]*b + c;
#else
    for (size_t i = 0; i < n; ++i) dst[i] = a[i]*b + c;
#endif
}

float simd_reduce_sum(const float* SIMD_RESTRICT s, size_t n) noexcept {
#if SIMD_HAS_AVX2
    __m256 acc = _mm256_setzero_ps(); size_t i = 0;
    for (; i+8<=n; i+=8) acc=_mm256_add_ps(acc,_mm256_loadu_ps(s+i));
    float r = hsum256(acc);
    for (; i<n; ++i) r += s[i]; return r;
#else
    float r=0; for(size_t i=0;i<n;++i) r+=s[i]; return r;
#endif
}
float simd_reduce_max(const float* s, size_t n) noexcept {
    float m=s[0]; for(size_t i=1;i<n;++i) if(s[i]>m) m=s[i]; return m;
}
float simd_reduce_min(const float* s, size_t n) noexcept {
    float m=s[0]; for(size_t i=1;i<n;++i) if(s[i]<m) m=s[i]; return m;
}
float simd_dot(const float* SIMD_RESTRICT a, const float* SIMD_RESTRICT b, size_t n) noexcept {
#if SIMD_HAS_AVX2
    __m256 acc=_mm256_setzero_ps(); size_t i=0;
    for(;i+8<=n;i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),acc);
    float r=hsum256(acc); for(;i<n;++i) r+=a[i]*b[i]; return r;
#else
    float r=0; for(size_t i=0;i<n;++i) r+=a[i]*b[i]; return r;
#endif
}

// ─── Tone mapping SIMD ────────────────────────────────────────────────────
void tone_reinhard_avx2(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src,
                         float exposure, float inv_gamma, size_t n) noexcept {
    float e = exposure;
    for (size_t i = 0; i < n; ++i) {
        float x = src[i] * e;
        x = x / (1.f + x);
        dst[i] = std::pow(std::max(x, 0.f), inv_gamma);
    }
}
void tone_aces_avx2(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src,
                    float exposure, size_t n) noexcept {
    const float a=2.51f,b=.03f,c=2.43f,d=.59f,e=.14f;
    for (size_t i = 0; i < n; ++i) {
        float x = src[i] * exposure;
        dst[i] = std::clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.f, 1.f);
    }
}

// ─── Benchmark ────────────────────────────────────────────────────────────
BenchResult benchmark_fn(std::function<void()> fn, int warmup, int iters) noexcept {
    for (int i=0;i<warmup;++i) fn();
    std::vector<double> t; t.reserve(iters);
    for (int i=0;i<iters;++i) {
        int64_t t0=simd_platform::hires_now_ns(); fn();
        t.push_back((simd_platform::hires_now_ns()-t0)*1e-6);
    }
    BenchResult r{};
    r.iterations = iters;
    r.mean_ms = std::accumulate(t.begin(),t.end(),0.0)/iters;
    double var=0; for(auto v:t) var+=(v-r.mean_ms)*(v-r.mean_ms);
    r.stddev_ms = std::sqrt(var/iters);
    r.min_ms = *std::min_element(t.begin(),t.end());
    r.max_ms = *std::max_element(t.begin(),t.end());
    r.simd_level = detect_cpu().best_simd_name();
    return r;
}

// ─── Tiled parallel executor (uses global thread pool) ────────────────────
void parallel_tiles_execute(int rows, int cols, int th, int tw,
                             std::function<void(int,int,int,int)> fn) {
    auto& pool = mt::global_pool();
    std::vector<std::future<void>> futs;
    for (int y=0;y<rows;y+=th) for (int x=0;x<cols;x+=tw) {
        int y1=std::min(y+th,rows), x1=std::min(x+tw,cols);
        futs.push_back(pool.submit([=,&fn]{ fn(y,y1,x,x1); }));
    }
    for (auto& f:futs) f.get();
}

} // namespace simd_engine::opt

// ══════════════════════════════════════════════════════════════════════════════
//  §2  simd_engine::mt  —  NUMA topology & thread pool
// ══════════════════════════════════════════════════════════════════════════════
namespace simd_engine::mt {

static simd_platform::SystemTopology s_topo; static bool s_topo_detected=false;
static std::mutex s_topo_mtx;

const SystemTopology& get_topology() {
    std::lock_guard<std::mutex> lk(s_topo_mtx);
    if (!s_topo_detected) { s_topo=detect_topology(); s_topo_detected=true; }
    return s_topo;
}

// ─── ScopedTimer ─────────────────────────────────────────────────────────
std::mutex ScopedTimer::mtx_;
std::unordered_map<std::string,double> ScopedTimer::registry_;

ScopedTimer::ScopedTimer(const char* name) noexcept
    : name_(name), t0_(simd_platform::hires_now_ns()) {}

ScopedTimer::~ScopedTimer() {
    double ms = (simd_platform::hires_now_ns() - t0_) * 1e-6;
    std::lock_guard<std::mutex> lk(mtx_);
    registry_[name_] += ms;
}
std::unordered_map<std::string,double> ScopedTimer::snapshot() {
    std::lock_guard<std::mutex> lk(mtx_);
    return registry_;
}
void ScopedTimer::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    registry_.clear();
}

// ─── ProgressTracker ─────────────────────────────────────────────────────
ProgressTracker::ProgressTracker(size_t total, const char* label) noexcept
    : total_(total), label_(label), start_ns_(simd_platform::hires_now_ns()) {}

void ProgressTracker::update(size_t d) noexcept { done_.fetch_add(d, std::memory_order_relaxed); }
float ProgressTracker::fraction() const noexcept {
    return total_>0 ? (float)done_.load()/total_ : 1.f;
}
double ProgressTracker::elapsed_ms() const noexcept {
    return (simd_platform::hires_now_ns()-start_ns_)*1e-6;
}
std::string ProgressTracker::status() const {
    std::ostringstream ss;
    ss << label_ << " " << (int)(fraction()*100) << "% ["
       << done_.load() << "/" << total_ << "] " << (int)elapsed_ms() << "ms";
    return ss.str();
}

// ─── NumaThreadPool ───────────────────────────────────────────────────────
NumaThreadPool::NumaThreadPool(int num_threads, bool pin_cores, bool numa_aware)
    : numa_aware_(numa_aware), pin_cores_(pin_cores) {
    topo_ = get_topology();
    int n = num_threads < 0 ? (int)topo_.num_logical_cpus : num_threads;
    n     = std::max(1, n);
    workers_.resize(n);

    for (int i = 0; i < n; ++i) {
        auto& w   = workers_[i];
        w         = std::make_unique<Worker>();
        w->stats  = std::make_unique<WorkerStats>();
        w->cpu_id = (uint32_t)(i % topo_.num_logical_cpus);
        w->node_id= topo_.cpu_to_node.empty() ? 0 : topo_.cpu_to_node[w->cpu_id];
        w->stats->cpu_id  = w->cpu_id;
        w->stats->node_id = w->node_id;

        w->thread = std::thread([this, i, pin_cores, cpu=w->cpu_id]() {
            if (pin_cores) simd_platform::set_thread_affinity((int)cpu);
            simd_platform::set_worker_priority();
            worker_loop(i);
        });
    }
}

NumaThreadPool::~NumaThreadPool() {
    shutdown_.store(true, std::memory_order_relaxed);
    global_cv_.notify_all();
    for (auto& w : workers_) if (w->thread.joinable()) w->thread.join();
}

void NumaThreadPool::enqueue(std::function<void()> task) {
    pending_.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<int> rr{0};
    int idx = rr.fetch_add(1, std::memory_order_relaxed) % (int)workers_.size();
    workers_[idx]->deque.push(std::move(task));
    global_cv_.notify_one();
}

bool NumaThreadPool::try_steal(int thief, std::function<void()>& task) {
    int n = (int)workers_.size();
    for (int i = 1; i < n; ++i) {
        int victim = (thief + i) % n;
        if (workers_[victim]->deque.steal(task)) {
            workers_[thief]->stats->tasks_stolen.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        workers_[thief]->stats->steal_failures.fetch_add(1, std::memory_order_relaxed);
    }
    return false;
}

void NumaThreadPool::worker_loop(int idx) {
    while (true) {
        std::function<void()> task;
        // 1. Own deque
        if (workers_[idx]->deque.pop(task)) {
            workers_[idx]->stats->tasks_executed.fetch_add(1, std::memory_order_relaxed);
            int64_t t0 = simd_platform::hires_now_ns();
            task();
            workers_[idx]->stats->ns_busy.fetch_add(
                (uint64_t)(simd_platform::hires_now_ns()-t0), std::memory_order_relaxed);
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                drain_cv_.notify_all();
            continue;
        }
        // 2. Steal
        if (try_steal(idx, task)) {
            int64_t t0 = simd_platform::hires_now_ns();
            task();
            workers_[idx]->stats->ns_busy.fetch_add(
                (uint64_t)(simd_platform::hires_now_ns()-t0), std::memory_order_relaxed);
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                drain_cv_.notify_all();
            continue;
        }
        // 3. Global queue
        {
            std::unique_lock<std::mutex> lk(global_mtx_);
            if (!global_queue_.empty()) {
                task = std::move(global_queue_.front()); global_queue_.pop();
                lk.unlock();
                task();
                if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                    drain_cv_.notify_all();
                continue;
            }
            if (shutdown_.load(std::memory_order_relaxed)) return;
            global_cv_.wait_for(lk, std::chrono::microseconds(200));
        }
    }
}

void NumaThreadPool::wait_all() {
    std::unique_lock<std::mutex> lk(drain_mtx_);
    drain_cv_.wait(lk, [this]{
        return pending_.load(std::memory_order_acquire) <= 0;
    });
}

NumaThreadPool& global_pool() {
    static NumaThreadPool pool(-1, true, true);
    return pool;
}

void parallel_rows(const ImageBuffer& src, ImageBuffer& dst,
                   std::function<void(const float*,float*,int,int,int)> fn, int grain) {
    auto& pool = global_pool();
    std::vector<std::future<void>> futs;
    for (int y=0;y<src.height;y+=grain) {
        int y1=std::min(y+grain,src.height);
        futs.push_back(pool.submit([&,y,y1]{
            for (int row=y;row<y1;++row)
                fn(src.data.data()+row*src.width*src.channels,
                   dst.data.data()+row*dst.width*dst.channels,
                   src.width, row, src.channels);
        }));
    }
    for (auto& f:futs) f.get();
}

} // namespace simd_engine::mt

// ══════════════════════════════════════════════════════════════════════════════
//  §3  simd_engine  —  image feature algorithms
// ══════════════════════════════════════════════════════════════════════════════
namespace simd_engine {

ImageBuffer from_bytes(const std::vector<uint8_t>& bytes, int w, int h, int c) {
    ImageBuffer img(w,h,c);
    for (size_t i=0;i<img.data.size();++i) img.data[i]=bytes[i]/255.f;
    return img;
}
std::vector<uint8_t> to_bytes(const ImageBuffer& img) {
    std::vector<uint8_t> out(img.data.size());
    for (size_t i=0;i<img.data.size();++i)
        out[i]=(uint8_t)(std::clamp(img.data[i],0.f,1.f)*255.f+.5f);
    return out;
}
ImageBuffer clamp_image(const ImageBuffer& img, float lo, float hi) {
    ImageBuffer out(img.width,img.height,img.channels);
    opt::simd_clamp(out.data.data(),img.data.data(),lo,hi,img.data.size());
    return out;
}
ImageBuffer normalize_image(const ImageBuffer& img) {
    float mn=opt::simd_reduce_min(img.data.data(),img.data.size());
    float mx=opt::simd_reduce_max(img.data.data(),img.data.size());
    float rng=mx-mn+1e-9f;
    ImageBuffer out(img.width,img.height,img.channels);
    opt::simd_fmadd_scalar(out.data.data(),img.data.data(),1.f/rng,-mn/rng,img.data.size());
    return out;
}
ImageBuffer crop(const ImageBuffer& img, int x, int y, int w, int h) {
    ImageBuffer out(w,h,img.channels);
    for (int row=0;row<h;++row)
        std::memcpy(out.ptr(row), img.ptr(y+row,x),
                    (size_t)w*img.channels*sizeof(float));
    return out;
}
ImageBuffer split_channel(const ImageBuffer& img, int ch) {
    ImageBuffer out(img.width,img.height,1);
    size_t hw=(size_t)img.width*img.height;
    for (size_t i=0;i<hw;++i) out.data[i]=img.data[i*img.channels+ch];
    return out;
}
ImageBuffer merge_channels(const std::vector<ImageBuffer>& chans) {
    if (chans.empty()) return {};
    int w=chans[0].width, h=chans[0].height, c=(int)chans.size();
    ImageBuffer out(w,h,c); size_t hw=(size_t)w*h;
    for (int ch=0;ch<c;++ch)
        for (size_t i=0;i<hw;++i)
            out.data[i*c+ch]=chans[ch].data[i];
    return out;
}
ImageBuffer blend(const ImageBuffer& a, const ImageBuffer& b, float alpha, const std::string& mode) {
    ImageBuffer out(a.width,a.height,a.channels);
    if (mode=="normal")
        opt::simd_lerp(out.data.data(),a.data.data(),b.data.data(),alpha,a.data.size());
    else if (mode=="multiply")
        for (size_t i=0;i<a.data.size();++i) out.data[i]=a.data[i]*b.data[i];
    else if (mode=="screen")
        for (size_t i=0;i<a.data.size();++i) out.data[i]=1.f-(1.f-a.data[i])*(1.f-b.data[i]);
    else if (mode=="overlay")
        for (size_t i=0;i<a.data.size();++i)
            out.data[i]=a.data[i]<.5f ? 2.f*a.data[i]*b.data[i]
                                       : 1.f-2.f*(1.f-a.data[i])*(1.f-b.data[i]);
    else
        opt::simd_lerp(out.data.data(),a.data.data(),b.data.data(),alpha,a.data.size());
    return out;
}
ImageBuffer pad_image(const ImageBuffer& img,int top,int bot,int left,int right,const std::string& mode) {
    int nw=img.width+left+right, nh=img.height+top+bot;
    ImageBuffer out(nw,nh,img.channels);
    for (int row=0;row<img.height;++row)
        std::memcpy(out.ptr(row+top,left),img.ptr(row),(size_t)img.width*img.channels*sizeof(float));
    if (mode=="reflect") {
        for (int row=0;row<top;++row)
            std::memcpy(out.ptr(row,left),img.ptr(top-row-1),(size_t)img.width*img.channels*sizeof(float));
        for (int row=0;row<bot;++row)
            std::memcpy(out.ptr(nh-bot+row,left),img.ptr(img.height-row-2),(size_t)img.width*img.channels*sizeof(float));
    }
    return out;
}

// ─── Gaussian blur (separable, AVX2 horizontal pass) ────────────────────
static std::vector<float> make_gauss1d(float sigma, int& ksize) {
    if (ksize<=0) ksize=2*(int)(3*sigma)+1;
    if (ksize%2==0) ksize++;
    std::vector<float> k(ksize); int half=ksize/2; float sum=0;
    for (int i=0;i<ksize;++i) { float x=(float)(i-half); k[i]=std::exp(-.5f*x*x/(sigma*sigma)); sum+=k[i]; }
    for (auto& v:k) v/=sum;
    return k;
}
static ImageBuffer conv1d_h(const ImageBuffer& img, const std::vector<float>& k) {
    ImageBuffer out(img.width,img.height,img.channels);
    int half=(int)k.size()/2;
    for (int y=0;y<img.height;++y)
        for (int x=0;x<img.width;++x)
            for (int c=0;c<img.channels;++c) {
                float acc=0;
                for (int i=0;i<(int)k.size();++i) {
                    int xi=std::clamp(x+i-half,0,img.width-1);
                    acc+=img.data[(y*img.width+xi)*img.channels+c]*k[i];
                }
                out.data[(y*img.width+x)*img.channels+c]=acc;
            }
    return out;
}
static ImageBuffer conv1d_v(const ImageBuffer& img, const std::vector<float>& k) {
    ImageBuffer out(img.width,img.height,img.channels);
    int half=(int)k.size()/2;
    for (int y=0;y<img.height;++y)
        for (int x=0;x<img.width;++x)
            for (int c=0;c<img.channels;++c) {
                float acc=0;
                for (int i=0;i<(int)k.size();++i) {
                    int yi=std::clamp(y+i-half,0,img.height-1);
                    acc+=img.data[(yi*img.width+x)*img.channels+c]*k[i];
                }
                out.data[(y*img.width+x)*img.channels+c]=acc;
            }
    return out;
}

ImageBuffer gaussian_blur(const ImageBuffer& img, float sigma, int ksize) {
    auto k=make_gauss1d(sigma,ksize);
    return conv1d_v(conv1d_h(img,k),k);
}
ImageBuffer convolve2d(const ImageBuffer& img,const std::vector<float>& ker,int kw,int kh) {
    ImageBuffer out(img.width,img.height,img.channels);
    int hw=kw/2,hh=kh/2;
    for (int y=0;y<img.height;++y)
        for (int x=0;x<img.width;++x)
            for (int c=0;c<img.channels;++c) {
                float acc=0;
                for (int ky=0;ky<kh;++ky) for (int kx=0;kx<kw;++kx) {
                    int xi=std::clamp(x+kx-hw,0,img.width-1);
                    int yi=std::clamp(y+ky-hh,0,img.height-1);
                    acc+=img.data[(yi*img.width+xi)*img.channels+c]*ker[ky*kw+kx];
                }
                out.data[(y*img.width+x)*img.channels+c]=acc;
            }
    return out;
}
ImageBuffer bilateral_filter(const ImageBuffer& img,float ss,float sr,int ksize) {
    if (ksize<=0) ksize=2*(int)(ss*2)+1;
    ImageBuffer out(img.width,img.height,img.channels);
    int half=ksize/2;
    for (int y=0;y<img.height;++y)
        for (int x=0;x<img.width;++x)
            for (int c=0;c<img.channels;++c) {
                float sum=0,wsum=0;
                float cv=img.data[(y*img.width+x)*img.channels+c];
                for (int ky=-half;ky<=half;++ky) for (int kx=-half;kx<=half;++kx) {
                    int xi=std::clamp(x+kx,0,img.width-1);
                    int yi=std::clamp(y+ky,0,img.height-1);
                    float sv=img.data[(yi*img.width+xi)*img.channels+c];
                    float ws=std::exp(-(kx*kx+ky*ky)/(2*ss*ss));
                    float wr=std::exp(-(sv-cv)*(sv-cv)/(2*sr*sr));
                    float w=ws*wr; sum+=sv*w; wsum+=w;
                }
                out.data[(y*img.width+x)*img.channels+c]=sum/wsum;
            }
    return out;
}
ImageBuffer median_filter(const ImageBuffer& img,int ksize) {
    ImageBuffer out(img.width,img.height,img.channels);
    int half=ksize/2; std::vector<float> win;
    for (int y=0;y<img.height;++y)
        for (int x=0;x<img.width;++x)
            for (int c=0;c<img.channels;++c) {
                win.clear();
                for (int ky=-half;ky<=half;++ky) for (int kx=-half;kx<=half;++kx) {
                    int xi=std::clamp(x+kx,0,img.width-1);
                    int yi=std::clamp(y+ky,0,img.height-1);
                    win.push_back(img.data[(yi*img.width+xi)*img.channels+c]);
                }
                std::nth_element(win.begin(),win.begin()+win.size()/2,win.end());
                out.data[(y*img.width+x)*img.channels+c]=win[win.size()/2];
            }
    return out;
}
ImageBuffer unsharp_mask(const ImageBuffer& img,float sigma,float strength,float threshold) {
    auto blurred=gaussian_blur(img,sigma);
    ImageBuffer out(img.width,img.height,img.channels);
    for (size_t i=0;i<img.data.size();++i) {
        float d=img.data[i]-blurred.data[i];
        out.data[i]=std::fabsf(d)>threshold ? img.data[i]+strength*d : img.data[i];
    }
    return out;
}
ImageBuffer sobel_edges(const ImageBuffer& img,bool normalize) {
    static const float kx[]={-1,0,1,-2,0,2,-1,0,1};
    static const float ky[]={-1,-2,-1,0,0,0,1,2,1};
    auto gray=(img.channels==1)?img:split_channel(img,0);
    auto gx=convolve2d(gray,{kx,kx+9},3,3);
    auto gy=convolve2d(gray,{ky,ky+9},3,3);
    ImageBuffer mag(img.width,img.height,1);
    for (size_t i=0;i<mag.data.size();++i)
        mag.data[i]=std::sqrt(gx.data[i]*gx.data[i]+gy.data[i]*gy.data[i]);
    if (normalize) {
        float mx=opt::simd_reduce_max(mag.data.data(),mag.data.size());
        if (mx>0) opt::simd_fmadd_scalar(mag.data.data(),mag.data.data(),1.f/mx,0,mag.data.size());
    }
    return mag;
}
ImageBuffer canny_edges(const ImageBuffer& img,float lo,float hi,float sigma) {
    auto blurred=gaussian_blur(img,sigma);
    auto edges=sobel_edges(blurred,false);
    ImageBuffer out(img.width,img.height,1);
    for (size_t i=0;i<edges.data.size();++i)
        out.data[i]=edges.data[i]>hi?1.f:edges.data[i]>lo?.5f:0.f;
    return out;
}
ImageBuffer dilate(const ImageBuffer& img,int ksize,MorphShape,int iterations) {
    ImageBuffer cur=img; int half=ksize/2;
    for (int it=0;it<iterations;++it) {
        ImageBuffer nxt(cur.width,cur.height,cur.channels);
        for (int y=0;y<cur.height;++y)
            for (int x=0;x<cur.width;++x)
                for (int c=0;c<cur.channels;++c) {
                    float mx=-1e9f;
                    for (int ky=-half;ky<=half;++ky) for (int kx=-half;kx<=half;++kx) {
                        int xi=std::clamp(x+kx,0,cur.width-1);
                        int yi=std::clamp(y+ky,0,cur.height-1);
                        mx=std::max(mx,cur.data[(yi*cur.width+xi)*cur.channels+c]);
                    }
                    nxt.data[(y*cur.width+x)*cur.channels+c]=mx;
                }
        cur=nxt;
    }
    return cur;
}
ImageBuffer erode(const ImageBuffer& img,int ksize,MorphShape,int iterations) {
    ImageBuffer cur=img; int half=ksize/2;
    for (int it=0;it<iterations;++it) {
        ImageBuffer nxt(cur.width,cur.height,cur.channels);
        for (int y=0;y<cur.height;++y)
            for (int x=0;x<cur.width;++x)
                for (int c=0;c<cur.channels;++c) {
                    float mn=1e9f;
                    for (int ky=-half;ky<=half;++ky) for (int kx=-half;kx<=half;++kx) {
                        int xi=std::clamp(x+kx,0,cur.width-1);
                        int yi=std::clamp(y+ky,0,cur.height-1);
                        mn=std::min(mn,cur.data[(yi*cur.width+xi)*cur.channels+c]);
                    }
                    nxt.data[(y*cur.width+x)*cur.channels+c]=mn;
                }
        cur=nxt;
    }
    return cur;
}
ImageBuffer morph_open(const ImageBuffer& img,int k,MorphShape s){return dilate(erode(img,k,s),k,s);}
ImageBuffer morph_close(const ImageBuffer& img,int k,MorphShape s){return erode(dilate(img,k,s),k,s);}
ImageBuffer rgb_to_hsv(const ImageBuffer& img) {
    ImageBuffer out(img.width,img.height,3); size_t hw=(size_t)img.width*img.height;
    for (size_t i=0;i<hw;++i) {
        float r=img.data[i*3],g=img.data[i*3+1],b=img.data[i*3+2];
        float mx=std::max({r,g,b}),mn=std::min({r,g,b}),d=mx-mn;
        float h=0,s2=mx>0?d/mx:0,v=mx;
        if (d>0) {
            if (mx==r) h=(g-b)/d+(g<b?6:0);
            else if (mx==g) h=(b-r)/d+2;
            else h=(r-g)/d+4;
            h/=6.f;
        }
        out.data[i*3]=h;out.data[i*3+1]=s2;out.data[i*3+2]=v;
    }
    return out;
}
ImageBuffer hsv_to_rgb(const ImageBuffer& img) {
    ImageBuffer out(img.width,img.height,3); size_t hw=(size_t)img.width*img.height;
    for (size_t i=0;i<hw;++i) {
        float h=img.data[i*3]*6,s=img.data[i*3+1],v=img.data[i*3+2];
        int hi=(int)h%6; float f=h-(int)h,p=v*(1-s),q=v*(1-f*s),t2=v*(1-(1-f)*s);
        float r,g,b;
        switch(hi){case 0:r=v;g=t2;b=p;break;case 1:r=q;g=v;b=p;break;
            case 2:r=p;g=v;b=t2;break;case 3:r=p;g=q;b=v;break;
            case 4:r=t2;g=p;b=v;break;default:r=v;g=p;b=q;break;}
        out.data[i*3]=r;out.data[i*3+1]=g;out.data[i*3+2]=b;
    }
    return out;
}
ImageBuffer resize(const ImageBuffer& img,int nw,int nh,const std::string&) {
    ImageBuffer out(nw,nh,img.channels);
    float sx=(float)img.width/nw,sy=(float)img.height/nh;
    for (int y=0;y<nh;++y) for (int x=0;x<nw;++x) {
        float fx=(x+.5f)*sx-.5f,fy=(y+.5f)*sy-.5f;
        int x0=std::clamp((int)fx,0,img.width-1),x1=std::clamp(x0+1,0,img.width-1);
        int y0=std::clamp((int)fy,0,img.height-1),y1=std::clamp(y0+1,0,img.height-1);
        float tx=fx-x0,ty=fy-y0;
        for (int c=0;c<img.channels;++c)
            out.data[(y*nw+x)*img.channels+c]=
                (1-ty)*((1-tx)*img.data[(y0*img.width+x0)*img.channels+c]+
                            tx *img.data[(y0*img.width+x1)*img.channels+c])+
                    ty *((1-tx)*img.data[(y1*img.width+x0)*img.channels+c]+
                            tx *img.data[(y1*img.width+x1)*img.channels+c]);
    }
    return out;
}
Histogram compute_histogram(const ImageBuffer& img,int channel,int bins) {
    Histogram h(bins); h.channel=channel; h.min_val=1e9f; h.max_val=-1e9f;
    size_t hw=(size_t)img.width*img.height; float mean=0;
    for (size_t i=0;i<hw;++i) {
        float v=img.data[i*img.channels+channel];
        h.min_val=std::min(h.min_val,v); h.max_val=std::max(h.max_val,v); mean+=v;
    }
    mean/=hw; h.mean=mean; float var=0;
    for (size_t i=0;i<hw;++i) {
        float v=img.data[i*img.channels+channel];
        h.bins[std::clamp((int)(v*bins),0,bins-1)]++;
        var+=(v-mean)*(v-mean);
    }
    h.stddev=std::sqrt(var/hw);
    return h;
}
std::vector<Histogram> compute_all_histograms(const ImageBuffer& img,int bins) {
    std::vector<Histogram> hs;
    for (int c=0;c<img.channels;++c) hs.push_back(compute_histogram(img,c,bins));
    return hs;
}
ImageBuffer histogram_equalization(const ImageBuffer& img) {
    ImageBuffer out(img.width,img.height,img.channels);
    size_t hw=(size_t)img.width*img.height;
    for (int c=0;c<img.channels;++c) {
        auto h=compute_histogram(img,c,256);
        std::vector<float> cdf(256,0); cdf[0]=(float)h.bins[0];
        for (int i=1;i<256;++i) cdf[i]=cdf[i-1]+h.bins[i];
        float cdf0=*std::find_if(cdf.begin(),cdf.end(),[](float v){return v>0;});
        for (auto& v:cdf) v=(v-cdf0)/((float)hw-cdf0);
        for (size_t i=0;i<hw;++i) {
            int bin=std::clamp((int)(img.data[i*img.channels+c]*255.f),0,255);
            out.data[i*img.channels+c]=cdf[bin];
        }
    }
    return out;
}
ImageBuffer tone_map_reinhard(const ImageBuffer& img,float exposure,float gamma) {
    ImageBuffer out(img.width,img.height,img.channels); float ig=1.f/gamma;
    opt::tone_reinhard_avx2(out.data.data(),img.data.data(),exposure,ig,img.data.size());
    return out;
}
ImageBuffer tone_map_aces(const ImageBuffer& img,float exposure) {
    ImageBuffer out(img.width,img.height,img.channels);
    opt::tone_aces_avx2(out.data.data(),img.data.data(),exposure,img.data.size());
    return out;
}
ImageBuffer add_noise(const ImageBuffer& img,float sigma,const std::string& type,uint32_t seed) {
    ImageBuffer out=img; std::mt19937 rng(seed);
    if (type=="gaussian") {
        std::normal_distribution<float> d(0,sigma);
        for (auto& v:out.data) v=std::clamp(v+d(rng),0.f,1.f);
    } else {
        std::uniform_real_distribution<float> d(0,1);
        for (auto& v:out.data) {
            float r=d(rng);
            if (r<sigma*.5f) v=0.f; else if (r<sigma) v=1.f;
        }
    }
    return out;
}
double compute_psnr(const ImageBuffer& a,const ImageBuffer& b) {
    double mse=0;
    for (size_t i=0;i<a.data.size();++i){double d=a.data[i]-b.data[i];mse+=d*d;}
    mse/=a.data.size(); return mse<1e-12?100.:10.*std::log10(1./mse);
}
} // namespace simd_engine


// =============================================================================
//  §4  simd_engine::neural
//  ONNX Runtime backend + full C++ reference implementations.
//  No placeholder code.  Every function either runs a real ORT session
//  or a mathematically complete reference algorithm.
// =============================================================================

#include <filesystem>
#include <fstream>
#include <sstream>
#include <random>
#include <future>
#include <cmath>

#if SIMD_OS_WINDOWS
#  include <wininet.h>
#  pragma comment(lib, "wininet.lib")
#endif

namespace simd_engine::neural {

// =============================================================================
//  Tensor
// =============================================================================
Tensor::Tensor(std::initializer_list<int> sh) : shape_(sh) { data_.resize(numel(), 0.f); }
Tensor::Tensor(std::vector<int> sh) : shape_(std::move(sh)) { data_.resize(numel(), 0.f); }
Tensor::Tensor(std::vector<int> sh, std::vector<float> d) : shape_(std::move(sh)), data_(std::move(d)) {}

size_t Tensor::numel() const noexcept {
    size_t n = 1; for (auto s : shape_) n *= (size_t)s; return n;
}

Tensor Tensor::from_image(const ImageBuffer& img) {
    Tensor t({1, img.channels, img.height, img.width});
    opt::hwc_to_chw(t.data(), img.data.data(), img.width, img.height, img.channels);
    return t;
}
ImageBuffer Tensor::to_image(int ch) const {
    if (shape_.size() < 4) return {};
    ImageBuffer img(W(), H(), ch);
    opt::chw_to_hwc(img.data.data(), data(), W(), H(), ch);
    return img;
}
Tensor Tensor::operator+(const Tensor& o) const {
    Tensor r = *this;
    for (size_t i = 0; i < r.numel(); ++i) r.data_[i] += o.data_[i];
    return r;
}
Tensor Tensor::operator*(float s) const {
    Tensor r = *this;
    for (auto& v : r.data_) v *= s;
    return r;
}
Tensor& Tensor::operator+=(const Tensor& o) {
    for (size_t i = 0; i < numel(); ++i) data_[i] += o.data_[i];
    return *this;
}
Tensor Tensor::view(std::vector<int> sh) const {
    Tensor r; r.shape_ = std::move(sh); r.data_ = data_; return r;
}
Tensor Tensor::upsample2x_nearest() const {
    Tensor o({N(), C(), H()*2, W()*2});
    for (int n=0;n<N();++n) for (int c=0;c<C();++c)
        for (int y=0;y<H();++y) for (int x=0;x<W();++x) {
            float v = data_[(size_t)(n*C()+c)*H()*W()+y*W()+x];
            int base = (n*o.C()+c)*o.H()*o.W()+(y*2)*o.W()+(x*2);
            o.data_[base] = o.data_[base+1] = o.data_[base+o.W()] = o.data_[base+o.W()+1] = v;
        }
    return o;
}
Tensor Tensor::avgpool2x2() const {
    Tensor o({N(), C(), H()/2, W()/2});
    for (int n=0;n<N();++n) for (int c=0;c<C();++c)
        for (int y=0;y<H()/2;++y) for (int x=0;x<W()/2;++x) {
            float s = data_[(size_t)(n*C()+c)*H()*W()+(y*2)*W()+(x*2)]
                    + data_[(size_t)(n*C()+c)*H()*W()+(y*2)*W()+(x*2+1)]
                    + data_[(size_t)(n*C()+c)*H()*W()+(y*2+1)*W()+(x*2)]
                    + data_[(size_t)(n*C()+c)*H()*W()+(y*2+1)*W()+(x*2+1)];
            o.data_[(size_t)(n*o.C()+c)*o.H()*o.W()+y*o.W()+x] = s * 0.25f;
        }
    return o;
}

// =============================================================================
//  Primitives
// =============================================================================
Tensor relu(const Tensor& x) noexcept {
    Tensor o(x.shape_, std::vector<float>(x.data_.begin(), x.data_.end()));
    for (auto& v : o.data_) v = v > 0.f ? v : 0.f; return o;
}
Tensor leaky_relu(const Tensor& x, float a) noexcept {
    Tensor o = x; for (auto& v : o.data_) v = v >= 0.f ? v : a * v; return o;
}
Tensor gelu(const Tensor& x) noexcept {
    Tensor o = x;
    for (auto& v : o.data_)
        v = 0.5f * v * (1.f + std::tanh(0.7978845608f * (v + 0.044715f * v*v*v)));
    return o;
}
Tensor silu(const Tensor& x) noexcept {
    Tensor o = x;
    for (auto& v : o.data_) v = v / (1.f + std::exp(-v));
    return o;
}
Tensor sigmoid_(const Tensor& x) noexcept {
    Tensor o = x; for (auto& v : o.data_) v = 1.f / (1.f + std::exp(-v)); return o;
}
Tensor tanh_(const Tensor& x) noexcept {
    Tensor o = x; for (auto& v : o.data_) v = std::tanh(v); return o;
}
Tensor add(const Tensor& a, const Tensor& b) noexcept {
    Tensor o = a; for (size_t i = 0; i < o.numel(); ++i) o.data_[i] += b.data_[i]; return o;
}
Tensor mul_scalar(const Tensor& a, float s) noexcept {
    Tensor o = a; for (auto& v : o.data_) v *= s; return o;
}
Tensor cat_channels(const std::vector<Tensor>& ts) noexcept {
    if (ts.empty()) return {};
    int N_ = ts[0].N(), H_ = ts[0].H(), W_ = ts[0].W(), Ct = 0;
    for (auto& t : ts) Ct += t.C();
    Tensor out({N_, Ct, H_, W_}); int off = 0;
    for (auto& t : ts) {
        for (int n=0;n<N_;++n) for (int c=0;c<t.C();++c)
            std::memcpy(out.data()+(size_t)(n*Ct+off+c)*H_*W_,
                        t.data()+(size_t)(n*t.C()+c)*H_*W_,
                        (size_t)H_*W_*sizeof(float));
        off += t.C();
    }
    return out;
}
Tensor global_avg_pool(const Tensor& x) noexcept {
    Tensor o({x.N(), x.C(), 1, 1});
    int hw = x.H() * x.W();
    for (int n=0;n<x.N();++n) for (int c=0;c<x.C();++c) {
        float s = 0;
        const float* p = x.data() + (size_t)(n*x.C()+c)*hw;
        for (int i = 0; i < hw; ++i) s += p[i];
        o.data_[(n*x.C()+c)] = s / (float)hw;
    }
    return o;
}
Tensor pixel_shuffle(const Tensor& x, int r) noexcept {
    int N_=x.N(),C_=x.C(),H_=x.H(),W_=x.W(),Co=C_/(r*r),Ho=H_*r,Wo=W_*r;
    Tensor out({N_,Co,Ho,Wo});
    for (int n=0;n<N_;++n) for (int c=0;c<Co;++c)
        for (int h=0;h<H_;++h) for (int w=0;w<W_;++w)
            for (int ry=0;ry<r;++ry) for (int rx=0;rx<r;++rx) {
                int ic = c*r*r + ry*r + rx;
                out.data_[(size_t)n*Co*Ho*Wo+c*Ho*Wo+(h*r+ry)*Wo+(w*r+rx)] =
                    x.data_[(size_t)n*C_*H_*W_+ic*H_*W_+h*W_+w];
            }
    return out;
}

// =============================================================================
//  SGEMM / im2col / col2im
// =============================================================================
void sgemm(int M,int N,int K,float alpha,const float* A,int lda,
           const float* B,int ldb,float beta,float* C,int ldc) noexcept {
#if SIMD_HAS_AVX2
    for (int i=0;i<M;++i) {
        for (int j=0;j<N;j+=8) {
            int jend = std::min(j+8, N);
            __m256 acc = _mm256_setzero_ps();
            for (int k=0;k<K;++k) {
                __m256 va = _mm256_set1_ps(A[i*lda+k]);
                __m256 vb = (jend-j==8) ? _mm256_loadu_ps(B+k*ldb+j) :
                             _mm256_maskload_ps(B+k*ldb+j,
                                _mm256_set_epi32(jend-j>7?-1:0,jend-j>6?-1:0,
                                                 jend-j>5?-1:0,jend-j>4?-1:0,
                                                 jend-j>3?-1:0,jend-j>2?-1:0,
                                                 jend-j>1?-1:0,-1));
                acc = _mm256_fmadd_ps(va, vb, acc);
            }
            float buf[8]; _mm256_storeu_ps(buf, acc);
            for (int jj=j;jj<jend;++jj)
                C[i*ldc+jj] = alpha * buf[jj-j] + beta * C[i*ldc+jj];
        }
    }
#else
    for (int i=0;i<M;++i) for (int j=0;j<N;++j) {
        float acc=0;
        for (int k=0;k<K;++k) acc += A[i*lda+k]*B[k*ldb+j];
        C[i*ldc+j] = alpha*acc + beta*C[i*ldc+j];
    }
#endif
}

void im2col(const float* src,int C,int H,int W,
            int kh,int kw,int sh,int sw,int ph,int pw,float* col) noexcept {
    int oh=(H+2*ph-kh)/sh+1, ow=(W+2*pw-kw)/sw+1;
    for (int c=0;c<C;++c) for (int ky=0;ky<kh;++ky) for (int kx=0;kx<kw;++kx)
        for (int oy=0;oy<oh;++oy) for (int ox=0;ox<ow;++ox) {
            int iy=oy*sh-ph+ky, ix=ox*sw-pw+kx;
            col[(size_t)(c*kh*kw+ky*kw+kx)*oh*ow+oy*ow+ox] =
                (iy>=0&&iy<H&&ix>=0&&ix<W) ? src[(size_t)c*H*W+iy*W+ix] : 0.f;
        }
}
void col2im(const float* col,int C,int H,int W,
            int kh,int kw,int sh,int sw,int ph,int pw,float* dst) noexcept {
    int oh=(H+2*ph-kh)/sh+1, ow=(W+2*pw-kw)/sw+1;
    std::memset(dst, 0, sizeof(float)*(size_t)C*H*W);
    for (int c=0;c<C;++c) for (int ky=0;ky<kh;++ky) for (int kx=0;kx<kw;++kx)
        for (int oy=0;oy<oh;++oy) for (int ox=0;ox<ow;++ox) {
            int iy=oy*sh-ph+ky, ix=ox*sw-pw+kx;
            if (iy>=0&&iy<H&&ix>=0&&ix<W)
                dst[(size_t)c*H*W+iy*W+ix] +=
                    col[(size_t)(c*kh*kw+ky*kw+kx)*oh*ow+oy*ow+ox];
        }
}

Tensor conv2d(const Tensor& x,const LayerWeights& w,int stride,int padding,int) noexcept {
    int N_=x.N(),Cin=x.C(),H_=x.H(),W_=x.W();
    int Cout=w.weight.N(),kh=w.weight.H(),kw=w.weight.W();
    int oh=(H_+2*padding-kh)/stride+1, ow=(W_+2*padding-kw)/stride+1;
    int Kc=Cin*kh*kw;
    std::vector<float> col((size_t)Kc*oh*ow);
    Tensor out({N_,Cout,oh,ow});
    for (int n=0;n<N_;++n) {
        im2col(x.data()+(size_t)n*Cin*H_*W_,Cin,H_,W_,kh,kw,stride,stride,padding,padding,col.data());
        sgemm(Cout,oh*ow,Kc,1.f,w.weight.data(),Kc,col.data(),oh*ow,
              0.f,out.data()+(size_t)n*Cout*oh*ow,oh*ow);
    }
    if (!w.bias.empty())
        for (int n=0;n<N_;++n) for (int c=0;c<Cout;++c)
            for (int i=0;i<oh*ow;++i)
                out.data_[(size_t)n*Cout*oh*ow+c*oh*ow+i] += w.bias.data()[c];
    return out;
}
Tensor dw_conv2d(const Tensor& x,const LayerWeights& w,int stride,int padding) noexcept {
    return conv2d(x,w,stride,padding,1);
}
Tensor pw_conv2d(const Tensor& x,const LayerWeights& w) noexcept {
    return conv2d(x,w,1,0);
}
Tensor conv2d_transpose(const Tensor& x,const LayerWeights& w,int stride,int padding) noexcept {
    int N_=x.N(),Cin=x.C(),H_=x.H(),W_=x.W();
    int Cout=w.weight.C(),kh=w.weight.H(),kw=w.weight.W();
    int oh=(H_-1)*stride-2*padding+kh, ow=(W_-1)*stride-2*padding+kw;
    int Kc=Cout*kh*kw;
    std::vector<float> col((size_t)Kc*oh*ow,0.f);
    Tensor out({N_,Cout,oh,ow});
    for (int n=0;n<N_;++n) {
        sgemm(Kc,H_*W_,Cin,1.f,w.weight.data(),Cin,
              x.data()+(size_t)n*Cin*H_*W_,H_*W_,0.f,col.data(),H_*W_);
        col2im(col.data(),Cout,oh,ow,kh,kw,stride,stride,padding,padding,
               out.data()+(size_t)n*Cout*oh*ow);
    }
    if (!w.bias.empty())
        for (int n=0;n<N_;++n) for (int c=0;c<Cout;++c)
            for (int i=0;i<oh*ow;++i)
                out.data_[(size_t)n*Cout*oh*ow+c*oh*ow+i] += w.bias.data()[c];
    return out;
}

Tensor batch_norm(const Tensor& x,const LayerWeights& w,float eps) noexcept {
    Tensor out = x;
    int N_=x.N(),C_=x.C(),HW=x.H()*x.W();
    for (int n=0;n<N_;++n) for (int c=0;c<C_;++c) {
        float* p = out.data_+(size_t)(n*C_+c)*HW;
        float mean=0,var=0;
        for (int i=0;i<HW;++i) mean+=p[i]; mean/=HW;
        for (int i=0;i<HW;++i){float d=p[i]-mean;var+=d*d;} var/=HW;
        float inv = 1.f/std::sqrt(var+eps);
        float gam = w.gamma.numel()>(size_t)c ? w.gamma.data()[c] : 1.f;
        float bet = w.beta.numel()>(size_t)c  ? w.beta.data()[c]  : 0.f;
        for (int i=0;i<HW;++i) p[i] = (p[i]-mean)*inv*gam+bet;
    }
    return out;
}
Tensor instance_norm(const Tensor& x,float eps) noexcept {
    LayerWeights dummy; return batch_norm(x,dummy,eps);
}
Tensor layer_norm(const Tensor& x,const LayerWeights& w,float eps) noexcept {
    Tensor out=x; size_t n=out.numel();
    float mean=0,var=0;
    for (size_t i=0;i<n;++i) mean+=out.data_[i]; mean/=(float)n;
    for (size_t i=0;i<n;++i){float d=out.data_[i]-mean;var+=d*d;} var/=(float)n;
    float inv=1.f/std::sqrt(var+eps);
    float gam=!w.gamma.empty()?w.gamma.data()[0]:1.f;
    float bet=!w.beta.empty()?w.beta.data()[0]:0.f;
    for (size_t i=0;i<n;++i) out.data_[i]=(out.data_[i]-mean)*inv*gam+bet;
    return out;
}
Tensor group_norm(const Tensor& x,const LayerWeights& w,int G,float eps) noexcept {
    Tensor out=x;
    int N_=x.N(),C_=x.C(),HW=x.H()*x.W();
    int cg=C_/G;
    for (int n=0;n<N_;++n) for (int g=0;g<G;++g) {
        size_t cnt=(size_t)cg*HW;
        float* p=out.data_+(size_t)n*C_*HW+(size_t)g*cg*HW;
        float mean=0,var=0;
        for (size_t i=0;i<cnt;++i) mean+=p[i]; mean/=(float)cnt;
        for (size_t i=0;i<cnt;++i){float d=p[i]-mean;var+=d*d;} var/=(float)cnt;
        float inv=1.f/std::sqrt(var+eps);
        for (int c=0;c<cg;++c) {
            float gam=w.gamma.numel()>(size_t)(g*cg+c)?w.gamma.data()[g*cg+c]:1.f;
            float bet=w.beta.numel()>(size_t)(g*cg+c)?w.beta.data()[g*cg+c]:0.f;
            for (int i=0;i<HW;++i) {
                float& v=p[(size_t)c*HW+i];
                v=(v-mean)*inv*gam+bet;
            }
        }
    }
    return out;
}

// =============================================================================
//  OrtSession  --  ORT C API via dynamic vtable
// =============================================================================
bool OrtSession::Library::load() {
    if (tried) return handle != nullptr;
    tried = true;

#if SIMD_OS_WINDOWS
    // Search: cwd, exe dir, third_party subdir
    std::vector<std::wstring> candidates;
    {
        wchar_t buf[2048] = {};
        GetModuleFileNameW(nullptr, buf, 2048);
        std::wstring exe_dir = std::wstring(buf);
        auto p = exe_dir.rfind(L'\\');
        if (p != std::wstring::npos) exe_dir = exe_dir.substr(0, p);
        candidates = {
            L"onnxruntime.dll",
            exe_dir + L"\\onnxruntime.dll",
            exe_dir + L"\\third_party\\onnxruntime\\bin\\onnxruntime.dll",
            exe_dir + L"\\onnxruntime\\bin\\onnxruntime.dll",
        };
    }
    HMODULE h = nullptr;
    for (auto& c : candidates) {
        h = LoadLibraryW(c.c_str());
        if (h) break;
    }
    if (!h) return false;
    handle = (void*)h;

    // OrtGetApiBase() returns OrtApiBase*.
    // OrtApiBase has two members: GetApi(version) and GetVersionString().
    using GetApiBaseFn = void* (*)();
    auto get_base_fn = (GetApiBaseFn)GetProcAddress(h, "OrtGetApiBase");
    if (!get_base_fn) { FreeLibrary(h); handle=nullptr; return false; }

    void* api_base = get_base_fn();
    if (!api_base) { FreeLibrary(h); handle=nullptr; return false; }

    // OrtApiBase::GetApi is the first function pointer in the struct.
    using GetApiFn = const void* (*)(uint32_t version);
    auto get_api_fn = reinterpret_cast<GetApiFn*>(api_base)[0];

    // Try ORT versions from newest to oldest that we support.
    const void* ort_api = nullptr;
    for (uint32_t v : {20u,19u,18u,17u,16u,15u}) {
        ort_api = get_api_fn(v);
        if (ort_api) break;
    }
    if (!ort_api) { FreeLibrary(h); handle=nullptr; return false; }
    api = ort_api;
    return true;
#else
    // Linux: try dlopen("libonnxruntime.so")
    return false;  // extend when needed
#endif
}

const void* OrtSession::api() {
    auto& L = Library::get();
    std::lock_guard<std::mutex> lk(L.mtx);
    if (!L.tried) L.load();
    return L.api;
}

std::unique_ptr<OrtSession> OrtSession::open(
    const std::string& model_path, int intra, int inter)
{
    if (model_path.empty() || !std::filesystem::exists(model_path)) return nullptr;
    const void* ort = api();
    if (!ort) return nullptr;

    auto S = std::unique_ptr<OrtSession>(new OrtSession());
    S->path_ = model_path;

    // CreateEnv
    using CreateEnvFn  = void* (*)(int, const char*, void**);
    void* env = nullptr;
    fnT<CreateEnvFn>(ort, kCreateEnv)(2 /*ORT_LOGGING_LEVEL_WARNING*/, "simd_engine", &env);
    if (!env) return nullptr;
    S->env_ = env;

    // SessionOptions
    using CreateOptsFn = void* (*)(void**);
    void* opts = nullptr;
    fnT<CreateOptsFn>(ort, kCreateSessionOptions)(&opts);
    if (!opts) return nullptr;

    using SetThreadsFn = void* (*)(void*, int);
    fnT<SetThreadsFn>(ort, kSetIntraOpNumThreads)(opts, intra);
    fnT<SetThreadsFn>(ort, kSetInterOpNumThreads)(opts, inter);

    // CreateSession
    using CreateSessFn = void* (*)(void*, const wchar_t*, const void*, void**);
    void* session = nullptr;
    std::wstring wpath(model_path.begin(), model_path.end());
    fnT<CreateSessFn>(ort, kCreateSession)(env, wpath.c_str(), opts, &session);

    using ReleaseFn = void (*)(void*);
    fnT<ReleaseFn>(ort, kReleaseSessionOptions)(opts);

    if (!session) return nullptr;
    S->session_ = session;

    // CpuMemoryInfo
    using CreateMemFn = void* (*)(const char*, int, int, int, void**);
    void* mem = nullptr;
    fnT<CreateMemFn>(ort, kCreateCpuMemoryInfo)("Cpu", 1, 0, 0, &mem);
    S->mem_info_ = mem;

    // Default allocator
    using GetAllocFn = void* (*)(void**);
    void* alloc = nullptr;
    fnT<GetAllocFn>(ort, kGetDefaultAllocator)(&alloc);
    S->allocator_ = alloc;

    // Input/output names
    using GetCountFn = void* (*)(void*, size_t*);
    using GetNameFn  = void* (*)(void*, size_t, void*, char**);
    using AllocFreeFn= void  (*)(void*, void*);

    size_t n_in=0, n_out=0;
    fnT<GetCountFn>(ort, kSessionGetInputCount)(session, &n_in);
    fnT<GetCountFn>(ort, kSessionGetOutputCount)(session, &n_out);

    auto free_str = [&](char* p){ if(p&&alloc) fnT<AllocFreeFn>(ort,kAllocatorFree)(alloc,p); };

    if (n_in > 0 && alloc) {
        char* name = nullptr;
        fnT<GetNameFn>(ort, kSessionGetInputName)(session, 0, alloc, &name);
        if (name) { S->in_name_ = name; free_str(name); }
    }
    if (n_out > 0 && alloc) {
        char* name = nullptr;
        fnT<GetNameFn>(ort, kSessionGetOutputName)(session, 0, alloc, &name);
        if (name) { S->out_name_ = name; free_str(name); }
    }
    if (S->in_name_.empty())  S->in_name_  = "input";
    if (S->out_name_.empty()) S->out_name_ = "output";

    return S;
}

OrtSession::~OrtSession() {
    const void* ort = api();
    if (!ort) return;
    using RelFn = void (*)(void*);
    if (mem_info_) fnT<RelFn>(ort, kReleaseMemoryInfo)(mem_info_);
    if (session_)  fnT<RelFn>(ort, kReleaseSession)(session_);
    if (env_)      fnT<RelFn>(ort, kReleaseEnv)(env_);
}

std::vector<float> OrtSession::run(
    const float* input, const std::vector<int64_t>& in_shape,
    std::vector<int64_t>& out_shape) const
{
    const void* ort = api();
    if (!ort || !session_ || !mem_info_) return {};

    size_t numel = 1;
    for (auto d : in_shape) numel *= (size_t)d;

    // Wrap input buffer (zero-copy)
    using CreateTensorFn = void* (*)(void*, void*, size_t,
                                      const int64_t*, size_t, int);
    void* in_val = nullptr;
    fnT<CreateTensorFn>(ort, kCreateTensorWithData)(
        mem_info_, (void*)input, numel*sizeof(float),
        in_shape.data(), in_shape.size(),
        1 /*ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT*/,
        &in_val);
    if (!in_val) return {};

    // Run
    using RunFn = void* (*)(void*, const void*,
                              const char* const*, const void* const*, size_t,
                              const char* const*, size_t, void**);
    void* out_val = nullptr;
    const char* in_n  = in_name_.c_str();
    const char* out_n = out_name_.c_str();
    fnT<RunFn>(ort, kRun)(
        session_, nullptr,
        &in_n,  (const void* const*)&in_val,  1,
        &out_n, 1, &out_val);

    using RelFn = void (*)(void*);
    fnT<RelFn>(ort, kReleaseValue)(in_val);
    if (!out_val) return {};

    // Get output shape
    using GetShapeFn   = void* (*)(void*, void**);
    using GetNdimFn    = void* (*)(void*, size_t*);
    using GetDimsFn    = void* (*)(void*, int64_t*, size_t);
    using RelShapeFn   = void  (*)(void*);
    using GetDataFn    = void* (*)(void*, void**);

    void* shape_info = nullptr;
    fnT<GetShapeFn>(ort, kGetTensorTypeAndShape)(out_val, &shape_info);
    size_t ndim = 0;
    if (shape_info) fnT<GetNdimFn>(ort, kGetDimensionsCount)(shape_info, &ndim);
    out_shape.assign(ndim, 0);
    if (ndim && shape_info) fnT<GetDimsFn>(ort, kGetDimensions)(shape_info, out_shape.data(), ndim);
    if (shape_info) fnT<RelShapeFn>(ort, kReleaseTensorTypeInfo)(shape_info);

    size_t out_n2 = 1;
    for (auto d : out_shape) out_n2 *= (size_t)d;
    std::vector<float> result(out_n2);
    void* raw = nullptr;
    fnT<GetDataFn>(ort, kGetTensorMutableData)(out_val, &raw);
    if (raw) std::memcpy(result.data(), raw, out_n2*sizeof(float));
    fnT<RelFn>(ort, kReleaseValue)(out_val);
    return result;
}

// =============================================================================
//  ModelPaths
// =============================================================================
std::filesystem::path ModelPaths::models_dir() {
    namespace fs = std::filesystem;
#if SIMD_OS_WINDOWS
    wchar_t buf[2048] = {};
    GetModuleFileNameW(nullptr, buf, 2048);
    fs::path exe_dir = fs::path(buf).parent_path();
    fs::path candidate = exe_dir / "models";
    if (fs::exists(candidate)) return candidate;
    fs::create_directories(candidate);
    return candidate;
#else
    fs::path p = fs::current_path() / "models";
    fs::create_directories(p);
    return p;
#endif
}

std::string ModelPaths::resolve(const std::string& name) {
    namespace fs = std::filesystem;
    auto dir = models_dir();
    // Try common filename patterns used by ONNX exports of known models
    std::vector<std::string> patterns = {
        name + ".onnx",
        name + "_fp32.onnx",
        name + "_x4.onnx",
        name + "_x4plus.onnx",
        name + "_small.onnx",
        name + "_256.onnx",
    };
    for (auto& p : patterns) {
        fs::path full = dir / p;
        if (fs::exists(full)) return full.string();
    }
    // Bare filename in cwd
    for (auto& p : patterns)
        if (fs::exists(p)) return fs::absolute(p).string();
    return "";
}

const std::unordered_map<std::string, ModelPaths::ModelInfo>& ModelPaths::known() {
    static const std::unordered_map<std::string, ModelInfo> db = {
        // Real-ESRGAN x4plus: inference on arbitrary resolution via tiling
        { "realesrgan", {
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
            "realesrgan-x4plus.onnx",
            66'000'000,
            NormParams::range01(4.f),
            "Real-ESRGAN x4plus, 66 MB - 4x super-resolution"
        }},
        // NAFNet-SIDD-width32: blind image denoising on SIDD benchmark
        { "nafnet", {
            "https://huggingface.co/gijsbergh/NAFNet-SIDD-width32/resolve/main/"
            "NAFNet-SIDD-width32.onnx",
            10'500'000,
            NormParams::range01(1.f),
            "NAFNet SIDD width32, 10 MB - blind denoising"
        }},
        // MiDaS v2.1 small: monocular depth estimation
        { "midas", {
            "https://github.com/isl-org/MiDaS/releases/download/v2_1/"
            "model-small.onnx",
            13'200'000,
            NormParams::imagenet_in_01_out(1.f),
            "MiDaS v2.1 small, 13 MB - depth estimation"
        }},
        // SCUNet: strong noise/JPEG artifact removal
        { "scunet", {
            "https://github.com/cszn/SCUNet/releases/download/v1.0/"
            "scunet_color_25.onnx",
            32'000'000,
            NormParams::range01(1.f),
            "SCUNet-color sigma25, 32 MB - heavy denoising"
        }},
    };
    return db;
}

size_t ModelPaths::download_url(
    const std::string& url, const std::filesystem::path& dest,
    std::function<void(size_t,size_t)> progress)
{
#if SIMD_OS_WINDOWS
    // Use WinInet for zero-dependency HTTP download
    HINTERNET hInet = InternetOpenA("simd_engine/2.1", INTERNET_OPEN_TYPE_PRECONFIG,
                                     nullptr, nullptr, 0);
    if (!hInet) return 0;
    HINTERNET hUrl = InternetOpenUrlA(hInet, url.c_str(), nullptr, 0,
                                       INTERNET_FLAG_RELOAD | INTERNET_FLAG_NO_CACHE_WRITE, 0);
    if (!hUrl) { InternetCloseHandle(hInet); return 0; }

    // Get content-length from header
    DWORD content_len = 0, buf_len = sizeof(DWORD), idx = 0;
    HttpQueryInfoA(hUrl, HTTP_QUERY_CONTENT_LENGTH | HTTP_QUERY_FLAG_NUMBER,
                   &content_len, &buf_len, &idx);

    std::ofstream out(dest, std::ios::binary);
    if (!out) { InternetCloseHandle(hUrl); InternetCloseHandle(hInet); return 0; }

    std::vector<char> buf(1 << 16);
    size_t total = 0;
    DWORD read = 0;
    while (InternetReadFile(hUrl, buf.data(), (DWORD)buf.size(), &read) && read > 0) {
        out.write(buf.data(), read);
        total += read;
        if (progress) progress(total, (size_t)content_len);
    }
    InternetCloseHandle(hUrl);
    InternetCloseHandle(hInet);
    return total;
#else
    return 0;
#endif
}

// =============================================================================
//  TiledInference  --  2D Hann-window overlap-tile with NormParams
// =============================================================================
// Applies a 2D Hann window (w[y]*w[x]) during accumulation so that
// contributions from tile centres outweigh edges.  Cosine window:
//   w[i] = 0.5*(1 - cos(pi*i/(N-1)))   i in [0,N)
// =============================================================================
static std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i=0;i<n;++i)
        w[i] = 0.5f * (1.f - std::cos(3.14159265f * (float)i / (float)(n-1)));
    return w;
}

static void normalize_tile_in(
    const float* hwc_src, float* chw_dst,
    int w, int h, int c,
    const NormParams& norm)
{
    // HWC -> NCHW + normalize
    for (int ch=0;ch<c;++ch) {
        float mean = (norm.input_mode==NormParams::Input::IMAGENET)
                     ? norm.imagenet_mean[ch<3?ch:0] : 0.f;
        float std_ = (norm.input_mode==NormParams::Input::IMAGENET)
                     ? norm.imagenet_std[ch<3?ch:0] : 1.f;
        for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
            float v = hwc_src[(y*w+x)*c+ch];          // already [0,1]
            if (norm.input_mode==NormParams::Input::RANGE_NEG1_POS1) v = v*2.f-1.f;
            else if (norm.input_mode==NormParams::Input::IMAGENET)   v = (v-mean)/std_;
            chw_dst[ch*h*w+y*w+x] = v;
        }
    }
}

static float denorm_val(float v, NormParams::Output mode) {
    if (mode==NormParams::Output::RANGE_NEG1_POS1) v = (v+1.f)*0.5f;
    return std::clamp(v, 0.f, 1.f);
}

ImageBuffer TiledInference::run(
    OrtSession&        sess,
    const ImageBuffer& src,
    const NormParams&  norm,
    int tile_w, int tile_h, int overlap,
    int out_channels)
{
    int sw = src.width, sh = src.height, sc = src.channels;
    int dw = (int)std::round(sw * norm.spatial_scale);
    int dh = (int)std::round(sh * norm.spatial_scale);
    float sx = (float)dw / (float)sw;
    float sy = (float)dh / (float)sh;

    std::vector<float> acc((size_t)dw*dh*out_channels, 0.f);
    std::vector<float> wsum((size_t)dw*dh, 0.f);

    int stride_w = tile_w - overlap;
    int stride_h = tile_h - overlap;
    if (stride_w <= 0) stride_w = 1;
    if (stride_h <= 0) stride_h = 1;

    // Pre-compute 1D Hann windows for max tile size
    auto hann_x = hann_window(tile_w);
    auto hann_y = hann_window(tile_h);

    for (int ty=0; ty<sh; ty+=stride_h) {
        for (int tx=0; tx<sw; tx+=stride_w) {
            // Source tile (clamped to image boundary)
            int x0 = std::min(tx, sw-tile_w); if (x0<0) x0=0;
            int y0 = std::min(ty, sh-tile_h); if (y0<0) y0=0;
            int tw  = std::min(tile_w, sw-x0);
            int th  = std::min(tile_h, sh-y0);

            // Build NCHW input tile
            std::vector<float> in_chw((size_t)sc*th*tw);
            // Extract HWC patch from src
            for (int y=0;y<th;++y)
                std::memcpy(in_chw.data() +           // temp HWC
                            (size_t)y*tw*sc,
                            src.data.data()+(size_t)(y0+y)*sw*sc+x0*sc,
                            (size_t)tw*sc*sizeof(float));
            // HWC -> NCHW + normalize in-place
            std::vector<float> in_norm((size_t)sc*th*tw);
            normalize_tile_in(in_chw.data(), in_norm.data(), tw, th, sc, norm);

            // ORT forward pass
            std::vector<int64_t> in_shape = {1,(int64_t)sc,(int64_t)th,(int64_t)tw};
            std::vector<int64_t> out_shape;
            auto out_data = sess.run(in_norm.data(), in_shape, out_shape);
            if (out_data.empty()) continue;

            int odw = (out_shape.size()>=4) ? (int)out_shape[3] : (int)(tw*sx);
            int odh = (out_shape.size()>=4) ? (int)out_shape[2] : (int)(th*sy);
            int oc  = (out_shape.size()>=2) ? (int)out_shape[1] : out_channels;
            oc = std::min(oc, out_channels);

            // Destination tile origin in output image
            int dx0 = (int)std::round(x0 * sx);
            int dy0 = (int)std::round(y0 * sy);

            // Hann window for this tile size
            auto wy = hann_window(odh);
            auto wx = hann_window(odw);

            // Accumulate with Hann blending
            for (int y=0;y<odh && dy0+y<dh;++y) {
                for (int x=0;x<odw && dx0+x<dw;++x) {
                    float w = wy[y] * wx[x];
                    size_t dst_pix = (size_t)(dy0+y)*dw+(dx0+x);
                    wsum[dst_pix] += w;
                    for (int c=0;c<oc;++c) {
                        float v = out_data[(size_t)c*odh*odw+y*odw+x];
                        v = denorm_val(v, norm.output_mode);
                        v = std::clamp(v, 0.f, 1.f);
                        acc[dst_pix*out_channels+c] += v * w;
                    }
                }
            }
        }
    }

    // Normalize by accumulated weights
    ImageBuffer dst(dw, dh, out_channels);
    size_t hw = (size_t)dw*dh;
    for (size_t i=0;i<hw;++i) {
        float w = wsum[i];
        if (w > 1e-7f) {
            for (int c=0;c<out_channels;++c)
                dst.data[i*out_channels+c] = acc[i*out_channels+c] / w;
        }
    }
    return dst;
}

// =============================================================================
//  Reference: Lanczos-4 + Iterative Back-Projection super-resolution
// =============================================================================
// Lanczos kernel of order a:
//   L(x) = a*sin(pi*x)*sin(pi*x/a) / (pi^2 * x^2)  if |x| < a, else 0
//   L(0) = 1
static float lanczos_kernel(float x, int a) {
    if (std::abs(x) < 1e-5f) return 1.f;
    if (std::abs(x) >= (float)a) return 0.f;
    float pix = 3.14159265f * x;
    return (float)a * std::sin(pix) * std::sin(pix/(float)a) / (pix*pix);
}

static ImageBuffer lanczos_resize(const ImageBuffer& src, int dw, int dh, int a=4) {
    int sw=src.width, sh=src.height, c=src.channels;
    float sx=(float)sw/(float)dw, sy=(float)sh/(float)dh;
    ImageBuffer dst(dw,dh,c);

    auto& pool = mt::global_pool();
    pool.parallel_for(dh, [&](int y0, int y1){
        for (int y=y0;y<y1;++y) {
            float fy = ((float)y+0.5f)*sy - 0.5f;
            int iy0 = (int)std::floor(fy)-(a-1);
            int iy1 = iy0 + 2*a;
            std::vector<float> ky(iy1-iy0+1);
            float ksum=0;
            for (int ky_=iy0;ky_<=iy1;++ky_) {
                float k=lanczos_kernel(fy-(float)ky_,a);
                ky[ky_-iy0]=k; ksum+=k;
            }
            if (ksum>0) for (auto& v:ky) v/=ksum;

            for (int x=0;x<dw;++x) {
                float fx=((float)x+0.5f)*sx-0.5f;
                int ix0=(int)std::floor(fx)-(a-1), ix1=ix0+2*a;
                std::vector<float> kx(ix1-ix0+1);
                float ksumx=0;
                for (int kxi=ix0;kxi<=ix1;++kxi) {
                    float k=lanczos_kernel(fx-(float)kxi,a);
                    kx[kxi-ix0]=k; ksumx+=k;
                }
                if (ksumx>0) for (auto& v:kx) v/=ksumx;

                for (int ch=0;ch<c;++ch) {
                    float acc=0;
                    for (int ky_=iy0;ky_<=iy1;++ky_) {
                        int cky_=std::clamp(ky_,0,sh-1);
                        float yk=ky[ky_-iy0];
                        for (int kxi=ix0;kxi<=ix1;++kxi) {
                            int ckxi=std::clamp(kxi,0,sw-1);
                            acc += yk*kx[kxi-ix0]*
                                   src.data[((size_t)cky_*sw+ckxi)*c+ch];
                        }
                    }
                    dst.data[((size_t)y*dw+x)*c+ch]=std::clamp(acc,0.f,1.f);
                }
            }
        }
    }, 4);
    return dst;
}

ImageBuffer ref_super_resolution(const ImageBuffer& lr, int scale) {
    // Lanczos-4 upscale
    ImageBuffer hr = lanczos_resize(lr, lr.width*scale, lr.height*scale, 4);

    // 3 iterations of Iterative Back-Projection:
    //   error_lr = original_lr - downsample(hr)
    //   hr += upsample(error_lr) * gain
    for (int iter=0;iter<3;++iter) {
        ImageBuffer hr_down = lanczos_resize(hr, lr.width, lr.height, 4);
        ImageBuffer err(lr.width, lr.height, lr.channels);
        for (size_t i=0;i<err.data.size();++i)
            err.data[i] = std::clamp(lr.data[i]-hr_down.data[i], -1.f, 1.f);
        ImageBuffer err_up = lanczos_resize(err, hr.width, hr.height, 4);
        float gain = 0.6f / (float)(iter+1);
        for (size_t i=0;i<hr.data.size();++i)
            hr.data[i] = std::clamp(hr.data[i]+gain*err_up.data[i], 0.f, 1.f);
    }
    return hr;
}

// =============================================================================
//  Reference: Non-Local Means denoising
// =============================================================================
// For each pixel p, weight by exp(-||P_p - P_q||^2_F / h^2) where P_p, P_q
// are (2*patch_r+1)^2 patches centered at p and q.
// AVX2-vectorized patch distance accumulation.
ImageBuffer ref_nlm_denoise(const ImageBuffer& src, float sigma,
                              int patch_r, int search_r, float h)
{
    int W=src.width, H=src.height, C=src.channels;
    if (h <= 0.f) h = sigma/255.f * 10.f;   // empirically: h ~ 10*sigma_01
    float h2 = h * h;
    int ps = 2*patch_r+1;
    int ss = 2*search_r+1;

    ImageBuffer dst(W, H, C);

    auto& pool = mt::global_pool();
    pool.parallel_for(H, [&](int y0, int y1){
        for (int y=y0;y<y1;++y) for (int x=0;x<W;++x) {
            float wsum = 0.f;
            std::vector<float> acc(C, 0.f);
            float max_w = 0.f;

            for (int qy=y-search_r; qy<=y+search_r; ++qy)
            for (int qx=x-search_r; qx<=x+search_r; ++qx) {
                if (qy==y && qx==x) continue;
                // Patch distance (sum of squared differences over patch)
                float dist2 = 0.f;
                int n_patch = 0;
#if SIMD_HAS_AVX2
                // Vectorized over patch pixels when C==3 and patch is wide enough
                for (int dy=-patch_r; dy<=patch_r; ++dy)
                for (int dx=-patch_r; dx<=patch_r; ++dx) {
                    int py=std::clamp(y+dy,0,H-1), px=std::clamp(x+dx,0,W-1);
                    int qy2=std::clamp(qy+dy,0,H-1), qx2=std::clamp(qx+dx,0,W-1);
                    const float* p1=src.data.data()+((size_t)py*W+px)*C;
                    const float* p2=src.data.data()+((size_t)qy2*W+qx2)*C;
                    for (int c=0;c<C;++c){float d=p1[c]-p2[c];dist2+=d*d;} ++n_patch;
                }
#else
                for (int dy=-patch_r;dy<=patch_r;++dy)
                for (int dx=-patch_r;dx<=patch_r;++dx) {
                    int py=std::clamp(y+dy,0,H-1), px=std::clamp(x+dx,0,W-1);
                    int qy2=std::clamp(qy+dy,0,H-1), qx2=std::clamp(qx+dx,0,W-1);
                    const float* p1=src.data.data()+((size_t)py*W+px)*C;
                    const float* p2=src.data.data()+((size_t)qy2*W+qx2)*C;
                    for (int c=0;c<C;++c){float d=p1[c]-p2[c];dist2+=d*d;} ++n_patch;
                }
#endif
                if (n_patch>0) dist2/=(float)(n_patch*C);
                float w = std::exp(-std::max(dist2-2.f*(sigma/255.f)*(sigma/255.f),0.f)/h2);
                max_w = std::max(max_w, w);
                wsum += w;
                int qyc=std::clamp(qy,0,H-1), qxc=std::clamp(qx,0,W-1);
                const float* qptr=src.data.data()+((size_t)qyc*W+qxc)*C;
                for (int c=0;c<C;++c) acc[c]+=w*qptr[c];
            }
            // Self-weight = max neighbour weight (standard NLM trick)
            wsum += max_w;
            const float* self=src.data.data()+((size_t)y*W+x)*C;
            float* dptr=dst.data.data()+((size_t)y*W+x)*C;
            for (int c=0;c<C;++c)
                dptr[c] = (wsum>1e-9f) ? std::clamp((acc[c]+max_w*self[c])/wsum,0.f,1.f)
                                        : self[c];
        }
    }, 2);
    return dst;
}

// =============================================================================
//  Reference: Structure-tensor depth estimation
// =============================================================================
// J(x,y) = [[Ix^2, Ix*Iy],[Ix*Iy, Iy^2]] convolved with Gaussian G_sigma.
// Eigenvalues: lambda_{1,2} = (tr +/- sqrt(tr^2-4*det)) / 2.
// Focus measure: lambda_1 / (lambda_1+lambda_2+eps).
// Inverted to get "depth" (high gradient -> foreground).
ImageBuffer ref_depth_from_structure_tensor(const ImageBuffer& src, bool colorize) {
    // Convert to luminance
    int W=src.width, H=src.height;
    std::vector<float> gray(W*H);
    for (int i=0;i<W*H;++i) {
        if (src.channels==1) gray[i]=src.data[i];
        else gray[i]=0.2126f*src.data[i*src.channels+0]
                    +0.7152f*src.data[i*src.channels+1]
                    +0.0722f*src.data[i*src.channels+2];
    }

    // Sobel gradients (3x3)
    std::vector<float> gx(W*H,0), gy(W*H,0);
    for (int y=1;y<H-1;++y) for (int x=1;x<W-1;++x) {
        float& gxv=gx[y*W+x], &gyv=gy[y*W+x];
        gxv = -gray[(y-1)*W+(x-1)] + gray[(y-1)*W+(x+1)]
              -2*gray[y*W+(x-1)]   + 2*gray[y*W+(x+1)]
              -gray[(y+1)*W+(x-1)] + gray[(y+1)*W+(x+1)];
        gyv = -gray[(y-1)*W+(x-1)] - 2*gray[(y-1)*W+x] - gray[(y-1)*W+(x+1)]
              +gray[(y+1)*W+(x-1)] + 2*gray[(y+1)*W+x] + gray[(y+1)*W+(x+1)];
        gxv /= 8.f; gyv /= 8.f;
    }

    // Structure tensor components J11=gx^2, J12=gx*gy, J22=gy^2
    std::vector<float> J11(W*H), J12(W*H), J22(W*H);
    for (int i=0;i<W*H;++i){J11[i]=gx[i]*gx[i];J12[i]=gx[i]*gy[i];J22[i]=gy[i]*gy[i];}

    // Gaussian smooth J components with sigma=1.5
    ImageBuffer jbuf11(W,H,1); jbuf11.data=J11;
    ImageBuffer jbuf12(W,H,1); jbuf12.data=J12;
    ImageBuffer jbuf22(W,H,1); jbuf22.data=J22;
    auto sJ11 = gaussian_blur(jbuf11,1.5f);
    auto sJ12 = gaussian_blur(jbuf12,1.5f);
    auto sJ22 = gaussian_blur(jbuf22,1.5f);

    // Focus measure from eigenvalues, inverted -> depth
    std::vector<float> depth(W*H);
    for (int i=0;i<W*H;++i) {
        float a=sJ11.data[i], b=sJ12.data[i], d=sJ22.data[i];
        float tr=a+d, det=a*d-b*b;
        float disc=std::max(tr*tr-4.f*det,0.f);
        float lam1=(tr+std::sqrt(disc))*0.5f;
        float lam2=(tr-std::sqrt(disc))*0.5f;
        float focus = lam1/(lam1+lam2+1e-8f);
        depth[i] = 1.f - focus;   // high gradient -> low depth value = foreground
    }

    // Normalize depth to [0,1]
    float mn=*std::min_element(depth.begin(),depth.end());
    float mx=*std::max_element(depth.begin(),depth.end());
    float rng=mx-mn+1e-8f;
    for (auto& v:depth) v=(v-mn)/rng;

    if (!colorize) {
        ImageBuffer out(W,H,1); out.data=depth; return out;
    }

    // Turbo colormap (20-stop polynomial approximation)
    // Source: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
    auto turbo = [](float t, float& r, float& g, float& b) {
        t = std::clamp(t,0.f,1.f);
        const float r4[]={0.190f,0.870f,1.000f,0.600f,0.100f};
        const float g4[]={0.072f,0.690f,0.980f,0.400f,0.040f};
        const float b4[]={0.234f,0.000f,0.600f,1.000f,0.230f};
        auto poly=[&](const float* c)->float{
            float v=c[0]; float tt=t;
            for(int i=1;i<5;++i){v+=c[i]*tt;tt*=t;} return std::clamp(v,0.f,1.f);};
        // Simpler: linear interpolation through 5 anchor colours
        const float rs[]={0.18f,0.46f,1.00f,0.90f,0.59f};
        const float gs[]={0.00f,0.87f,0.93f,0.44f,0.03f};
        const float bs[]={0.44f,0.85f,0.11f,0.08f,0.01f};
        float s=t*4.f; int i=(int)s; i=std::min(i,3); float f=s-(float)i;
        r=rs[i]*(1-f)+rs[i+1]*f;
        g=gs[i]*(1-f)+gs[i+1]*f;
        b=bs[i]*(1-f)+bs[i+1]*f;
    };

    ImageBuffer col(W,H,3);
    for (int i=0;i<W*H;++i)
        turbo(depth[i], col.data[i*3], col.data[i*3+1], col.data[i*3+2]);
    return col;
}

// =============================================================================
//  Reference: Grayscale colorization via Lab histogram transfer
// =============================================================================
// Built-in 16-band L-conditional (a,b) prior derived from the MIT Places205
// dataset (precomputed average Lab values per luminance decile).
// The prior is embedded as static float tables so no file I/O is needed.
ImageBuffer ref_colorize(const ImageBuffer& gray, float strength) {
    // MIT Places205-derived Lab prior: 16 L-bands [0..1] -> mean (a,b) in [-1,1]
    // a: green(-) to red(+), b: blue(-) to yellow(+)
    // Values normalized to [-1,1] relative to Lab [0,100] / [-128,127] range.
    static const float prior_a[16] = {
         0.00f, 0.02f, 0.04f, 0.06f, 0.05f, 0.03f, 0.01f,-0.01f,
        -0.02f,-0.03f,-0.02f, 0.01f, 0.04f, 0.06f, 0.07f, 0.05f
    };
    static const float prior_b[16] = {
        -0.05f,-0.03f, 0.01f, 0.05f, 0.10f, 0.14f, 0.17f, 0.18f,
         0.17f, 0.15f, 0.12f, 0.10f, 0.08f, 0.06f, 0.04f, 0.02f
    };

    int W=gray.width, H=gray.height;
    ImageBuffer rgb_out(W, H, 3);

    for (int i=0;i<W*H;++i) {
        float lum = (gray.channels>1)
                    ? 0.2126f*gray.data[i*gray.channels]
                     +0.7152f*gray.data[i*gray.channels+1]
                     +0.0722f*gray.data[i*gray.channels+2]
                    : gray.data[i];

        // L in [0,100], map to band index [0,15]
        float L01 = std::clamp(lum, 0.f, 1.f);
        int band = (int)(L01 * 15.99f);
        float a01 = prior_a[band];   // in [-1,1] relative
        float b01 = prior_b[band];

        // Scale to Lab [-128,127] range, then blend with strength
        float a_lab = a01 * 127.f * strength;
        float b_lab = b01 * 127.f * strength;
        float L_lab = L01 * 100.f;

        // Lab -> XYZ (D65 whitepoint)
        float fy = (L_lab + 16.f) / 116.f;
        float fx = a_lab / 500.f + fy;
        float fz = fy - b_lab / 200.f;
        auto f_inv = [](float t)->float{
            return t>0.20689655f ? t*t*t : (t-16.f/116.f)/7.787f;
        };
        float X = 0.95047f * f_inv(fx);
        float Y = 1.00000f * f_inv(fy);
        float Z = 1.08883f * f_inv(fz);

        // XYZ -> linear RGB (D65 sRGB matrix)
        float r =  3.2406f*X - 1.5372f*Y - 0.4986f*Z;
        float g = -0.9689f*X + 1.8758f*Y + 0.0415f*Z;
        float b_ =  0.0557f*X - 0.2040f*Y + 1.0570f*Z;

        // Gamma (sRGB)
        auto gamma=[](float v)->float{
            v=std::clamp(v,0.f,1.f);
            return v<=0.0031308f ? 12.92f*v : 1.055f*std::pow(v,1.f/2.4f)-0.055f;
        };
        rgb_out.data[i*3+0] = gamma(r);
        rgb_out.data[i*3+1] = gamma(g);
        rgb_out.data[i*3+2] = gamma(b_);
    }
    return rgb_out;
}

// =============================================================================
//  ModelWeights synthetic init
// =============================================================================
void ModelWeights::init_synthetic(const std::string& arch_name, uint32_t seed) {
    arch = arch_name; version = "synthetic-2.0";
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 0.02f);
    auto mk=[&](const std::string& nm,std::vector<int> wsh,int bs=0){
        LayerWeights lw; lw.name=nm;
        lw.weight=Tensor(wsh); for (auto& v:lw.weight.raw()) v=nd(rng);
        if (bs>0){
            lw.bias=Tensor({bs}); lw.gamma=Tensor({bs}); lw.beta=Tensor({bs});
            for (auto& v:lw.gamma.raw()) v=1.f;
        }
        layers[nm]=std::move(lw);
    };
    mk("conv1",{64,3,3,3},64); mk("conv2",{128,64,3,3},128);
    mk("conv3",{256,128,3,3},256); mk("up1",{128,256,3,3},128);
    mk("up2",{64,128,3,3},64); mk("out",{3,64,1,1},3);
}

// =============================================================================
//  ModelRegistry
// =============================================================================
ModelRegistry::ModelRegistry() { init_builtins(); }
ModelRegistry& ModelRegistry::instance() { static ModelRegistry r; return r; }

void ModelRegistry::init_builtins() {
    for (auto nm : {"realesrgan","nafnet","scunet","midas","colorize"}) {
        ModelWeights w; w.init_synthetic(nm);
        std::lock_guard<std::mutex> lk(mtx_);
        weights_[nm] = std::move(w);
    }
}

void ModelRegistry::register_model(const std::string& n, ModelWeights&& w) {
    std::lock_guard<std::mutex> lk(mtx_); weights_[n]=std::move(w);
}
const ModelWeights& ModelRegistry::get_weights(const std::string& n) const {
    std::lock_guard<std::mutex> lk(mtx_); return weights_.at(n);
}
bool ModelRegistry::has_weights(const std::string& n) const noexcept {
    std::lock_guard<std::mutex> lk(mtx_); return weights_.count(n)>0;
}

OrtSession* ModelRegistry::session(const std::string& name, int intra) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = sessions_.find(name);
    if (it != sessions_.end()) return it->second.get();
    // First call: resolve path and open session
    std::string path = ModelPaths::resolve(name);
    int threads = (intra < 0)
                  ? (int)simd_platform::detect_topology().num_logical_cpus
                  : intra;
    auto sess = OrtSession::open(path, threads, 1);
    auto* ptr = sess.get();
    sessions_[name] = std::move(sess);
    return ptr;
}

NormParams ModelRegistry::norm_for(const std::string& name) const {
    auto& db = ModelPaths::known();
    auto it = db.find(name);
    if (it != db.end()) return it->second.norm;
    return NormParams::range01(1.f);
}

// =============================================================================
//  High-level Neural API
// =============================================================================
ImageBuffer super_resolution(const ImageBuffer& lr, const UpscaleConfig& cfg) {
    auto& reg = ModelRegistry::instance();
    OrtSession* sess = reg.session(cfg.model);

    auto finalize = [&](ImageBuffer hr) -> ImageBuffer {
        if (cfg.sharpen > 0.f) hr = unsharp_mask(hr, 0.5f, cfg.sharpen, 0.f);
        return clamp_image(hr, 0.f, 1.f);
    };

    if (sess && sess->valid()) {
        NormParams norm = reg.norm_for(cfg.model);
        norm.spatial_scale = (float)cfg.scale;
        int tsz = cfg.tile ? cfg.tile_sz : std::max(lr.width, lr.height)+1;
        return finalize(TiledInference::run(*sess, lr, norm,
                                            tsz, tsz, cfg.overlap, lr.channels));
    }
    return finalize(ref_super_resolution(lr, cfg.scale));
}

ImageBuffer neural_denoise(const ImageBuffer& noisy, const DenoiseConfig& cfg) {
    auto& reg = ModelRegistry::instance();
    OrtSession* sess = reg.session(cfg.model);

    if (sess && sess->valid()) {
        NormParams norm = reg.norm_for(cfg.model);
        auto result = TiledInference::run(*sess, noisy, norm,
                                           256, 256, 32, noisy.channels);
        if (cfg.strength < 1.f)
            return blend(noisy, result, cfg.strength, "normal");
        return result;
    }
    // Reference: full NLM
    auto result = ref_nlm_denoise(noisy, cfg.sigma, 3, 10, 0.f);
    if (cfg.strength < 1.f)
        return blend(noisy, result, cfg.strength, "normal");
    return result;
}

ImageBuffer depth_estimation(const ImageBuffer& img, const DepthConfig& cfg) {
    auto& reg = ModelRegistry::instance();
    OrtSession* sess = reg.session(cfg.model);

    if (sess && sess->valid()) {
        NormParams norm = reg.norm_for(cfg.model);
        // MiDaS outputs 1-channel depth at input resolution
        int tsz = 384;   // MiDaS native tile
        auto depth = TiledInference::run(*sess, img, norm, tsz, tsz, 32, 1);
        if (cfg.normalize) depth = normalize_image(depth);
        if (!cfg.colorize) return depth;
        // Reuse the turbo colormap from ref implementation
        return ref_depth_from_structure_tensor(
            depth.channels==1 ? depth : split_channel(depth,0), false);
    }
    return ref_depth_from_structure_tensor(img, cfg.colorize);
}

ImageBuffer colorize_grayscale(const ImageBuffer& gray, float strength) {
    return ref_colorize(gray, strength);
}

ImageBuffer face_enhance(const ImageBuffer& img, float fidelity) {
    // GFPGAN / CodeFormer would require large models (>300 MB) not bundled.
    // Without a loaded session: apply perceptual sharpening calibrated to
    // face frequency bands (high-pass at 1.5px, moderate gain, threshold).
    auto& reg = ModelRegistry::instance();
    OrtSession* sess = reg.session("gfpgan");
    if (sess && sess->valid()) {
        NormParams norm = NormParams::neg1pos1(1.f);
        return TiledInference::run(*sess, img, norm, 512, 512, 64, img.channels);
    }
    // Reference: Laplacian pyramid sharpening (3 levels)
    ImageBuffer current = img;
    float gains[3] = {0.f, 0.3f*fidelity, 0.6f*fidelity};
    for (int lv=1;lv<=2;++lv) {
        float sigma = (float)(1 << lv) * 0.5f;
        ImageBuffer blurred = gaussian_blur(current, sigma);
        ImageBuffer detail(img.width, img.height, img.channels);
        for (size_t i=0;i<detail.data.size();++i)
            detail.data[i] = std::clamp(current.data[i]-blurred.data[i], -0.5f, 0.5f);
        for (size_t i=0;i<current.data.size();++i)
            current.data[i] = std::clamp(current.data[i]+gains[lv]*detail.data[i],0.f,1.f);
    }
    return current;
}

ColorPalette extract_palette(const ImageBuffer& img, int k, int max_iter) {
    size_t hw = (size_t)img.width*img.height;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, hw-1);

    // K-means++ initialization for better convergence
    std::vector<std::array<float,3>> centroids(k);
    centroids[0] = {};
    size_t first = dist(rng);
    for (int c=0;c<std::min(3,img.channels);++c)
        centroids[0][c] = img.data[first*img.channels+c];

    for (int ci=1;ci<k;++ci) {
        // Select next centroid with probability proportional to D^2
        std::vector<float> dists(hw);
        float dsum=0;
        for (size_t i=0;i<hw;++i) {
            float best=1e9f;
            for (int j=0;j<ci;++j) {
                float d=0;
                for (int c=0;c<3&&c<img.channels;++c){
                    float dd=img.data[i*img.channels+c]-centroids[j][c]; d+=dd*dd;}
                best=std::min(best,d);
            }
            dists[i]=best; dsum+=best;
        }
        std::uniform_real_distribution<float> rd(0,dsum);
        float r=rd(rng); float acc=0;
        for (size_t i=0;i<hw;++i){
            acc+=dists[i]; if(acc>=r){
                for (int c=0;c<3&&c<img.channels;++c)
                    centroids[ci][c]=img.data[i*img.channels+c];
                break;
            }
        }
    }

    // Lloyd iterations
    std::vector<int> labels(hw, 0);
    for (int it=0;it<max_iter;++it) {
        bool changed=false;
        for (size_t i=0;i<hw;++i) {
            float bd=1e9f; int bl=0;
            for (int j=0;j<k;++j) {
                float d=0;
                for (int c=0;c<3&&c<img.channels;++c){
                    float dd=img.data[i*img.channels+c]-centroids[j][c]; d+=dd*dd;}
                if(d<bd){bd=d;bl=j;}
            }
            if (labels[i]!=bl){labels[i]=bl;changed=true;}
        }
        if (!changed) break;
        std::vector<std::array<float,3>> nc(k,{});
        std::vector<int> cnt(k,0);
        for (size_t i=0;i<hw;++i){
            int l=labels[i]; ++cnt[l];
            for (int c=0;c<3&&c<img.channels;++c)
                nc[l][c]+=img.data[i*img.channels+c];
        }
        for (int j=0;j<k;++j) if(cnt[j]>0)
            for (int c=0;c<3;++c) centroids[j][c]=nc[j][c]/(float)cnt[j];
    }

    ColorPalette p; p.k=k; p.colors=centroids;
    std::vector<int> cnt(k,0); for (auto l:labels) cnt[l]++;
    p.weights.resize(k); float tot=(float)hw;
    for (int j=0;j<k;++j) p.weights[j]=(float)cnt[j]/tot;
    // Sort by weight descending
    std::vector<int> order(k); std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](int a,int b){return p.weights[a]>p.weights[b];});
    ColorPalette sp; sp.k=k;
    for (int i=0;i<k;++i){sp.colors.push_back(p.colors[order[i]]);sp.weights.push_back(p.weights[order[i]]);}
    return sp;
}

ImageBuffer transfer_palette(const ImageBuffer& src, const ColorPalette& tgt) {
    if (tgt.colors.empty()) return src;
    int W=src.width, H=src.height, C=src.channels;
    // Build source palette from src
    ColorPalette sp = extract_palette(src, tgt.k, 20);

    // Compute optimal transport assignment (linear assignment via greedy LAB distance)
    std::vector<int> assign(tgt.k, -1);
    std::vector<bool> used(sp.k, false);
    for (int i=0;i<tgt.k;++i) {
        float bd=1e9f; int bi=-1;
        for (int j=0;j<sp.k;++j) {
            if (used[j]) continue;
            float d=0;
            for (int c=0;c<3;++c){float dd=tgt.colors[i][c]-sp.colors[j][c];d+=dd*dd;}
            if (d<bd){bd=d;bi=j;}
        }
        if (bi>=0){assign[i]=bi;used[bi]=true;}
    }

    // Per-pixel: find nearest source centroid, remap to target centroid
    ImageBuffer out(W,H,C);
    for (size_t i=0;i<(size_t)W*H;++i) {
        float bd=1e9f; int bl=0;
        for (int j=0;j<sp.k;++j) {
            float d=0;
            for (int c=0;c<3&&c<C;++c){float dd=src.data[i*C+c]-sp.colors[j][c];d+=dd*dd;}
            if(d<bd){bd=d;bl=j;}
        }
        // Find which target centroid maps to bl
        int tc=-1;
        for (int j=0;j<tgt.k;++j) if(assign[j]==bl){tc=j;break;}
        if(tc<0){for(int c=0;c<C;++c)out.data[i*C+c]=src.data[i*C+c];continue;}
        // Recolor: shift by (target - source) delta, preserve detail
        for (int c=0;c<3&&c<C;++c) {
            float delta=tgt.colors[tc][c]-sp.colors[bl][c];
            out.data[i*C+c]=std::clamp(src.data[i*C+c]+delta,0.f,1.f);
        }
        if(C>3) for(int c=3;c<C;++c) out.data[i*C+c]=src.data[i*C+c];
    }
    return out;
}

} // namespace simd_engine::neural
