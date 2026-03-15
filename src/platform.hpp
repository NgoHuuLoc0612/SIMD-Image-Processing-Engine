#pragma once
// ══════════════════════════════════════════════════════════════════════════════
// platform.hpp — Cross-platform abstraction for Windows (MSVC/Clang-cl/MinGW)
//                and Linux/macOS (GCC/Clang).
//
// Every POSIX-ism, GCC intrinsic, and Linux syscall used elsewhere in the
// engine is shielded here so the rest of the code is #ifdef-free.
// ══════════════════════════════════════════════════════════════════════════════

// ── Compiler / OS detection ────────────────────────────────────────────────
#if defined(_MSC_VER)
#  define SIMD_MSVC   1
#  define SIMD_CLANG  0
#  define SIMD_GCC    0
#elif defined(__clang__)
#  define SIMD_MSVC   0
#  define SIMD_CLANG  1
#  define SIMD_GCC    0
#else
#  define SIMD_MSVC   0
#  define SIMD_CLANG  0
#  define SIMD_GCC    1
#endif

#if defined(_WIN32) || defined(_WIN64)
#  define SIMD_OS_WINDOWS 1
#  define SIMD_OS_LINUX   0
#  define SIMD_OS_MACOS   0
#elif defined(__APPLE__)
#  define SIMD_OS_WINDOWS 0
#  define SIMD_OS_LINUX   0
#  define SIMD_OS_MACOS   1
#else
#  define SIMD_OS_WINDOWS 0
#  define SIMD_OS_LINUX   1
#  define SIMD_OS_MACOS   0
#endif

// ── Windows headers (must come first, minimal surface) ────────────────────
#if SIMD_OS_WINDOWS
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <sysinfoapi.h>   // GetLogicalProcessorInformationEx
#  include <intrin.h>       // __cpuid, __cpuidex, _BitScanForward
#  if _WIN32_WINNT >= 0x0601  // Windows 7+
#    include <processtopologyapi.h>
#  endif
#endif

// ── Standard C++ ──────────────────────────────────────────────────────────
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <cassert>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

// ── SIMD headers ──────────────────────────────────────────────────────────
#if defined(__AVX512F__) || (SIMD_MSVC && defined(_M_AVX512))
#  include <immintrin.h>
#  define SIMD_HAS_AVX512 1
#else
#  define SIMD_HAS_AVX512 0
#endif

#if defined(__AVX2__) || (SIMD_MSVC && defined(_M_AVX2))
#  include <immintrin.h>
#  define SIMD_HAS_AVX2 1
#else
#  define SIMD_HAS_AVX2 0
#endif

#if defined(__AVX__)
#  include <immintrin.h>
#  define SIMD_HAS_AVX 1
#else
#  define SIMD_HAS_AVX 0
#endif

#if defined(__SSE4_2__) || (SIMD_MSVC && defined(_M_IX86_FP))
#  include <nmmintrin.h>
#  define SIMD_HAS_SSE42 1
#else
#  define SIMD_HAS_SSE42 0
#endif

#if defined(__FMA__) || (SIMD_MSVC && defined(__AVX2__))
#  define SIMD_HAS_FMA 1
#else
#  define SIMD_HAS_FMA 0
#endif

// ── Force-inline ──────────────────────────────────────────────────────────
#if SIMD_MSVC
#  define SIMD_FORCEINLINE __forceinline
#  define SIMD_NOINLINE    __declspec(noinline)
#  define SIMD_RESTRICT    __restrict
#  define SIMD_ASSUME_ALIGNED(ptr, align) (ptr)  // MSVC has no equivalent
#else
#  define SIMD_FORCEINLINE __attribute__((always_inline)) inline
#  define SIMD_NOINLINE    __attribute__((noinline))
#  define SIMD_RESTRICT    __restrict__
#  define SIMD_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned(ptr, align)
#endif

// ── Likely / Unlikely ─────────────────────────────────────────────────────
#if SIMD_MSVC
#  define SIMD_LIKELY(x)   (x)
#  define SIMD_UNLIKELY(x) (x)
#else
#  define SIMD_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define SIMD_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// ── Prefetch ──────────────────────────────────────────────────────────────
#if SIMD_MSVC
SIMD_FORCEINLINE void simd_prefetch_r(const void* p) { _mm_prefetch((const char*)p, _MM_HINT_T0); }
SIMD_FORCEINLINE void simd_prefetch_w(const void* p) { _mm_prefetch((const char*)p, _MM_HINT_T0); }
#else
SIMD_FORCEINLINE void simd_prefetch_r(const void* p) { __builtin_prefetch(p, 0, 3); }
SIMD_FORCEINLINE void simd_prefetch_w(const void* p) { __builtin_prefetch(p, 1, 3); }
#endif

// ── CPU-ID ────────────────────────────────────────────────────────────────
namespace simd_platform {

struct CpuidRegs { uint32_t eax, ebx, ecx, edx; };

SIMD_FORCEINLINE CpuidRegs cpuid(uint32_t leaf, uint32_t subleaf = 0) {
    CpuidRegs r{};
#if SIMD_OS_WINDOWS
    int info[4];
    __cpuidex(info, (int)leaf, (int)subleaf);
    r.eax = (uint32_t)info[0]; r.ebx = (uint32_t)info[1];
    r.ecx = (uint32_t)info[2]; r.edx = (uint32_t)info[3];
#elif SIMD_GCC || SIMD_CLANG
#  include <cpuid.h>
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

// ── Aligned memory ────────────────────────────────────────────────────────
SIMD_FORCEINLINE void* aligned_alloc_impl(size_t bytes, size_t alignment = 64) {
#if SIMD_OS_WINDOWS
    return _aligned_malloc(bytes, alignment);
#else
    void* p = nullptr;
    ::posix_memalign(&p, alignment, bytes);
    return p;
#endif
}

SIMD_FORCEINLINE void aligned_free_impl(void* p) {
#if SIMD_OS_WINDOWS
    _aligned_free(p);
#else
    ::free(p);
#endif
}

// ── Thread affinity ───────────────────────────────────────────────────────
// Pin the calling thread to a specific logical processor index.
inline bool set_thread_affinity(int logical_cpu) {
#if SIMD_OS_WINDOWS
    DWORD_PTR mask = (DWORD_PTR)1 << logical_cpu;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#elif SIMD_OS_LINUX
#  if defined(_GNU_SOURCE) || defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(logical_cpu, &cs);
    return pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) == 0;
#  else
    (void)logical_cpu; return false;
#  endif
#else
    (void)logical_cpu; return false;
#endif
}

// Set thread priority to above-normal for worker threads
inline void set_worker_priority() {
#if SIMD_OS_WINDOWS
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
}

// ── High-resolution timer ─────────────────────────────────────────────────
SIMD_FORCEINLINE int64_t hires_now_ns() {
#if SIMD_OS_WINDOWS
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (cnt.QuadPart * 1'000'000'000LL) / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
#endif
}

// ── NUMA topology (Windows) ───────────────────────────────────────────────
struct NumaNodeInfo {
    uint32_t node_id;
    std::vector<uint32_t> processor_ids;   // logical CPU indices in this node
    uint64_t avail_memory_bytes;
    uint64_t total_memory_bytes;
};

struct SystemTopology {
    bool     numa_available    = false;
    uint32_t num_numa_nodes    = 1;
    uint32_t num_physical_cores= 1;
    uint32_t num_logical_cpus  = 1;
    uint32_t l1_cache_kb       = 0;
    uint32_t l2_cache_kb       = 0;
    uint32_t l3_cache_kb       = 0;
    std::vector<NumaNodeInfo> nodes;
    std::vector<uint32_t>     cpu_to_node;   // cpu_index → node_id
};

inline SystemTopology detect_topology() {
    SystemTopology topo;
    topo.num_logical_cpus = (uint32_t)std::thread::hardware_concurrency();

#if SIMD_OS_WINDOWS
    // ── Physical cores, logical CPUs, NUMA via GLPIEX ──────────────
    DWORD buf_len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &buf_len);
    std::vector<uint8_t> buf(buf_len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (GetLogicalProcessorInformationEx(RelationAll, info, &buf_len)) {
        uint8_t* ptr = buf.data();
        while (ptr < buf.data() + buf_len) {
            auto* rec = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
            switch (rec->Relationship) {
            case RelationProcessorCore:
                topo.num_physical_cores++;
                break;
            case RelationNumaNode: {
                NumaNodeInfo ni;
                ni.node_id = rec->NumaNode.NodeNumber;
                // Walk the group affinity to collect CPU ids
                for (WORD g = 0; g < rec->NumaNode.GroupCount; ++g) {
                    auto mask = rec->NumaNode.GroupMasks[g].Mask;
                    WORD  grp  = rec->NumaNode.GroupMasks[g].Group;
                    for (int b = 0; b < 64; ++b) {
                        if (mask & ((KAFFINITY)1 << b)) {
                            uint32_t cpu = grp * 64u + b;
                            ni.processor_ids.push_back(cpu);
                        }
                    }
                }
                ULONGLONG avail = 0;
                GetNumaAvailableMemoryNodeEx((USHORT)ni.node_id, &avail);
                ni.avail_memory_bytes = avail;
                ni.total_memory_bytes = avail; // approximation
                topo.nodes.push_back(std::move(ni));
                break;
            }
            case RelationCache: {
                auto& ci = rec->Cache;
                if (ci.Level == 1 && topo.l1_cache_kb == 0)
                    topo.l1_cache_kb = ci.CacheSize / 1024;
                if (ci.Level == 2 && topo.l2_cache_kb == 0)
                    topo.l2_cache_kb = ci.CacheSize / 1024;
                if (ci.Level == 3 && topo.l3_cache_kb == 0)
                    topo.l3_cache_kb = ci.CacheSize / 1024;
                break;
            }
            default: break;
            }
            ptr += rec->Size;
        }
    }
    topo.numa_available  = topo.nodes.size() > 1;
    topo.num_numa_nodes  = (uint32_t)std::max((size_t)1, topo.nodes.size());

#elif SIMD_OS_LINUX
#  if __has_include(<numa.h>)
    if (numa_available() >= 0) {
        topo.numa_available = true;
        topo.num_numa_nodes = (uint32_t)(numa_max_node() + 1);
        for (uint32_t n = 0; n < topo.num_numa_nodes; ++n) {
            NumaNodeInfo ni; ni.node_id = n;
            bitmask* bm = numa_allocate_cpumask();
            numa_node_to_cpus((int)n, bm);
            for (uint32_t c = 0; c < topo.num_logical_cpus; ++c)
                if (numa_bitmask_isbitset(bm, c)) ni.processor_ids.push_back(c);
            numa_free_cpumask(bm);
            ni.total_memory_bytes = (uint64_t)numa_node_size((int)n, nullptr);
            ni.avail_memory_bytes = ni.total_memory_bytes;
            topo.nodes.push_back(std::move(ni));
        }
    }
#  endif
#endif

    // Build cpu_to_node lookup
    topo.cpu_to_node.resize(topo.num_logical_cpus, 0);
    for (auto& n : topo.nodes)
        for (auto c : n.processor_ids)
            if (c < topo.num_logical_cpus) topo.cpu_to_node[c] = n.node_id;

    if (topo.num_physical_cores == 0)
        topo.num_physical_cores = std::max(1u, topo.num_logical_cpus / 2);
    if (topo.nodes.empty()) {
        NumaNodeInfo ni; ni.node_id = 0;
        for (uint32_t i = 0; i < topo.num_logical_cpus; ++i) ni.processor_ids.push_back(i);
        topo.nodes.push_back(std::move(ni));
    }
    return topo;
}

// ── NUMA-local allocation ─────────────────────────────────────────────────
inline void* numa_alloc_on_node(size_t bytes, uint32_t node_id) {
#if SIMD_OS_WINDOWS
    // VirtualAllocExNuma — preferred NUMA node hint
    void* p = VirtualAllocExNuma(GetCurrentProcess(), nullptr, bytes,
                                  MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE,
                                  (DWORD)node_id);
    return p;
#elif __has_include(<numa.h>)
    return numa_alloc_onnode(bytes, (int)node_id);
#else
    return aligned_alloc_impl(bytes, 64);
#endif
}

inline void numa_free(void* p, size_t bytes) {
#if SIMD_OS_WINDOWS
    VirtualFree(p, 0, MEM_RELEASE);
#elif __has_include(<numa.h>)
    numa_free(p, bytes);
#else
    aligned_free_impl(p);
#endif
}

// ── RDTSC / CPU cycles ────────────────────────────────────────────────────
SIMD_FORCEINLINE uint64_t read_tsc() {
#if SIMD_MSVC
    return __rdtsc();
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}

// ── Pause hint (spin-wait) ────────────────────────────────────────────────
SIMD_FORCEINLINE void cpu_pause() {
#if SIMD_MSVC
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause");
#else
    std::this_thread::yield();
#endif
}

} // namespace simd_platform
