#pragma once
#include "platform.hpp"
#include "image_features.hpp"
#include <functional>
#include <cstring>
#include <numeric>

namespace simd_engine::opt {

struct CpuFeatures {
    bool avx512f=false,avx512bw=false,avx512dq=false,avx512vl=false;
    bool avx2=false,avx=false,sse42=false,sse41=false;
    bool fma=false,bmi2=false,popcnt=false;
    int  simd_width_float=4;
    int  num_logical_cores=1,num_physical_cores=1;
    int  l1_cache_kb=0,l2_cache_kb=0,l3_cache_kb=0;

    // Named best_simd_name() to match implementation
    const char* best_simd_name() const noexcept {
        if(avx512f) return "AVX-512F";
        if(avx2)    return "AVX2+FMA";
        if(avx)     return "AVX";
        if(sse42)   return "SSE4.2";
        return "Scalar";
    }
};

const CpuFeatures& detect_cpu() noexcept;

struct TileConfig {
    int tile_w,tile_h,overlap;
    bool use_l1_blocking;
    int  vector_width;
    static TileConfig auto_tune(int w,int h,int radius,const CpuFeatures& cpu) noexcept {
        TileConfig c;
        c.vector_width = cpu.simd_width_float;
        c.overlap      = radius;
        int l1 = (cpu.l1_cache_kb > 0 ? cpu.l1_cache_kb : 32) * 1024;
        int side = (int)std::sqrt((double)(l1*3/5)/(2*sizeof(float)));
        side = std::max(side & ~(c.vector_width-1), 32);
        c.tile_w = std::min(w, side);
        c.tile_h = std::min(h, side);
        c.use_l1_blocking = (w > c.tile_w || h > c.tile_h);
        return c;
    }
};

struct BenchResult {
    double mean_ms,stddev_ms,min_ms,max_ms,throughput_mpps;
    int iterations;
    const char* simd_level;
};

BenchResult benchmark_fn(std::function<void()> fn, int warmup=3, int iters=10) noexcept;

float* alloc_float(size_t n, size_t alignment=64);
void   free_float(float* p);

void hwc_to_chw(float* SIMD_RESTRICT chw, const float* SIMD_RESTRICT hwc, int w, int h, int c) noexcept;
void chw_to_hwc(float* SIMD_RESTRICT hwc, const float* SIMD_RESTRICT chw, int w, int h, int c) noexcept;

void  simd_clamp      (float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src, float lo, float hi, size_t n) noexcept;
void  simd_lerp       (float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT a,   const float* SIMD_RESTRICT b, float t, size_t n) noexcept;
void  simd_fmadd_scalar(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT a, float b, float c, size_t n) noexcept;
float simd_dot        (const float* SIMD_RESTRICT a, const float* SIMD_RESTRICT b, size_t n) noexcept;
float simd_reduce_sum (const float* SIMD_RESTRICT src, size_t n) noexcept;
float simd_reduce_max (const float* SIMD_RESTRICT src, size_t n) noexcept;
float simd_reduce_min (const float* SIMD_RESTRICT src, size_t n) noexcept;

void separable_filter_avx2(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src,
    const float* kx, int kxlen, const float* ky, int kylen, int w, int h, int ch) noexcept;
void convolve_general(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src,
    const float* kernel, int kw, int kh, int w, int h, int ch) noexcept;

#if SIMD_HAS_AVX2
SIMD_FORCEINLINE __m256 fast_exp_avx2(__m256 x) noexcept {
    x = _mm256_max_ps(x,_mm256_set1_ps(-88.3762626647950f));
    x = _mm256_min_ps(x,_mm256_set1_ps( 88.3762626647950f));
    return _mm256_castsi256_ps(_mm256_cvtps_epi32(
        _mm256_fmadd_ps(_mm256_set1_ps(12102203.f),x,_mm256_set1_ps(1065353216.f))));
}
SIMD_FORCEINLINE __m256 fast_rcp_nr_avx2(__m256 x) noexcept {
    __m256 r=_mm256_rcp_ps(x);
    return _mm256_fnmadd_ps(_mm256_mul_ps(x,r),r,_mm256_add_ps(r,r));
}
SIMD_FORCEINLINE float hsum256(__m256 v) noexcept {
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); lo=_mm_hadd_ps(lo,lo); lo=_mm_hadd_ps(lo,lo);
    return _mm_cvtss_f32(lo);
}
#endif
#if SIMD_HAS_AVX512
SIMD_FORCEINLINE __m512 fast_exp_avx512(__m512 x) noexcept {
    x=_mm512_max_ps(x,_mm512_set1_ps(-88.376f));
    x=_mm512_min_ps(x,_mm512_set1_ps( 88.376f));
    return _mm512_castsi512_ps(_mm512_cvtps_epi32(
        _mm512_fmadd_ps(_mm512_set1_ps(12102203.f),x,_mm512_set1_ps(1065353216.f))));
}
#endif

void histogram_simd   (uint32_t* hist, const float* SIMD_RESTRICT src, size_t n, int bins, float vmin, float vmax) noexcept;
void resize_lanczos_avx2(float* SIMD_RESTRICT dst, int dw, int dh, const float* SIMD_RESTRICT src, int sw, int sh, int ch, int a=3) noexcept;
void resize_bilinear_avx2(float* SIMD_RESTRICT dst, int dw, int dh, const float* SIMD_RESTRICT src, int sw, int sh, int ch) noexcept;
void dilate_1d_simd   (float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src, int length, int ksize) noexcept;
void erode_1d_simd    (float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src, int length, int ksize) noexcept;
void bilateral_avx2   (float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src, int w, int h, int ch, float ss, float sr, int ks) noexcept;
void sobel_magnitude_avx2(float* SIMD_RESTRICT mag, float* SIMD_RESTRICT dir, const float* SIMD_RESTRICT gray, int w, int h) noexcept;
void median_3x3_avx2  (float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src, int w, int h, int ch) noexcept;
void fft_radix2_avx2  (float* SIMD_RESTRICT re, float* SIMD_RESTRICT im, int n, bool inverse) noexcept;
void fft2d_avx2       (float* SIMD_RESTRICT re, float* SIMD_RESTRICT im, int w, int h) noexcept;
void ifft2d_avx2      (float* SIMD_RESTRICT re, float* SIMD_RESTRICT im, int w, int h) noexcept;
void gaussian_box3_avx2(float* SIMD_RESTRICT dst, float* SIMD_RESTRICT tmp, const float* SIMD_RESTRICT src, int w, int h, int ch, float sigma) noexcept;
void rgb_to_lab_avx2  (float* SIMD_RESTRICT lab, const float* SIMD_RESTRICT rgb, size_t npx) noexcept;
void rgb_to_hsv_avx2  (float* SIMD_RESTRICT hsv, const float* SIMD_RESTRICT rgb, size_t npx) noexcept;
void hsv_to_rgb_avx2  (float* SIMD_RESTRICT rgb, const float* SIMD_RESTRICT hsv, size_t npx) noexcept;
void tone_reinhard_avx2(float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src, float exposure, float inv_gamma, size_t n) noexcept;
void tone_aces_avx2   (float* SIMD_RESTRICT dst, const float* SIMD_RESTRICT src, float exposure, size_t n) noexcept;
void parallel_tiles_execute(int rows, int cols, int tile_h, int tile_w, std::function<void(int,int,int,int)> fn);

} // namespace simd_engine::opt
