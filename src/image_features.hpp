#pragma once
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <immintrin.h>
#include <cstring>

namespace simd_engine {

struct ImageBuffer {
    std::vector<float> data;
    int width, height, channels;
    
    ImageBuffer() : width(0), height(0), channels(0) {}
    ImageBuffer(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c, 0.0f) {}
    
    float* ptr(int y = 0, int x = 0, int c = 0) {
        return data.data() + (y * width + x) * channels + c;
    }
    const float* ptr(int y = 0, int x = 0, int c = 0) const {
        return data.data() + (y * width + x) * channels + c;
    }
    size_t total() const { return (size_t)width * height * channels; }
};

struct ProcessingStats {
    double elapsed_ms;
    double throughput_mpps;
    std::string simd_level;
    int threads_used;
    bool numa_aware;
};

// ─── Histogram & Tone Mapping ──────────────────────────────────────────────
struct Histogram {
    std::vector<uint32_t> bins;
    float min_val, max_val, mean, stddev;
    int channel;
    
    Histogram(int num_bins = 256) : bins(num_bins, 0), min_val(0), max_val(0), mean(0), stddev(0), channel(0) {}
};

Histogram compute_histogram(const ImageBuffer& img, int channel, int bins = 256);

std::vector<Histogram> compute_all_histograms(const ImageBuffer& img, int bins = 256);

ImageBuffer histogram_equalization(const ImageBuffer& img);

ImageBuffer clahe(const ImageBuffer& img, int tile_w = 64, int tile_h = 64, float clip_limit = 4.0f);

ImageBuffer tone_map_reinhard(const ImageBuffer& hdr, float exposure = 1.0f, float gamma = 2.2f);

ImageBuffer tone_map_aces(const ImageBuffer& hdr, float exposure = 1.0f);

ImageBuffer tone_map_filmic(const ImageBuffer& hdr, float exposure = 1.0f, float shoulder = 0.22f);

// ─── Convolution & Filtering ───────────────────────────────────────────────
ImageBuffer convolve2d(const ImageBuffer& img, const std::vector<float>& kernel, int kw, int kh);

ImageBuffer convolve_separable(const ImageBuffer& img,
                                const std::vector<float>& kx,
                                const std::vector<float>& ky);

ImageBuffer gaussian_blur(const ImageBuffer& img, float sigma, int ksize = 0);

ImageBuffer bilateral_filter(const ImageBuffer& img, float sigma_s, float sigma_r, int ksize = 0);

ImageBuffer guided_filter(const ImageBuffer& src, const ImageBuffer& guide, float radius, float eps);

ImageBuffer median_filter(const ImageBuffer& img, int ksize = 3);

ImageBuffer unsharp_mask(const ImageBuffer& img, float sigma, float strength, float threshold = 0.0f);

// ─── Edge Detection ────────────────────────────────────────────────────────
ImageBuffer sobel_edges(const ImageBuffer& img, bool normalize = true);

ImageBuffer canny_edges(const ImageBuffer& img, float low_thresh, float high_thresh, float sigma = 1.0f);

ImageBuffer laplacian_of_gaussian(const ImageBuffer& img, float sigma);

ImageBuffer scharr_edges(const ImageBuffer& img);

struct EdgeMap {
    ImageBuffer magnitude;
    ImageBuffer direction;
    ImageBuffer phase;
};

EdgeMap compute_gradient_field(const ImageBuffer& img);

// ─── Morphological Operations ──────────────────────────────────────────────
enum class MorphShape { RECT, ELLIPSE, CROSS };

ImageBuffer dilate(const ImageBuffer& img, int ksize = 3, MorphShape shape = MorphShape::RECT, int iterations = 1);

ImageBuffer erode(const ImageBuffer& img, int ksize = 3, MorphShape shape = MorphShape::RECT, int iterations = 1);

ImageBuffer morph_open(const ImageBuffer& img, int ksize = 3, MorphShape shape = MorphShape::RECT);

ImageBuffer morph_close(const ImageBuffer& img, int ksize = 3, MorphShape shape = MorphShape::RECT);

ImageBuffer morph_gradient(const ImageBuffer& img, int ksize = 3);

ImageBuffer top_hat(const ImageBuffer& img, int ksize = 15);

ImageBuffer black_hat(const ImageBuffer& img, int ksize = 15);

// ─── Color Space Conversion ────────────────────────────────────────────────
ImageBuffer rgb_to_lab(const ImageBuffer& img);

ImageBuffer lab_to_rgb(const ImageBuffer& img);

ImageBuffer rgb_to_hsv(const ImageBuffer& img);

ImageBuffer hsv_to_rgb(const ImageBuffer& img);

ImageBuffer rgb_to_ycbcr(const ImageBuffer& img);

ImageBuffer ycbcr_to_rgb(const ImageBuffer& img);

ImageBuffer rgb_to_xyz(const ImageBuffer& img, const std::string& illuminant = "D65");

ImageBuffer xyz_to_rgb(const ImageBuffer& img, const std::string& illuminant = "D65");

ImageBuffer apply_lut(const ImageBuffer& img, const std::vector<float>& lut_r,
                       const std::vector<float>& lut_g, const std::vector<float>& lut_b);

// ─── Frequency Domain ─────────────────────────────────────────────────────
struct ComplexBuffer {
    std::vector<float> real, imag;
    int width, height;
    
    ComplexBuffer(int w, int h) : width(w), height(h), real(w * h, 0), imag(w * h, 0) {}
};

ComplexBuffer fft2d(const ImageBuffer& img);

ImageBuffer ifft2d(const ComplexBuffer& freq);

ImageBuffer apply_frequency_filter(const ImageBuffer& img,
                                    const std::function<float(float, float)>& filter_fn);

ImageBuffer butterworth_lowpass(const ImageBuffer& img, float cutoff, int order = 2);

ImageBuffer butterworth_highpass(const ImageBuffer& img, float cutoff, int order = 2);

ImageBuffer band_pass_filter(const ImageBuffer& img, float low_cutoff, float high_cutoff);

ImageBuffer wiener_deconvolution(const ImageBuffer& blurred,
                                  const std::vector<float>& psf_kernel, int kw, int kh, float snr = 40.0f);

// ─── Geometric Transforms ─────────────────────────────────────────────────
ImageBuffer resize(const ImageBuffer& img, int new_w, int new_h, const std::string& interp = "lanczos");

ImageBuffer rotate(const ImageBuffer& img, float angle_deg, bool expand = false,
                   float fill = 0.0f);

ImageBuffer warp_affine(const ImageBuffer& img, const std::array<float, 6>& M, int out_w, int out_h);

ImageBuffer warp_perspective(const ImageBuffer& img, const std::array<float, 9>& H, int out_w, int out_h);

ImageBuffer lens_distortion_correct(const ImageBuffer& img,
                                     float k1, float k2, float p1, float p2, float k3 = 0.0f);

// ─── Image Quality Metrics ─────────────────────────────────────────────────
struct QualityMetrics {
    double psnr;
    double ssim;
    double ms_ssim;
    double lpips_approx;
    double brisque;
    double niqe;
};

QualityMetrics compute_quality(const ImageBuffer& original, const ImageBuffer& processed);

double compute_psnr(const ImageBuffer& a, const ImageBuffer& b);

double compute_ssim(const ImageBuffer& a, const ImageBuffer& b, int window = 11);

double compute_brisque(const ImageBuffer& img);

// ─── Advanced Color Grading ────────────────────────────────────────────────
ImageBuffer color_grade(const ImageBuffer& img,
                         float lift_r, float lift_g, float lift_b,
                         float gamma_r, float gamma_g, float gamma_b,
                         float gain_r, float gain_g, float gain_b);

ImageBuffer color_balance(const ImageBuffer& img, float shadows, float midtones, float highlights);

ImageBuffer selective_color(const ImageBuffer& img, const std::string& color_target,
                             float cyan_magenta, float yellow_black);

ImageBuffer vibrance(const ImageBuffer& img, float vibrance_amount, float saturation = 0.0f);

// ─── Noise Analysis & Estimation ──────────────────────────────────────────
struct NoiseProfile {
    float sigma_luminance;
    float sigma_chroma;
    std::vector<float> noise_per_channel;
    std::string noise_type;
};

NoiseProfile estimate_noise(const ImageBuffer& img);

ImageBuffer add_noise(const ImageBuffer& img, float sigma, const std::string& type = "gaussian", uint32_t seed = 42);

// ─── Utility ──────────────────────────────────────────────────────────────
ImageBuffer clamp_image(const ImageBuffer& img, float lo = 0.0f, float hi = 1.0f);

ImageBuffer normalize_image(const ImageBuffer& img);

ImageBuffer blend(const ImageBuffer& a, const ImageBuffer& b, float alpha, const std::string& mode = "normal");

ImageBuffer split_channel(const ImageBuffer& img, int channel);

ImageBuffer merge_channels(const std::vector<ImageBuffer>& channels);

ImageBuffer from_bytes(const std::vector<uint8_t>& bytes, int w, int h, int c);

std::vector<uint8_t> to_bytes(const ImageBuffer& img);

ImageBuffer pad_image(const ImageBuffer& img, int top, int bottom, int left, int right,
                       const std::string& mode = "reflect");

ImageBuffer crop(const ImageBuffer& img, int x, int y, int w, int h);

} // namespace simd_engine
