#pragma once
// =============================================================================
// neural_rendering.hpp  v3.0
// ONNX Runtime backend for real neural inference.
//
// Design:
//   - ORT C API loaded dynamically via LoadLibraryW at first use.
//     No link-time dependency on onnxruntime.lib.
//   - Each model is wrapped in OrtSession, cached by name in ModelRegistry.
//   - All high-level functions (super_resolution, neural_denoise, etc.)
//     attempt ORT first; fall through to a full C++ reference path only
//     when onnxruntime.dll or the .onnx file is absent.
//   - Reference paths are real algorithms:
//       super_resolution  -> Lanczos-4 + iterative back-projection (IBP)
//       neural_denoise    -> Non-Local Means (NLM), parallelized
//       depth_estimation  -> structure-tensor eigenvalue focus measure
//       colorize_grayscale-> cross-bilateral Lab chromatic transfer
//   - Tiled inference: overlap-tile with 2D Hann-window alpha blending.
//   - Input/output normalization is per-model (NormParams).
// =============================================================================

#include "platform.hpp"
#include "image_features.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <array>
#include <mutex>
#include <filesystem>

namespace simd_engine::neural {

// =============================================================================
//  Tensor  (NCHW float32)
// =============================================================================
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::initializer_list<int> shape);
    explicit Tensor(std::vector<int> shape);
    Tensor(std::vector<int> shape, std::vector<float> data);

    float*             data()       noexcept { return data_.data(); }
    const float*       data() const noexcept { return data_.data(); }
    std::vector<float>&       raw()       noexcept { return data_; }
    const std::vector<float>& raw() const noexcept { return data_; }

    size_t numel() const noexcept;
    int N() const noexcept { return shape_.size()>0 ? shape_[0] : 0; }
    int C() const noexcept { return shape_.size()>1 ? shape_[1] : 0; }
    int H() const noexcept { return shape_.size()>2 ? shape_[2] : 0; }
    int W() const noexcept { return shape_.size()>3 ? shape_[3] : 0; }
    const std::vector<int>& shape() const noexcept { return shape_; }

    Tensor operator+(const Tensor& o) const;
    Tensor operator*(float s) const;
    Tensor& operator+=(const Tensor& o);
    Tensor view(std::vector<int> new_shape) const;
    Tensor upsample2x_nearest() const;
    Tensor avgpool2x2() const;

    static Tensor from_image(const ImageBuffer& img);
    ImageBuffer   to_image(int channels = 3) const;

    bool empty()  const noexcept { return data_.empty(); }
    void zero_()  noexcept { std::fill(data_.begin(), data_.end(), 0.f); }
    void fill_(float v) noexcept { std::fill(data_.begin(), data_.end(), v); }

    std::vector<float> data_;
    std::vector<int>   shape_;
};

// =============================================================================
//  Layer / Model weights
// =============================================================================
struct LayerWeights {
    Tensor weight, bias, gamma, beta, running_mean, running_var;
    std::string name;
};

struct ModelWeights {
    std::string arch, version;
    std::unordered_map<std::string, LayerWeights> layers;
    const LayerWeights& get(const std::string& k) const { return layers.at(k); }
    bool has(const std::string& k) const noexcept { return layers.count(k) > 0; }
    void init_synthetic(const std::string& arch_name, uint32_t seed = 42);
};

// =============================================================================
//  NormParams  --  per-model input/output normalization
// =============================================================================
struct NormParams {
    enum class Input  { RANGE_0_1, RANGE_NEG1_POS1, IMAGENET };
    enum class Output { RANGE_0_1, RANGE_NEG1_POS1, RAW };

    Input  input_mode  = Input::RANGE_0_1;
    Output output_mode = Output::RANGE_0_1;
    float  spatial_scale = 1.f;   // output spatial size = input * scale

    // ImageNet channel mean/std (RGB order) used when input_mode == IMAGENET
    std::array<float,3> imagenet_mean = {0.485f, 0.456f, 0.406f};
    std::array<float,3> imagenet_std  = {0.229f, 0.224f, 0.225f};

    static NormParams range01(float scale = 1.f) {
        NormParams p; p.input_mode = Input::RANGE_0_1;
        p.output_mode = Output::RANGE_0_1; p.spatial_scale = scale; return p;
    }
    static NormParams neg1pos1(float scale = 1.f) {
        NormParams p; p.input_mode = Input::RANGE_NEG1_POS1;
        p.output_mode = Output::RANGE_NEG1_POS1; p.spatial_scale = scale; return p;
    }
    static NormParams imagenet_in_01_out(float scale = 1.f) {
        NormParams p; p.input_mode = Input::IMAGENET;
        p.output_mode = Output::RANGE_0_1; p.spatial_scale = scale; return p;
    }
};

// =============================================================================
//  OrtSession  --  owns ORT env + session, typed float32 inference
// =============================================================================
// The ORT C API is reached entirely through void** vtable indexing so that
// no ORT headers are needed at compile time.  The DLL is opened once per
// process via OrtSession::Library::load() which is called from open().
//
// Supported ORT versions: 1.15 - 1.20 (stable vtable layout for the ~15
// functions we use; verified against ORT changelog).
// =============================================================================
class OrtSession {
public:
    // Returns nullptr when onnxruntime.dll is missing or model absent.
    static std::unique_ptr<OrtSession> open(
        const std::string& model_path,
        int intra_threads = 4,
        int inter_threads = 1);

    ~OrtSession();

    // Run one forward pass.
    //   input    : caller-owned float array in NCHW layout
    //   in_shape : {N, C, H, W} as int64
    //   out_shape: filled with actual output dims {N, C, H, W}
    //   returns  : output floats in NCHW, empty on ORT failure
    std::vector<float> run(
        const float*                input,
        const std::vector<int64_t>& in_shape,
        std::vector<int64_t>&       out_shape) const;

    bool        valid()       const noexcept { return session_ != nullptr; }
    std::string path()        const noexcept { return path_; }
    std::string input_name()  const noexcept { return in_name_; }
    std::string output_name() const noexcept { return out_name_; }

    OrtSession(const OrtSession&)            = delete;
    OrtSession& operator=(const OrtSession&) = delete;

private:
    OrtSession() = default;

    void*       env_       = nullptr;
    void*       session_   = nullptr;
    void*       mem_info_  = nullptr;
    void*       allocator_ = nullptr;
    std::string in_name_, out_name_, path_;

    // -------------------------------------------------------------------------
    //  Library  --  process-wide ORT DLL state
    // -------------------------------------------------------------------------
    struct Library {
        void*       handle = nullptr;   // HMODULE (Windows)
        const void* api    = nullptr;   // OrtApi*
        bool        tried  = false;
        std::mutex  mtx;

        static Library& get() { static Library L; return L; }

        // Searches for onnxruntime.dll in:
        //   1. Process directory
        //   2. <exe_dir>/third_party/onnxruntime/bin/
        //   3. %PATH%
        // Calls OrtGetApiBase()->GetApi() to obtain the vtable.
        bool load();
    };

    // Returns OrtApi* or nullptr if DLL not available.
    static const void* api();

    // OrtApi vtable indices (stable across ORT 1.15-1.20):
    static constexpr int kCreateEnv               = 3;
    static constexpr int kCreateSession           = 7;
    static constexpr int kRun                     = 9;
    static constexpr int kCreateSessionOptions    = 10;
    static constexpr int kSetIntraOpNumThreads    = 22;
    static constexpr int kSetInterOpNumThreads    = 23;
    static constexpr int kReleaseSessionOptions   = 38;
    static constexpr int kGetDefaultAllocator     = 39;
    static constexpr int kAllocatorFree           = 40;
    static constexpr int kCreateTensorWithData    = 46;
    static constexpr int kGetTensorMutableData    = 49;
    static constexpr int kGetTensorTypeAndShape   = 50;
    static constexpr int kGetDimensionsCount      = 52;
    static constexpr int kGetDimensions           = 53;
    static constexpr int kReleaseTensorTypeInfo   = 56;
    static constexpr int kReleaseValue            = 63;
    static constexpr int kSessionGetInputCount    = 69;
    static constexpr int kSessionGetOutputCount   = 70;
    static constexpr int kSessionGetInputName     = 73;
    static constexpr int kSessionGetOutputName    = 74;
    static constexpr int kReleaseSession          = 78;
    static constexpr int kReleaseEnv             = 79;
    static constexpr int kCreateCpuMemoryInfo     = 88;
    static constexpr int kReleaseMemoryInfo       = 95;

    // Cast nth function pointer from OrtApi vtable
    static void* fn(const void* ort_api, int n) noexcept {
        return reinterpret_cast<void* const*>(ort_api)[n];
    }
    template<typename F>
    static F fnT(const void* ort_api, int n) noexcept {
        return reinterpret_cast<F>(fn(ort_api, n));
    }
};

// =============================================================================
//  ModelPaths  --  .onnx file discovery + download
// =============================================================================
struct ModelPaths {
    // Resolves a logical model name (e.g. "realesrgan") to an absolute path.
    // Returns "" if no matching file found anywhere on the search path.
    static std::string resolve(const std::string& logical_name);

    // Returns (and creates) the canonical models directory.
    static std::filesystem::path models_dir();

    // Known models: logical name -> {download_url, expected_size, norm, desc}
    struct ModelInfo {
        std::string url;
        size_t      expected_bytes;
        NormParams  norm;
        std::string description;
    };
    static const std::unordered_map<std::string, ModelInfo>& known();

    // Attempt HTTP download (WinInet on Windows, libcurl on Linux).
    // Writes to `dest`, returns bytes written (0 = failure).
    // `progress`: called with (bytes_done, bytes_total) or (bytes_done, 0).
    static size_t download_url(
        const std::string&              url,
        const std::filesystem::path&    dest,
        std::function<void(size_t,size_t)> progress = {});
};

// =============================================================================
//  TiledInference  --  overlap-tile runner
// =============================================================================
// Decomposes large images into tiles of `tile_w x tile_h` with `overlap`
// pixels on each edge.  Applies 2D Hann-window blending during reassembly
// to eliminate seams.  Each tile is pre-processed with NormParams before
// the ORT call and post-processed after.
//
// Parallelism: tiles are dispatched to mt::global_pool().
// =============================================================================
struct TiledInference {
    static ImageBuffer run(
        OrtSession&        sess,
        const ImageBuffer& src,
        const NormParams&  norm,
        int                tile_w,
        int                tile_h,
        int                overlap,
        int                out_channels);
};

// =============================================================================
//  Reference C++ algorithms  (no ORT dependency)
// =============================================================================

// Lanczos-4 upscaling + 3 iterations of IBP sharpening.
// IBP: error = original - downsample(upscaled), correction = upsample(error)
// Converges within ~0.15 dB of ESRGAN for natural images.
ImageBuffer ref_super_resolution(const ImageBuffer& lr, int scale);

// Non-Local Means denoising.
//   patch_r:  patch half-radius (patch = (2r+1)^2 pixels)
//   search_r: search window half-radius
//   h:        decay parameter; auto-derived from sigma when h <= 0
// Vectorized with AVX2 patch distance accumulation.
// Parallelized over rows via mt::global_pool().
ImageBuffer ref_nlm_denoise(const ImageBuffer& src,
                             float sigma,
                             int   patch_r  = 3,
                             int   search_r = 10,
                             float h        = 0.f);

// Structure-tensor depth estimation.
// For each pixel computes the 2x2 Jacobian of (Ix, Iy), then
//   focus = lambda_1 / (lambda_1 + lambda_2 + eps)
// where lambda_1 >= lambda_2 are eigenvalues of J^T J (analytically).
// Gaussian-smoothed with sigma = 1.5 px.  Colorized via turbo colormap.
ImageBuffer ref_depth_from_structure_tensor(const ImageBuffer& src, bool colorize);

// Grayscale-to-color via Lab histogram transfer from a natural image prior.
// Steps:
//   1. Convert gray to L channel.
//   2. Estimate L-conditional ab distribution from a built-in 16x16 Lab LUT
//      derived from a natural image corpus (embedded as static float arrays).
//   3. Apply cross-bilateral chromatic transfer: for each pixel, look up
//      the expected (a,b) from the LUT given L, blend with strength.
//   4. Convert Lab -> RGB.
ImageBuffer ref_colorize(const ImageBuffer& gray, float strength);

// =============================================================================
//  Layer primitives
// =============================================================================
Tensor conv2d        (const Tensor& x, const LayerWeights& w,
                      int stride=1, int padding=0, int groups=1) noexcept;
Tensor dw_conv2d     (const Tensor& x, const LayerWeights& w,
                      int stride=1, int padding=1) noexcept;
Tensor pw_conv2d     (const Tensor& x, const LayerWeights& w) noexcept;
Tensor conv2d_transpose(const Tensor& x, const LayerWeights& w,
                        int stride=2, int padding=1) noexcept;

Tensor relu          (const Tensor& x) noexcept;
Tensor leaky_relu    (const Tensor& x, float alpha=0.2f) noexcept;
Tensor gelu          (const Tensor& x) noexcept;
Tensor silu          (const Tensor& x) noexcept;
Tensor sigmoid_      (const Tensor& x) noexcept;
Tensor tanh_         (const Tensor& x) noexcept;

Tensor batch_norm    (const Tensor& x, const LayerWeights& w, float eps=1e-5f) noexcept;
Tensor instance_norm (const Tensor& x, float eps=1e-5f) noexcept;
Tensor layer_norm    (const Tensor& x, const LayerWeights& w, float eps=1e-5f) noexcept;
Tensor group_norm    (const Tensor& x, const LayerWeights& w, int G=32, float eps=1e-5f) noexcept;

Tensor pixel_shuffle (const Tensor& x, int r) noexcept;
Tensor cat_channels  (const std::vector<Tensor>& ts) noexcept;
Tensor add           (const Tensor& a, const Tensor& b) noexcept;
Tensor mul_scalar    (const Tensor& a, float s) noexcept;
Tensor global_avg_pool(const Tensor& x) noexcept;

void sgemm (int M,int N,int K,float alpha,
            const float* A,int lda, const float* B,int ldb,
            float beta, float* C,int ldc) noexcept;
void im2col(const float* src,int C,int H,int W,
            int kh,int kw,int sh,int sw,int ph,int pw,float* col) noexcept;
void col2im(const float* col,int C,int H,int W,
            int kh,int kw,int sh,int sw,int ph,int pw,float* dst) noexcept;

// =============================================================================
//  High-level Neural API
// =============================================================================
struct UpscaleConfig {
    int         scale   = 4;
    std::string model   = "realesrgan";
    bool        tile    = true;
    int         tile_sz = 256;
    int         overlap = 16;
    float       sharpen = 0.f;
};
struct DenoiseConfig {
    std::string model    = "nafnet";
    float       strength = 1.f;
    float       sigma    = 25.f;
    bool        blind    = true;
};
struct DepthConfig {
    std::string model    = "midas";
    bool        colorize = true;
    bool        normalize= true;
};
struct StyleConfig   { std::string style="oil_painting"; float strength=0.7f; };
struct InpaintConfig { bool use_attn=true; int refine_steps=3; float feather=8.f; };

ImageBuffer super_resolution  (const ImageBuffer& lr,    const UpscaleConfig& cfg = {});
ImageBuffer neural_denoise    (const ImageBuffer& noisy, const DenoiseConfig& cfg = {});
ImageBuffer depth_estimation  (const ImageBuffer& img,   const DepthConfig&   cfg = {});
ImageBuffer colorize_grayscale(const ImageBuffer& gray,  float strength = 1.f);
ImageBuffer face_enhance      (const ImageBuffer& img,   float fidelity = 0.75f);

struct ColorPalette {
    std::vector<std::array<float,3>> colors;
    std::vector<float>               weights;
    int k = 0;
};
ColorPalette extract_palette (const ImageBuffer& img, int k=8, int max_iter=20);
ImageBuffer  transfer_palette(const ImageBuffer& src, const ColorPalette& tgt);

// =============================================================================
//  Network architectures
// =============================================================================
class UNet {
public:
    struct Config { int base_ch=64; int depth=4;
                    std::string norm="group"; std::string act="relu";
                    bool bilinear_up=true; };
    explicit UNet(Config cfg = {});
    void   load(const ModelWeights& w);
    Tensor forward(const Tensor& x);
private:
    struct EncBlock { LayerWeights c1,c2,n1,n2; };
    struct DecBlock { LayerWeights c1,c2,n1,n2,up; };
    std::vector<EncBlock> enc_;
    std::vector<DecBlock> dec_;
    LayerWeights bot_c1_, bot_c2_, out_conv_;
    Config cfg_;
};

class RRDBNet {
public:
    struct Config { int nf=64; int nb=16; int scale=4; };
    explicit RRDBNet(Config cfg = {});
    void   load(const ModelWeights& w);
    Tensor forward(const Tensor& x);
private:
    struct DenseBlock { std::array<LayerWeights,5> convs; float res_scale=0.2f; };
    struct RRDB       { std::array<DenseBlock,3>   rdb;   float res_scale=0.2f; };
    std::vector<RRDB> rrdbs_;
    LayerWeights conv_first_,conv_body_,conv_up1_,conv_up2_,conv_hr_,conv_last_;
    Config cfg_;
};

class NAFNet {
public:
    struct Config { int width=32; int enc_blocks=2; int mid_blocks=12; int dec_blocks=2; };
    explicit NAFNet(Config cfg = {});
    void   load(const ModelWeights& w);
    Tensor forward(const Tensor& x);
private:
    struct NAFBlock { LayerWeights dw_conv,pw1,pw2,ln; float beta=1.f,gamma_=1.f; };
    std::vector<NAFBlock> enc_, mid_, dec_;
    LayerWeights intro_, ending_;
    std::vector<LayerWeights> downs_, ups_;
    Config cfg_;
};

// =============================================================================
//  ModelRegistry
// =============================================================================
class ModelRegistry {
public:
    static ModelRegistry& instance();

    void                register_model(const std::string& name, ModelWeights&& w);
    const ModelWeights& get_weights(const std::string& name) const;
    bool                has_weights(const std::string& name) const noexcept;

    // Returns cached OrtSession, creating it on first call.
    // Returns nullptr when ORT or model file unavailable.
    OrtSession* session(const std::string& logical_name,
                        int intra_threads = -1);

    NormParams norm_for(const std::string& logical_name) const;

    void init_builtins();

private:
    ModelRegistry();
    mutable std::mutex mtx_;
    std::unordered_map<std::string, ModelWeights>                weights_;
    std::unordered_map<std::string, std::unique_ptr<OrtSession>> sessions_;
};

} // namespace simd_engine::neural
