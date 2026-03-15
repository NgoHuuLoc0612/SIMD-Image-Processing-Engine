#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "image_features.hpp"
#include "simd_optimization.hpp"
#include "multi_threading.hpp"
#include "neural_rendering.hpp"

namespace py = pybind11;
using namespace simd_engine;

// ── numpy <-> ImageBuffer ─────────────────────────────────────────────────
static ImageBuffer from_np_f32(py::array_t<float, py::array::c_style> arr) {
    auto info = arr.request();
    if (info.ndim == 2) {
        int h=(int)info.shape[0], w=(int)info.shape[1];
        ImageBuffer img(w,h,1);
        std::memcpy(img.data.data(), info.ptr, (size_t)w*h*sizeof(float));
        return img;
    } else if (info.ndim == 3) {
        int h=(int)info.shape[0], w=(int)info.shape[1], c=(int)info.shape[2];
        ImageBuffer img(w,h,c);
        std::memcpy(img.data.data(), info.ptr, (size_t)w*h*c*sizeof(float));
        return img;
    }
    throw std::runtime_error("Array must be 2-D (H,W) or 3-D (H,W,C)");
}

static py::array_t<float> to_np(const ImageBuffer& img) {
    if (img.channels == 1) {
        py::array_t<float> out({img.height, img.width});
        std::memcpy(out.mutable_data(), img.data.data(), (size_t)img.width*img.height*sizeof(float));
        return out;
    }
    py::array_t<float> out({img.height, img.width, img.channels});
    std::memcpy(out.mutable_data(), img.data.data(), img.data.size()*sizeof(float));
    return out;
}

static ImageBuffer from_np_u8(py::array_t<uint8_t, py::array::c_style> arr) {
    auto info = arr.request();
    int h,w,c;
    if (info.ndim==2){h=(int)info.shape[0];w=(int)info.shape[1];c=1;}
    else if (info.ndim==3){h=(int)info.shape[0];w=(int)info.shape[1];c=(int)info.shape[2];}
    else throw std::runtime_error("Array must be 2-D or 3-D");
    auto* src = static_cast<uint8_t*>(info.ptr);
    return from_bytes(std::vector<uint8_t>(src, src+(size_t)h*w*c), w, h, c);
}

static py::array_t<uint8_t> to_u8(const ImageBuffer& img) {
    auto bytes = to_bytes(img);
    py::array_t<uint8_t> out;
    if (img.channels==1) out = py::array_t<uint8_t>({img.height,img.width});
    else                  out = py::array_t<uint8_t>({img.height,img.width,img.channels});
    std::memcpy(out.mutable_data(), bytes.data(), bytes.size());
    return out;
}

// ── Module ────────────────────────────────────────────────────────────────
PYBIND11_MODULE(simd_engine, m) {
    m.doc() = "SIMD Image Processing Engine — enterprise C++/Python binding";

    // ImageBuffer
    py::class_<ImageBuffer>(m, "ImageBuffer")
        .def(py::init<>())
        .def(py::init<int,int,int>())
        .def_readwrite("width",    &ImageBuffer::width)
        .def_readwrite("height",   &ImageBuffer::height)
        .def_readwrite("channels", &ImageBuffer::channels)
        .def_readwrite("data",     &ImageBuffer::data)
        .def("to_numpy",  [](const ImageBuffer& s){ return to_np(s); })
        .def("to_uint8",  [](const ImageBuffer& s){ return to_u8(s); })
        .def_static("from_numpy", [](py::array_t<float,py::array::c_style> a){ return from_np_f32(a); })
        .def_static("from_uint8", [](py::array_t<uint8_t,py::array::c_style> a){ return from_np_u8(a); })
        .def("__repr__", [](const ImageBuffer& s){
            return "<ImageBuffer " + std::to_string(s.width) + "x" +
                   std::to_string(s.height) + "x" + std::to_string(s.channels) + ">";
        });

    // Histogram
    py::class_<Histogram>(m, "Histogram")
        .def_readwrite("bins",    &Histogram::bins)
        .def_readwrite("min_val", &Histogram::min_val)
        .def_readwrite("max_val", &Histogram::max_val)
        .def_readwrite("mean",    &Histogram::mean)
        .def_readwrite("stddev",  &Histogram::stddev)
        .def_readwrite("channel", &Histogram::channel);

    // MorphShape
    py::enum_<MorphShape>(m, "MorphShape")
        .value("RECT",    MorphShape::RECT)
        .value("ELLIPSE", MorphShape::ELLIPSE)
        .value("CROSS",   MorphShape::CROSS)
        .export_values();

    // CpuFeatures
    py::class_<opt::CpuFeatures>(m, "CpuFeatures")
        .def_readonly("avx512f",           &opt::CpuFeatures::avx512f)
        .def_readonly("avx2",              &opt::CpuFeatures::avx2)
        .def_readonly("fma",               &opt::CpuFeatures::fma)
        .def_readonly("num_logical_cores", &opt::CpuFeatures::num_logical_cores)
        .def_readonly("simd_width_float",  &opt::CpuFeatures::simd_width_float)
        .def("best_simd_name",             &opt::CpuFeatures::best_simd_name)
        .def("__repr__", [](const opt::CpuFeatures& f){
            return std::string("<CpuFeatures ") + f.best_simd_name() + ">";
        });

    m.def("detect_cpu", &opt::detect_cpu, py::return_value_policy::reference);

    // BenchResult
    py::class_<opt::BenchResult>(m, "BenchResult")
        .def_readwrite("mean_ms",         &opt::BenchResult::mean_ms)
        .def_readwrite("min_ms",          &opt::BenchResult::min_ms)
        .def_readwrite("max_ms",          &opt::BenchResult::max_ms)
        .def_readwrite("throughput_mpps", &opt::BenchResult::throughput_mpps)
        .def_readwrite("iterations",      &opt::BenchResult::iterations);

    m.def("benchmark", [](py::function fn, int warmup, int iters){
        return opt::benchmark_fn([&fn]{ fn(); }, warmup, iters);
    }, py::arg("fn"), py::arg("warmup")=3, py::arg("iters")=10);

    // SystemTopology (from platform.hpp via mt namespace alias)
    py::class_<simd_platform::NumaNodeInfo>(m, "NumaNode")
        .def_readonly("node_id",            &simd_platform::NumaNodeInfo::node_id)
        .def_readonly("processor_ids",      &simd_platform::NumaNodeInfo::processor_ids)
        .def_readonly("avail_memory_bytes", &simd_platform::NumaNodeInfo::avail_memory_bytes)
        .def_readonly("total_memory_bytes", &simd_platform::NumaNodeInfo::total_memory_bytes);

    py::class_<simd_platform::SystemTopology>(m, "NumaTopology")
        .def_readonly("numa_available",    &simd_platform::SystemTopology::numa_available)
        .def_readonly("num_numa_nodes",    &simd_platform::SystemTopology::num_numa_nodes)
        .def_readonly("num_logical_cpus",  &simd_platform::SystemTopology::num_logical_cpus)
        .def_readonly("num_physical_cores",&simd_platform::SystemTopology::num_physical_cores)
        .def_readonly("nodes",             &simd_platform::SystemTopology::nodes);

    m.def("get_topology", []() -> const simd_platform::SystemTopology& {
        return mt::get_topology();
    }, py::return_value_policy::reference);

    // WorkerStats
    py::class_<mt::WorkerStats>(m, "WorkerStats")
        .def_property_readonly("tasks_executed", [](const mt::WorkerStats& s){ return s.tasks_executed.load(); })
        .def_property_readonly("tasks_stolen",   [](const mt::WorkerStats& s){ return s.tasks_stolen.load(); })
        .def_readonly("cpu_id",  &mt::WorkerStats::cpu_id)
        .def_readonly("node_id", &mt::WorkerStats::node_id);

    // Image utilities
    m.def("clamp_image",     &clamp_image,      py::arg("img"), py::arg("lo")=0.f, py::arg("hi")=1.f);
    m.def("normalize_image", &normalize_image,  py::arg("img"));
    m.def("crop",            &crop,             py::arg("img"), py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"));
    m.def("pad_image",       &pad_image,        py::arg("img"), py::arg("top"), py::arg("bottom"), py::arg("left"), py::arg("right"), py::arg("mode")="reflect");
    m.def("split_channel",   &split_channel,    py::arg("img"), py::arg("channel"));
    m.def("merge_channels",  &merge_channels,   py::arg("channels"));
    m.def("blend",           &blend,            py::arg("a"), py::arg("b"), py::arg("alpha"), py::arg("mode")="normal");

    // Filters
    m.def("gaussian_blur",       &gaussian_blur,      py::arg("img"), py::arg("sigma"), py::arg("ksize")=0);
    m.def("bilateral_filter",    &bilateral_filter,   py::arg("img"), py::arg("sigma_s"), py::arg("sigma_r"), py::arg("ksize")=0);
    m.def("median_filter",       &median_filter,      py::arg("img"), py::arg("ksize")=3);
    m.def("unsharp_mask",        &unsharp_mask,       py::arg("img"), py::arg("sigma"), py::arg("strength"), py::arg("threshold")=0.f);
    m.def("convolve2d", [](ImageBuffer img, py::list kl, int kw, int kh){
        std::vector<float> k; for(auto v:kl) k.push_back(v.cast<float>());
        return convolve2d(img, k, kw, kh);
    }, py::arg("img"), py::arg("kernel"), py::arg("kw"), py::arg("kh"));

    // Edge detection
    m.def("sobel_edges", &sobel_edges, py::arg("img"), py::arg("normalize")=true);
    m.def("canny_edges", &canny_edges, py::arg("img"), py::arg("low_thresh"), py::arg("high_thresh"), py::arg("sigma")=1.f);

    // Morphology
    m.def("dilate",      &dilate,      py::arg("img"), py::arg("ksize")=3, py::arg("shape")=MorphShape::RECT, py::arg("iterations")=1);
    m.def("erode",       &erode,       py::arg("img"), py::arg("ksize")=3, py::arg("shape")=MorphShape::RECT, py::arg("iterations")=1);
    m.def("morph_open",  &morph_open,  py::arg("img"), py::arg("ksize")=3, py::arg("shape")=MorphShape::RECT);
    m.def("morph_close", &morph_close, py::arg("img"), py::arg("ksize")=3, py::arg("shape")=MorphShape::RECT);

    // Histogram & tone
    m.def("compute_histogram",       &compute_histogram,       py::arg("img"), py::arg("channel")=0, py::arg("bins")=256);
    m.def("compute_all_histograms",  &compute_all_histograms,  py::arg("img"), py::arg("bins")=256);
    m.def("histogram_equalization",  &histogram_equalization,  py::arg("img"));
    m.def("tone_map_reinhard",       &tone_map_reinhard,       py::arg("img"), py::arg("exposure")=1.f, py::arg("gamma")=2.2f);
    m.def("tone_map_aces",           &tone_map_aces,           py::arg("img"), py::arg("exposure")=1.f);

    // Color
    m.def("rgb_to_hsv", &rgb_to_hsv, py::arg("img"));
    m.def("hsv_to_rgb", &hsv_to_rgb, py::arg("img"));

    // Geometry
    m.def("resize", &resize, py::arg("img"), py::arg("width"), py::arg("height"), py::arg("interp")="bilinear");

    // Noise & quality
    m.def("add_noise",    &add_noise,    py::arg("img"), py::arg("sigma"), py::arg("type")="gaussian", py::arg("seed")=42);
    m.def("compute_psnr", &compute_psnr, py::arg("a"), py::arg("b"));

    // Neural configs
    py::class_<neural::UpscaleConfig>(m, "UpscaleConfig")
        .def(py::init<>())
        .def_readwrite("scale",    &neural::UpscaleConfig::scale)
        .def_readwrite("model",    &neural::UpscaleConfig::model)
        .def_readwrite("tile",     &neural::UpscaleConfig::tile)
        .def_readwrite("tile_sz",  &neural::UpscaleConfig::tile_sz)
        .def_readwrite("overlap",  &neural::UpscaleConfig::overlap)
        .def_readwrite("sharpen",  &neural::UpscaleConfig::sharpen);

    py::class_<neural::DenoiseConfig>(m, "DenoiseConfig")
        .def(py::init<>())
        .def_readwrite("model",    &neural::DenoiseConfig::model)
        .def_readwrite("strength", &neural::DenoiseConfig::strength)
        .def_readwrite("sigma",    &neural::DenoiseConfig::sigma)
        .def_readwrite("blind",    &neural::DenoiseConfig::blind);

    py::class_<neural::ColorPalette>(m, "ColorPalette")
        .def_readwrite("colors",  &neural::ColorPalette::colors)
        .def_readwrite("weights", &neural::ColorPalette::weights)
        .def_readwrite("k",       &neural::ColorPalette::k);

    // Neural functions — output always clamped to [0,1] at binding boundary
    m.def("super_resolution", [](const ImageBuffer& img, const neural::UpscaleConfig& cfg){
        return clamp_image(neural::super_resolution(img, cfg), 0.f, 1.f);
    }, py::arg("img"), py::arg("config")=neural::UpscaleConfig{});
    m.def("neural_denoise", [](const ImageBuffer& img, const neural::DenoiseConfig& cfg){
        return clamp_image(neural::neural_denoise(img, cfg), 0.f, 1.f);
    }, py::arg("img"), py::arg("config")=neural::DenoiseConfig{});
    m.def("depth_estimation", [](const ImageBuffer& img, const std::string& model, bool norm, bool col){
        neural::DepthConfig cfg; cfg.model=model; cfg.normalize=norm; cfg.colorize=col;
        return clamp_image(neural::depth_estimation(img, cfg), 0.f, 1.f);
    }, py::arg("img"), py::arg("model")="midas", py::arg("normalize")=true, py::arg("colorize")=false);
    m.def("colorize_grayscale", [](const ImageBuffer& img, float strength){
        return clamp_image(neural::colorize_grayscale(img, strength), 0.f, 1.f);
    }, py::arg("img"), py::arg("strength")=1.f);
    m.def("extract_palette",    &neural::extract_palette,    py::arg("img"), py::arg("k")=8, py::arg("max_iter")=20);

    // System info
    m.def("system_info", []() -> py::dict {
        const auto& cpu  = opt::detect_cpu();
        const auto& topo = mt::get_topology();
        py::dict d;
        d["simd_level"]       = cpu.best_simd_name();
        d["simd_width"]       = cpu.simd_width_float;
        d["logical_cores"]    = cpu.num_logical_cores;
        d["physical_cores"]   = cpu.num_physical_cores;
        d["avx512"]           = cpu.avx512f;
        d["avx2"]             = cpu.avx2;
        d["fma"]              = cpu.fma;
        d["numa_available"]   = topo.numa_available;
        d["numa_nodes"]       = (int)topo.num_numa_nodes;
        d["profiler_totals"]  = mt::ScopedTimer::snapshot();
        return d;
    });

    m.def("reset_profiler", &mt::ScopedTimer::reset);

    m.attr("__version__") = "2.1.0-enterprise";
}
