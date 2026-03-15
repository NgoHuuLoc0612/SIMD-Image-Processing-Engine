# SIMD Image Processing Engine

Image processing engine with a C++/AVX2 SIMD core, ONNX Runtime neural backend, NUMA-aware thread pool, and a Flask/WebSocket web interface.

---

## Architecture

```
simd_engine/
├── src/
│   ├── engine_impl.cpp          # Single TU: §1 CPU/SIMD  §2 threading  §3 image ops  §4 neural
│   ├── bindings.cpp             # pybind11 Python bindings
│   ├── platform.hpp             # Cross-platform: MSVC / MinGW / GCC / Clang
│   ├── image_features.hpp       # ImageBuffer + full algorithm surface
│   ├── simd_optimization.hpp    # AVX2/AVX-512 primitives, tile config, benchmark
│   ├── multi_threading.hpp      # Chase-Lev work-stealing pool, task graph, profiler
│   └── neural_rendering.hpp     # OrtSession, TiledInference, NormParams, ref algorithms
├── models/                      # ONNX model files (populated by convert_pth_to_onnx.py)
│   ├── realesrgan.onnx
│   └── midas.onnx
├── server.py                    # Flask + Flask-SocketIO server
├── index.html                   # Single-file web UI
├── CMakeLists.txt               # MSVC / MinGW / GCC, auto-detects AVX2/AVX-512
├── run.bat                      # One-shot build + launch
├── convert_pth_to_onnx.py       # .pth → .onnx model converter
├── download_models.py           # ONNX Runtime DLL + MiDaS download
└── test_onnx.py                 # ONNX + C++ extension test suite
```

**Runtime stack:**

```
Python (Flask + SocketIO)
    └── simd_engine.cp311-win_amd64.pyd   (pybind11)
            ├── §1  AVX2 SIMD primitives  (simd_optimization)
            ├── §2  NUMA thread pool      (multi_threading)
            ├── §3  Image algorithms      (image_features)
            └── §4  Neural inference      (neural_rendering)
                    ├── OrtSession → onnxruntime.dll  (real models)
                    └── Reference C++ fallback         (no DLL needed)
```

---

## Requirements

| Component | Minimum |
|---|---|
| OS | Windows 10 x64 |
| Python | 3.10 – 3.12 (CPython x64) |
| C++ compiler | MSVC 2019+, MinGW-w64 13+, or Clang-cl 16+ |
| CMake | 3.20+ |
| CPU | x86-64 with AVX2 + FMA (Intel Haswell 2013+ / AMD Ryzen 2017+) |
| RAM | 4 GB minimum, 8 GB recommended for large images |
| ONNX Runtime | 1.15 – 1.20 (optional, enables neural models) |

Python packages (installed automatically by `run.bat`):

```
flask  flask-cors  flask-socketio  gevent  gevent-websocket
numpy  Pillow  scipy
```

---

## Quick Start

```bat
:: Clone or extract the project, then:
cd simd_engine
run.bat
```

`run.bat` does the following in order:

1. `pip install -r requirements.txt`
2. Detects the best available CMake generator (Visual Studio 18/17/16, Ninja, NMake)
3. Configures with AVX2 (AVX-512 intentionally disabled for portability)
4. Builds `simd_engine.cp311-win_amd64.pyd`
5. Copies `.pyd` to the project root
6. Starts `server.py` on `http://localhost:5000`

Open `http://localhost:5000` in a browser to use the web interface.

---

## Neural Models (Optional)

Neural functions (`super_resolution`, `neural_denoise`, `depth_estimation`) use ONNX Runtime when both `onnxruntime.dll` and the corresponding `.onnx` file are present. Without them, a full C++ reference implementation is used automatically — no configuration needed.

### Step 1 — ONNX Runtime DLL

Download the Windows x64 release from Microsoft:

```
https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-win-x64-1.18.1.zip
```

Extract `onnxruntime-win-x64-1.18.1\lib\onnxruntime.dll` and place it next to `server.py`.

### Step 2 — MiDaS (depth estimation)

Direct binary, no conversion needed:

```bat
curl -L "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx" -o models\midas.onnx
```

### Step 3 — Real-ESRGAN (super resolution)

Download the official `.pth` weights and convert to ONNX:

```bat
:: Download weights (67 MB, direct binary)
curl -L "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -o RealESRGAN_x4plus.pth

:: Convert to ONNX (installs torch CPU + basicsr + onnx automatically)
python convert_pth_to_onnx.py
:: Output: models\realesrgan.onnx

:: Anime variant (optional, 17 MB, 6-block architecture)
curl -L "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" -o RealESRGAN_x4plus_anime_6B.pth
python convert_pth_to_onnx.py --pth RealESRGAN_x4plus_anime_6B.pth --anime
```

### Verify

```bat
pip install onnxruntime
python test_onnx.py --image photo.jpg
```

Expected output:

```
[PASS] import onnxruntime  -- version 1.x.x
[PASS] midas.onnx          -- load Xms  run Xms  in[1,3,256,256] -> out[1,1,256,256]
[PASS] realesrgan.onnx     -- load Xms  run Xms  in[1,3,64,64]   -> out[1,3,256,256]
[PASS] import simd_engine  -- 2.1.0
[PASS] super_resolution (x4, realesrgan)
[PASS] neural_denoise (sigma=25)
[PASS] depth_estimation (colorize)
```

---

## SIMD and Threading

The engine detects the CPU feature set at startup and selects the best available path:

| Level | Width | Enabled when |
|---|---|---|
| AVX2 + FMA | 8 floats/cycle | Default on Haswell+ |
| SSE4.2 | 4 floats/cycle | Fallback |
| Scalar | 1 float/cycle | Always available |

AVX-512 is disabled at compile time because MSVC's compile-time `CPUID` check does not reflect runtime availability — Alder Lake P-cores support AVX-512 but E-cores do not, causing illegal instruction faults on mixed-core schedules.

The thread pool is a Chase-Lev work-stealing deque pool pinned to logical CPUs. On NUMA systems (`numa_available = true`) allocations use `VirtualAllocExNuma` for node-local memory. On single-socket systems (the common laptop case) the pool still uses all logical cores with above-normal thread priority.

To check what the engine detected on your machine:

```python
import simd_engine
print(simd_engine.system_info())
```

---

## API Reference

### REST

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serve `index.html` |
| GET | `/api/system_info` | CPU features, SIMD level, thread count, NUMA topology |
| POST | `/api/process` | Apply one algorithm to an image |
| POST | `/api/histogram` | Per-channel histogram (256 bins, float) |
| POST | `/api/palette` | K-means++ dominant color extraction |
| POST | `/api/benchmark` | Benchmark a list of algorithms on an image |

`/api/process` request body:

```json
{
  "image":     "<data:image/png;base64,...>",
  "algorithm": "gaussian_blur",
  "params":    { "sigma": 2.0 }
}
```

### WebSocket events

| Event (client → server) | Payload |
|---|---|
| `process_frame` | `{ image, algorithm, params, frame_id }` |
| `start_pipeline` | `{ image, stages: [{algorithm, params}, ...] }` |
| `stop_pipeline` | — |

| Event (server → client) | Payload |
|---|---|
| `frame_result` | `{ result, stats, frame_id }` |
| `pipeline_stage` | `{ stage, total_stages, result, stats, progress }` |
| `pipeline_complete` | `{ result, message }` |
| `system_info` | `{ native_engine, simd_level, async_mode }` |

### Algorithms

| Name | Params | Description |
|---|---|---|
| `gaussian_blur` | `sigma` | Separable Gaussian, AVX2 horizontal pass |
| `bilateral_filter` | `sigma_s`, `sigma_r` | Edge-preserving bilateral filter |
| `median_filter` | `ksize` | Median filter, AVX2 3×3 sort network |
| `unsharp_mask` | `sigma`, `strength`, `threshold` | Unsharp masking |
| `sobel_edges` | `normalize` | Sobel magnitude + direction |
| `canny_edges` | `low_thresh`, `high_thresh`, `sigma` | Full Canny pipeline |
| `histogram_equalization` | — | Global HE, per-channel |
| `tone_map_reinhard` | `exposure`, `gamma` | Reinhard tone mapping |
| `tone_map_aces` | `exposure` | ACES filmic tone mapping |
| `dilate` / `erode` | `ksize`, `shape` | Morphological ops (RECT / ELLIPSE / CROSS) |
| `morph_open` / `morph_close` | `ksize` | Opening / closing |
| `rgb_to_hsv` | — | RGB → HSV color space |
| `add_noise` | `sigma`, `type` | Gaussian or salt-and-pepper noise |
| `resize` | `width`, `height` | Lanczos or bilinear resize |
| `super_resolution` | `scale`, `model` | 4× SR via Real-ESRGAN (ORT) or Lanczos-4 + IBP |
| `neural_denoise` | `sigma`, `strength` | NAFNet / SCUNet (ORT) or Non-Local Means |
| `depth_estimation` | `colorize` | MiDaS (ORT) or structure-tensor depth |
| `colorize_grayscale` | `strength` | Lab histogram transfer from natural image prior |
| `frequency_highpass` | `cutoff` | High-pass via Gaussian difference |
| `frequency_lowpass` | `cutoff` | Low-pass Gaussian |

---

## Building from Source

### MSVC (recommended)

```bat
cmake -B build -G "Visual Studio 17 2022" -A x64 -Dpybind11_DIR=<pybind11_cmake_dir>
cmake --build build --config Release
copy build\Release\simd_engine*.pyd .
```

### MinGW-w64

```bat
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=<pybind11_cmake_dir>
cmake --build build
copy build\simd_engine*.pyd .
```

pybind11 cmake dir:

```bat
python -c "import pybind11; print(pybind11.get_cmake_dir())"
```

### Build flags

| CMake variable | Default | Effect |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug` disables LTO and optimizations |
| `SIMD_ENGINE_FORCE_AVX2` | off | Force AVX2 even if AVX-512 detected |

---

## Project History

| Version | Changes |
|---|---|
| 2.1.0 | ONNX Runtime backend, TiledInference with Hann-window blending, NormParams per model, Real-ESRGAN + MiDaS support, Non-Local Means reference denoiser, Lanczos-4 + IBP reference SR, structure-tensor depth, Lab colorization prior |
| 2.0.0 | NUMA-aware thread pool, Chase-Lev work stealing, task graph, AVX2 bilateral + separable filter, ACES tone mapping |
| 1.0.0 | Initial release: AVX2 SIMD core, Flask server, WebSocket pipeline |

---

## License

MIT. Model weights (Real-ESRGAN, MiDaS) are subject to their respective licenses — see the original repositories for details.
