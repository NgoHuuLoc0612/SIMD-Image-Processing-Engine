"""
SIMD Image Processing Engine - Python Server
Windows-compatible: uses gevent (not eventlet) for async transport.
Falls back to threading mode if gevent is unavailable.
"""
from __future__ import annotations

# -- Async transport selection (MUST be before any other imports) -------------
import sys, os

ASYNC_MODE = os.environ.get("SIMD_ASYNC_MODE", "threading")

if ASYNC_MODE == "gevent":
    try:
        import gevent.monkey; gevent.monkey.patch_all()
    except ImportError:
        ASYNC_MODE = "threading"
elif ASYNC_MODE == "eventlet":
    try:
        import eventlet; eventlet.monkey_patch()
    except ImportError:
        ASYNC_MODE = "threading"

# -- Standard imports ---------------------------------------------------------
import base64, io, json, logging, math, time, traceback, threading
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image as PILImage
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# -- C++ extension (optional, graceful fallback) -------------------------------
NATIVE_ENGINE = False
_eng = None
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import simd_engine as _eng  # type: ignore
    NATIVE_ENGINE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("simd_engine")
log.info("Async mode   : %s", ASYNC_MODE)
log.info("Native engine: %s", NATIVE_ENGINE)

# -- Flask app -----------------------------------------------------------------
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Windows note: gevent mode requires allow_unsafe_werkzeug=True in newer flask-socketio
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode=ASYNC_MODE,
    max_http_buffer_size=128 * 1024 * 1024,   # 128 MB - for large images
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

# ------------------------------------------------------------------------------
# Image I/O helpers
# ------------------------------------------------------------------------------
def _decode(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.split(",")[-1])
    pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(pil, dtype=np.float32) / 255.0

def _encode(arr: np.ndarray, fmt: str = "PNG") -> str:
    u8  = np.clip(arr * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(u8).save(buf, format=fmt,
                                 optimize=(fmt == "PNG"),
                                 quality=90 if fmt == "JPEG" else None)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"

def _to_eng(arr: np.ndarray):
    return _eng.ImageBuffer.from_numpy(arr.astype(np.float32)) if NATIVE_ENGINE else arr

def _from_eng(obj) -> np.ndarray:
    return obj.to_numpy() if NATIVE_ENGINE else obj

# ------------------------------------------------------------------------------
# Pure-Python reference implementations
# (used when C++ extension not available, and as fallback per-algorithm)
# ------------------------------------------------------------------------------
class Ref:
    @staticmethod
    def gaussian_blur(img, sigma):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=[sigma, sigma, 0])

    @staticmethod
    def bilateral(img, ss, sr):
        try:
            import cv2
            u8  = (img * 255).clip(0, 255).astype(np.uint8)
            out = cv2.bilateralFilter(u8, -1, sr * 255, ss)
            return out.astype(np.float32) / 255.0
        except ImportError:
            return Ref.gaussian_blur(img, ss * 0.5)

    @staticmethod
    def median(img, k):
        from scipy.ndimage import median_filter
        return median_filter(img, size=(k, k, 1))

    @staticmethod
    def sobel(img):
        from scipy.ndimage import sobel as _sobel
        gray  = img.mean(axis=-1)
        sx, sy = _sobel(gray, 1), _sobel(gray, 0)
        mag   = np.hypot(sx, sy)
        mag  /= (mag.max() + 1e-9)
        return np.stack([mag, mag, mag], axis=-1)

    @staticmethod
    def canny(img, lo, hi, sigma):
        try:
            from skimage.feature import canny as _canny
            gray  = img.mean(axis=-1)
            edges = _canny(gray, sigma=sigma, low_threshold=lo, high_threshold=hi)
            e3    = edges.astype(np.float32)
            return np.stack([e3, e3, e3], axis=-1)
        except ImportError:
            return Ref.sobel(img)

    @staticmethod
    def hist_eq(img):
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            ch      = img[:, :, c]
            hist, _ = np.histogram(ch.flatten(), 256, [0, 1])
            cdf     = hist.cumsum().astype(np.float32)
            cdf    -= cdf[cdf > 0][0]
            cdf    /= (cdf[-1] + 1e-9)
            out[:, :, c] = np.interp(ch.flatten(), np.linspace(0, 1, 256), cdf).reshape(ch.shape)
        return out

    @staticmethod
    def tone_reinhard(img, exp=1.0, gamma=2.2):
        x = img * exp
        x = x / (1.0 + x)
        return np.power(np.clip(x, 0, None), 1.0 / gamma)

    @staticmethod
    def tone_aces(img, exp=1.0):
        x = img * exp
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return np.clip((x*(a*x+b))/(x*(c*x+d)+e), 0, 1)

    @staticmethod
    def unsharp(img, sigma, strength):
        return np.clip(img + strength * (img - Ref.gaussian_blur(img, sigma)), 0, 1)

    @staticmethod
    def dilate(img, k):
        from scipy.ndimage import grey_dilation
        return grey_dilation(img, size=(k, k, 1))

    @staticmethod
    def erode(img, k):
        from scipy.ndimage import grey_erosion
        return grey_erosion(img, size=(k, k, 1))

    @staticmethod
    def add_noise(img, sigma, ntype="gaussian", seed=42):
        rng = np.random.default_rng(seed)
        out = img.copy()
        if ntype == "gaussian":
            out += rng.normal(0, sigma, img.shape).astype(np.float32)
        elif ntype == "salt_pepper":
            m = rng.random(img.shape[:2])
            out[m < sigma / 2] = 0; out[m < sigma] = 1
        return np.clip(out, 0, 1)

    @staticmethod
    def psnr(a, b):
        mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
        return 100.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)

    @staticmethod
    def super_res(img, scale):
        h, w = img.shape[:2]
        pil  = PILImage.fromarray((img * 255).astype(np.uint8))
        pil  = pil.resize((w * scale, h * scale), PILImage.LANCZOS)
        return np.asarray(pil, dtype=np.float32) / 255.0

    @staticmethod
    def denoise(img, sigma):
        return Ref.gaussian_blur(img, sigma / 255.0 * 2.5)

    @staticmethod
    def depth(img):
        gray   = img.mean(axis=-1, keepdims=True)
        gx     = np.gradient(gray[:, :, 0], axis=1)[..., np.newaxis]
        gy     = np.gradient(gray[:, :, 0], axis=0)[..., np.newaxis]
        d      = np.sqrt(gx**2 + gy**2)
        d      = (d - d.min()) / (d.max() + 1e-9)
        col    = np.zeros((*img.shape[:2], 3), dtype=np.float32)
        col[..., 0] = d[..., 0]
        col[..., 1] = 1.0 - d[..., 0]
        col[..., 2] = 0.5
        return col

    @staticmethod
    def colorize(img, strength):
        v   = img.mean(axis=-1, keepdims=True)
        rgb = np.concatenate([
            np.clip(v * 1.1, 0, 1) * strength + v * (1 - strength),
            v,
            np.clip(v * 0.9, 0, 1) * strength + v * (1 - strength),
        ], axis=-1)
        return rgb.astype(np.float32)

    @staticmethod
    def palette(img, k):
        px = img.reshape(-1, 3)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(px), k, replace=False)
        centroids = px[idx].copy()
        for _ in range(25):
            d      = np.linalg.norm(px[:, None] - centroids[None], axis=-1)
            labels = d.argmin(axis=1)
            new_c  = np.array([px[labels == j].mean(0) if (labels == j).any()
                               else centroids[j] for j in range(k)])
            if np.allclose(centroids, new_c, atol=1e-4): break
            centroids = new_c
        labels    = np.linalg.norm(px[:, None] - centroids[None], axis=-1).argmin(1)
        counts    = np.bincount(labels, minlength=k)
        weights   = counts / counts.sum()
        order     = np.argsort(-weights)
        return [{"r": float(centroids[i][0]), "g": float(centroids[i][1]),
                 "b": float(centroids[i][2]), "weight": float(weights[i])}
                for i in order]

    @staticmethod
    def rgb_to_hsv(img):
        mx  = img.max(axis=-1)
        mn  = img.min(axis=-1)
        d   = mx - mn + 1e-9
        h   = np.zeros_like(mx)
        r, g, b = img[...,0], img[...,1], img[...,2]
        mask_r = (mx == r)
        mask_g = (mx == g) & ~mask_r
        mask_b = ~mask_r & ~mask_g
        h[mask_r] = ((g - b)[mask_r] / d[mask_r]) % 6
        h[mask_g] = (b - r)[mask_g] / d[mask_g] + 2
        h[mask_b] = (r - g)[mask_b] / d[mask_b] + 4
        h /= 6.0
        s = np.where(mx > 0, d / mx, 0)
        return np.stack([h, s, mx], axis=-1)

# ------------------------------------------------------------------------------
# Algorithm Dispatcher
# ------------------------------------------------------------------------------
def run_alg(name: str, img: np.ndarray, params: dict) -> Tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    result = img

    def eng(arr): return _to_eng(arr)
    def out(obj): return np.clip(_from_eng(obj), 0, 1).astype(np.float32)

    try:
        match name:
            case "gaussian_blur":
                sig = float(params.get("sigma", 2.0))
                result = out(_eng.gaussian_blur(eng(img), sig)) if NATIVE_ENGINE else Ref.gaussian_blur(img, sig)

            case "bilateral_filter":
                ss = float(params.get("sigma_s", 5.0)); sr = float(params.get("sigma_r", 0.1))
                result = out(_eng.bilateral_filter(eng(img), ss, sr)) if NATIVE_ENGINE else Ref.bilateral(img, ss, sr)

            case "median_filter":
                k = int(params.get("ksize", 3))
                result = out(_eng.median_filter(eng(img), k)) if NATIVE_ENGINE else Ref.median(img, k)

            case "sobel_edges":
                if NATIVE_ENGINE:
                    e = out(_eng.sobel_edges(eng(img), True))
                    if e.ndim == 2: e = e[..., np.newaxis]
                    result = np.repeat(e, 3, axis=-1) if e.shape[-1]==1 else e
                else:
                    result = Ref.sobel(img)

            case "canny_edges":
                lo = float(params.get("low_thresh",  0.05))
                hi = float(params.get("high_thresh", 0.15))
                sig = float(params.get("sigma", 1.0))
                if NATIVE_ENGINE:
                    e = out(_eng.canny_edges(eng(img), lo, hi, sig))
                    if e.ndim == 2: e = e[..., np.newaxis]
                    result = np.repeat(e, 3, axis=-1) if e.shape[-1]==1 else e
                else:
                    result = Ref.canny(img, lo, hi, sig)

            case "laplacian":
                sig = float(params.get("sigma", 1.0))
                blr = Ref.gaussian_blur(img, sig) if not NATIVE_ENGINE else out(_eng.gaussian_blur(eng(img), sig))
                result = Ref.sobel(blr)

            case "guided_filter":
                # Approximate via bilateral when no native guided filter
                ss  = float(params.get("radius", 5.0))
                eps = float(params.get("eps", 0.01))
                result = Ref.bilateral(img, ss, math.sqrt(eps)) if not NATIVE_ENGINE else Ref.bilateral(img, ss, math.sqrt(eps))

            case "histogram_equalization":
                result = out(_eng.histogram_equalization(eng(img))) if NATIVE_ENGINE else Ref.hist_eq(img)

            case "tone_map_reinhard":
                exp   = float(params.get("exposure", 1.0))
                gamma = float(params.get("gamma",    2.2))
                result = out(_eng.tone_map_reinhard(eng(img), exp, gamma)) if NATIVE_ENGINE else Ref.tone_reinhard(img, exp, gamma)

            case "tone_map_aces":
                exp = float(params.get("exposure", 1.0))
                result = out(_eng.tone_map_aces(eng(img), exp)) if NATIVE_ENGINE else Ref.tone_aces(img, exp)

            case "unsharp_mask":
                sig = float(params.get("sigma", 1.5)); strength = float(params.get("strength", 1.5))
                thr = float(params.get("threshold", 0.0))
                result = out(_eng.unsharp_mask(eng(img), sig, strength, thr)) if NATIVE_ENGINE else Ref.unsharp(img, sig, strength)

            case "dilate":
                k = int(params.get("ksize", 5))
                result = out(_eng.dilate(eng(img), k)) if NATIVE_ENGINE else Ref.dilate(img, k)

            case "erode":
                k = int(params.get("ksize", 5))
                result = out(_eng.erode(eng(img), k)) if NATIVE_ENGINE else Ref.erode(img, k)

            case "morph_open":
                k = int(params.get("ksize", 5))
                result = out(_eng.morph_open(eng(img), k)) if NATIVE_ENGINE else Ref.dilate(Ref.erode(img, k), k)

            case "morph_close":
                k = int(params.get("ksize", 5))
                result = out(_eng.morph_close(eng(img), k)) if NATIVE_ENGINE else Ref.erode(Ref.dilate(img, k), k)

            case "add_noise":
                sig = float(params.get("sigma", 0.05)); ntype = params.get("type", "gaussian")
                result = out(_eng.add_noise(eng(img), sig, ntype)) if NATIVE_ENGINE else Ref.add_noise(img, sig, ntype)

            case "rgb_to_hsv":
                result = out(_eng.rgb_to_hsv(eng(img))) if NATIVE_ENGINE else Ref.rgb_to_hsv(img)

            case "resize":
                w = int(params.get("width",  img.shape[1] * 2))
                h = int(params.get("height", img.shape[0] * 2))
                if NATIVE_ENGINE:
                    result = out(_eng.resize(eng(img), w, h))
                else:
                    pil    = PILImage.fromarray((img * 255).astype(np.uint8))
                    result = np.asarray(pil.resize((w, h), PILImage.LANCZOS), np.float32) / 255.0

            case "frequency_highpass":
                cutoff = float(params.get("cutoff", 0.1))
                blr    = Ref.gaussian_blur(img, cutoff * img.shape[0])
                result = np.clip(img - blr + 0.5, 0, 1)

            case "frequency_lowpass":
                cutoff = float(params.get("cutoff", 0.2))
                result = Ref.gaussian_blur(img, cutoff * img.shape[0])

            case "super_resolution":
                scale = int(params.get("scale_factor", 2))
                if NATIVE_ENGINE:
                    cfg = _eng.UpscaleConfig(); cfg.scale_factor = scale
                    result = out(_eng.super_resolution(eng(img), cfg))
                else:
                    result = Ref.super_res(img, scale)

            case "neural_denoise":
                sig = float(params.get("sigma", 25.0))
                if NATIVE_ENGINE:
                    cfg = _eng.DenoiseConfig(); cfg.sigma = sig
                    result = out(_eng.neural_denoise(eng(img), cfg))
                else:
                    result = Ref.denoise(img, sig)

            case "depth_estimation":
                if NATIVE_ENGINE:
                    result = out(_eng.depth_estimation(eng(img), colorize=True))
                    if result.ndim==2: result = result[...,np.newaxis]
                    if result.shape[-1]==1: result = np.repeat(result,3,axis=-1)
                else:
                    result = Ref.depth(img)

            case "colorize_grayscale":
                strength = float(params.get("strength", 1.0))
                result   = out(_eng.colorize_grayscale(eng(img), strength)) if NATIVE_ENGINE else Ref.colorize(img, strength)

            case _:
                raise ValueError(f"Unknown algorithm: {name!r}")

    except Exception as exc:
        log.error("Algorithm %r failed: %s", name, exc)
        log.debug(traceback.format_exc())
        result = img   # return original on failure

    elapsed = (time.perf_counter() - t0) * 1000.0
    h, w    = result.shape[:2]
    mpps    = (h * w) / (elapsed * 1000.0 + 1e-9)

    stats: dict = {
        "elapsed_ms":      round(elapsed, 2),
        "throughput_mpps": round(mpps, 3),
        "output_size":     [w, h],
        "native_engine":   NATIVE_ENGINE,
        "async_mode":      ASYNC_MODE,
        "algorithm":       name,
    }
    if NATIVE_ENGINE:
        info = _eng.system_info()
        stats.update({"simd_level": info["simd_level"],
                       "threads":    info["logical_cores"],
                       "numa_nodes": info["numa_nodes"]})

    result = np.clip(result, 0, 1).astype(np.float32)
    return result, stats

# ------------------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/system_info")
def api_system_info():
    if NATIVE_ENGINE:
        info = _eng.system_info()
    else:
        info = {
            "simd_level":    "Python-reference",
            "logical_cores":  os.cpu_count() or 1,
            "physical_cores": max(1, (os.cpu_count() or 2) // 2),
            "numa_nodes":     1,
            "numa_available": False,
            "avx512": False, "avx2": False, "fma": False,
        }
    info["native_engine"]    = NATIVE_ENGINE
    info["engine_version"]   = getattr(_eng, "__version__", "N/A") if NATIVE_ENGINE else "python-ref-2.1"
    info["async_mode"]       = ASYNC_MODE
    info["python_version"]   = sys.version.split()[0]
    info["platform"]         = sys.platform
    return jsonify(info)

@app.route("/api/process", methods=["POST"])
def api_process():
    data = request.get_json(force=True) or {}
    if "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400
    try:
        img    = _decode(data["image"])
        alg    = data.get("algorithm", "gaussian_blur")
        params = data.get("params", {})
        result, stats = run_alg(alg, img, params)
        stats["psnr"] = round(Ref.psnr(img, result), 2)
        return jsonify({"result": _encode(result), "stats": stats})
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/histogram", methods=["POST"])
def api_histogram():
    data  = request.get_json(force=True) or {}
    img   = _decode(data["image"])
    bins  = int(data.get("bins", 256))
    out   = {}
    for ci, name in enumerate(["red", "green", "blue"]):
        if ci >= img.shape[2]: break
        ch   = img[:, :, ci].ravel()
        hist, edges = np.histogram(ch, bins=bins, range=(0, 1))
        out[name] = {"bins":  hist.tolist(), "edges": edges.tolist(),
                     "mean":  float(ch.mean()), "std":  float(ch.std()),
                     "min":   float(ch.min()),  "max":  float(ch.max())}
    return jsonify(out)

@app.route("/api/palette", methods=["POST"])
def api_palette():
    data = request.get_json(force=True) or {}
    img  = _decode(data["image"])
    k    = int(data.get("k", 8))
    if NATIVE_ENGINE:
        pal = _eng.extract_palette(_to_eng(img), k)
        colors = [{"r": float(c[0]), "g": float(c[1]), "b": float(c[2]),
                   "weight": float(w)} for c, w in zip(pal.colors, pal.weights)]
    else:
        colors = Ref.palette(img, k)
    return jsonify({"palette": colors})

@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = request.get_json(force=True) or {}
    img  = _decode(data["image"])
    algs = data.get("algorithms", [
        "gaussian_blur","bilateral_filter","median_filter","sobel_edges",
        "histogram_equalization","tone_map_aces","dilate","erode","unsharp_mask"])
    results = {}
    for alg in algs:
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            run_alg(alg, img, {})
            times.append((time.perf_counter() - t0) * 1000)
        results[alg] = {
            "mean_ms": round(float(np.mean(times)), 2),
            "min_ms":  round(float(np.min(times)),  2),
            "max_ms":  round(float(np.max(times)),  2),
            "std_ms":  round(float(np.std(times)),  2),
        }
    return jsonify({"benchmark": results, "native_engine": NATIVE_ENGINE})

# ------------------------------------------------------------------------------
# WebSocket - real-time frame processing & pipeline
# ------------------------------------------------------------------------------
_pipelines: Dict[str, bool] = {}
_pip_lock = threading.Lock()

@socketio.on("connect")
def on_connect():
    log.info("WS connect: %s", request.sid)
    emit("system_info", {
        "native_engine": NATIVE_ENGINE,
        "simd_level":    _eng.system_info()["simd_level"] if NATIVE_ENGINE else "python-ref",
        "async_mode":    ASYNC_MODE,
    })

@socketio.on("disconnect")
def on_disconnect():
    with _pip_lock: _pipelines.pop(request.sid, None)
    log.info("WS disconnect: %s", request.sid)

@socketio.on("process_frame")
def on_frame(data):
    try:
        img    = _decode(data["image"])
        alg    = data.get("algorithm", "gaussian_blur")
        params = data.get("params", {})
        result, stats = run_alg(alg, img, params)
        emit("frame_result", {
            "result":   _encode(result, "JPEG"),
            "stats":    stats,
            "frame_id": data.get("frame_id", 0),
        })
    except Exception as e:
        emit("error", {"message": str(e)})

@socketio.on("start_pipeline")
def on_start_pipeline(data):
    sid    = request.sid
    stages = data.get("stages", [])
    with _pip_lock: _pipelines[sid] = True

    def _run():
        try:
            current = _decode(data["image"])
            for i, stage in enumerate(stages):
                with _pip_lock:
                    if not _pipelines.get(sid, False): break
                alg    = stage.get("algorithm", "gaussian_blur")
                params = stage.get("params", {})
                current, stats = run_alg(alg, current, params)
                socketio.emit("pipeline_stage", {
                    "stage":        i,
                    "total_stages": len(stages),
                    "result":       _encode(current, "JPEG"),
                    "stats":        stats,
                    "progress":     (i + 1) / len(stages),
                }, to=sid)
            socketio.emit("pipeline_complete", {
                "result":  _encode(current),
                "message": f"Pipeline of {len(stages)} stages complete",
            }, to=sid)
        except Exception as e:
            socketio.emit("error", {"message": str(e)}, to=sid)

    threading.Thread(target=_run, daemon=True).start()

@socketio.on("stop_pipeline")
def on_stop():
    with _pip_lock: _pipelines[request.sid] = False
    emit("pipeline_stopped", {})

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "0") == "1"
    host  = os.environ.get("HOST", "0.0.0.0")

    log.info("Starting server on http://localhost:%d  (debug=%s)", port, debug)
    log.info("Async mode: %s | Native C++: %s", ASYNC_MODE, NATIVE_ENGINE)

    # Windows + gevent: allow_unsafe_werkzeug needed when debug=False
    socketio.run(app, host=host, port=port, debug=debug,
                 use_reloader=False, allow_unsafe_werkzeug=True)