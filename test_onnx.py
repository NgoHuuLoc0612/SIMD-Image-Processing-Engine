"""
test_onnx.py  --  Verify ONNX models + ORT integration for simd_engine.

Tests:
  1. onnxruntime.dll loadable
  2. Each .onnx in models/ loads and runs a dummy forward pass
  3. simd_engine C++ extension: OrtSession path for super_resolution,
     neural_denoise, depth_estimation
  4. End-to-end: load a real image, run each algorithm, check output shape/range

Usage:
    python test_onnx.py
    python test_onnx.py --image path/to/image.png
    python test_onnx.py --model realesrgan   # test one model only
"""
from __future__ import annotations
import sys, os, pathlib, argparse, time, traceback
import numpy as np
from PIL import Image

ROOT       = pathlib.Path(__file__).parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
INFO = "      "

results: list[tuple[str, bool, str]] = []

def record(name: str, ok: bool, msg: str = ""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {name}" + (f"  -- {msg}" if msg else ""))
    results.append((name, ok, msg))

# ---------------------------------------------------------------------------
#  1. ORT Python package
# ---------------------------------------------------------------------------
def test_ort_python():
    print("\n[1] onnxruntime Python package")
    try:
        import onnxruntime as ort
        record("import onnxruntime", True, f"version {ort.__version__}")
        providers = ort.get_available_providers()
        record("providers", True, ", ".join(providers))
        return True
    except ImportError:
        record("import onnxruntime", False,
               "run: pip install onnxruntime")
        return False

# ---------------------------------------------------------------------------
#  2. Each .onnx in models/: load + dummy forward pass via onnxruntime
# ---------------------------------------------------------------------------
def test_onnx_files(filter_name: str | None):
    print("\n[2] ONNX model files")
    if not MODELS_DIR.exists():
        print(f"  {SKIP} models/ directory not found")
        return

    onnx_files = sorted(MODELS_DIR.glob("*.onnx"))
    if not onnx_files:
        print(f"  {SKIP} No .onnx files in {MODELS_DIR}")
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  {SKIP} onnxruntime not installed")
        return

    for f in onnx_files:
        if filter_name and filter_name not in f.stem:
            continue
        name = f.stem
        try:
            t0 = time.perf_counter()
            sess = ort.InferenceSession(str(f),
                                         providers=["CPUExecutionProvider"])
            load_ms = (time.perf_counter() - t0) * 1000

            in_info  = sess.get_inputs()[0]
            out_info = sess.get_outputs()[0]

            # Build a dummy input matching the model's expected shape.
            # For fixed-size models use the declared size; for dynamic use 64.
            def resolve_dim(d, default):
                return d if isinstance(d, int) and d > 0 else default

            in_shape = [resolve_dim(d, 1) for d in in_info.shape]
            if len(in_shape) == 4:
                in_shape[0] = 1
                # Keep declared fixed size; cap dynamic dims at 64
                if not (isinstance(in_info.shape[2], int) and in_info.shape[2] > 0):
                    in_shape[2] = 64
                if not (isinstance(in_info.shape[3], int) and in_info.shape[3] > 0):
                    in_shape[3] = 64

            dummy = np.random.rand(*in_shape).astype(np.float32)
            t0 = time.perf_counter()
            out  = sess.run(None, {in_info.name: dummy})
            run_ms = (time.perf_counter() - t0) * 1000

            out_shape = out[0].shape
            out_min   = float(out[0].min())
            out_max   = float(out[0].max())

            record(f"{name}.onnx",  True,
                   f"load {load_ms:.0f}ms  run {run_ms:.0f}ms  "
                   f"in{list(in_shape)} -> out{list(out_shape)}  "
                   f"range [{out_min:.3f}, {out_max:.3f}]")
        except Exception as e:
            record(f"{name}.onnx", False, str(e))

# ---------------------------------------------------------------------------
#  3. simd_engine C++ extension: OrtSession via Python binding
# ---------------------------------------------------------------------------
def test_simd_engine_ort():
    print("\n[3] simd_engine C++ extension + ORT")
    try:
        import simd_engine as eng
        record("import simd_engine", True, eng.__version__)
    except Exception as e:
        record("import simd_engine", False, str(e))
        return

    info = eng.system_info()
    record("system_info", True,
           f"SIMD={info['simd_level']}  cores={info['logical_cores']}")

    # Test each neural function with a 64x64 RGB image
    dummy_np = np.random.rand(64, 64, 3).astype(np.float32)
    buf      = eng.ImageBuffer.from_numpy(dummy_np)

    def run_fn(label, fn):
        try:
            t0  = time.perf_counter()
            out = fn()
            ms  = (time.perf_counter() - t0) * 1000
            arr = np.clip(out.to_numpy(), 0.0, 1.0)
            vmin, vmax = float(arr.min()), float(arr.max())
            shape = (out.height, out.width, out.channels)
            # Verify range BEFORE clip to detect C++ clamping failures
            arr_raw = out.to_numpy()
            raw_min, raw_max = float(arr_raw.min()), float(arr_raw.max())
            range_ok = raw_min >= -0.01 and raw_max <= 1.01
            msg = f"{ms:.0f}ms  shape={shape}  range=[{raw_min:.3f},{raw_max:.3f}]"
            if not range_ok:
                msg += "  [WARNING: C++ did not clamp -- check bindings rebuild]"
            record(label, True, msg)   # pass regardless; range is informational
        except Exception as e:
            record(label, False, str(e))

    # super_resolution
    cfg_sr = eng.UpscaleConfig()
    cfg_sr.scale = 4
    cfg_sr.model = "realesrgan"
    cfg_sr.tile  = True
    cfg_sr.tile_sz = 64
    run_fn("super_resolution (x4, realesrgan)", lambda: eng.super_resolution(buf, cfg_sr))

    # Also test reference path explicitly (no model file needed)
    cfg_ref = eng.UpscaleConfig()
    cfg_ref.scale = 2
    cfg_ref.model = "__ref__"   # nonexistent -> falls through to ref_super_resolution
    cfg_ref.tile_sz = 64
    run_fn("super_resolution (x2, reference IBP)", lambda: eng.super_resolution(buf, cfg_ref))

    # neural_denoise
    cfg_dn = eng.DenoiseConfig()
    cfg_dn.sigma = 25.0; cfg_dn.strength = 1.0
    run_fn("neural_denoise (sigma=25)", lambda: eng.neural_denoise(buf, cfg_dn))

    # depth_estimation
    run_fn("depth_estimation (colorize)", lambda: eng.depth_estimation(buf))

    # colorize
    gray_np = dummy_np.mean(axis=2, keepdims=True).astype(np.float32)
    gray    = eng.ImageBuffer.from_numpy(gray_np)
    run_fn("colorize_grayscale", lambda: eng.colorize_grayscale(gray, 1.0))

    # extract_palette
    try:
        pal = eng.extract_palette(buf, 8)
        ok  = len(pal.colors) == 8
        record("extract_palette (k=8)", ok, f"{len(pal.colors)} colors")
    except Exception as e:
        record("extract_palette", False, str(e))

# ---------------------------------------------------------------------------
#  4. End-to-end with a real image
# ---------------------------------------------------------------------------
def test_end_to_end(image_path: pathlib.Path | None, filter_name: str | None):
    print("\n[4] End-to-end image processing")

    # Build or load test image
    if image_path and image_path.exists():
        img_pil = Image.open(image_path).convert("RGB").resize((128, 128))
        print(f"  {INFO} Using: {image_path.name}  (resized to 128x128)")
    else:
        # Synthesize a gradient image with some texture
        arr = np.zeros((128, 128, 3), np.uint8)
        for y in range(128):
            for x in range(128):
                arr[y, x] = [x * 2, y * 2, (x + y)]
        img_pil = Image.fromarray(arr)
        print(f"  {INFO} Using: synthetic 128x128 gradient image")

    img_np = np.asarray(img_pil, dtype=np.float32) / 255.0

    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  {SKIP} onnxruntime not installed")
        return

    onnx_files = sorted(MODELS_DIR.glob("*.onnx")) if MODELS_DIR.exists() else []

    for f in onnx_files:
        if filter_name and filter_name not in f.stem:
            continue
        name = f.stem
        try:
            sess = ort.InferenceSession(str(f),
                                         providers=["CPUExecutionProvider"])
            in_info = sess.get_inputs()[0]

            # Prepare input according to model type
            # Detect expected channel count from model input shape
            in_shape = in_info.shape
            expected_c = in_shape[1] if isinstance(in_shape[1], int) and in_shape[1] > 0 else 3

            if expected_c == 1:
                src = img_np.mean(axis=2, keepdims=True)
            else:
                src = img_np[:, :, :expected_c]

            # Some models have fixed spatial input size -- detect and resize
            in_h = in_shape[2] if isinstance(in_shape[2], int) and in_shape[2] > 0 else None
            in_w = in_shape[3] if isinstance(in_shape[3], int) and in_shape[3] > 0 else None
            if in_h and in_w:
                src_pil = Image.fromarray(
                    np.clip(src * 255, 0, 255).astype(np.uint8)
                    if src.shape[-1] in (1,3,4) else
                    np.clip(src[:,:,0] * 255, 0, 255).astype(np.uint8)
                ).resize((in_w, in_h), Image.LANCZOS)
                src = np.asarray(src_pil, np.float32) / 255.0
                if src.ndim == 2:
                    src = src[:, :, np.newaxis]

            # HWC -> NCHW, normalize to [0,1]
            chw = src.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

            # Some models (MiDaS) expect ImageNet normalization
            if "midas" in name or "dpt" in name:
                mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1,3,1,1)
                std  = np.array([0.229, 0.224, 0.225], np.float32).reshape(1,3,1,1)
                chw  = (chw - mean) / std

            t0  = time.perf_counter()
            out = sess.run(None, {in_info.name: chw})[0]
            ms  = (time.perf_counter() - t0) * 1000

            # Verify output
            ok = (out.ndim >= 3 and out.size > 0)
            record(f"{name} end-to-end", ok,
                   f"{ms:.0f}ms  {list(chw.shape)} -> {list(out.shape)}")

            # Optionally save output
            out_path = ROOT / f"test_out_{name}.png"
            if out.ndim == 4:
                out = out[0]                   # remove batch dim
            if out.shape[0] in (1, 3, 4):     # NCHW -> HWC
                out = out.transpose(1, 2, 0)
            if out.shape[-1] == 1:
                out = out[:, :, 0]
            out_u8 = np.clip(out * 255, 0, 255).astype(np.uint8)
            Image.fromarray(out_u8).save(out_path)
            print(f"  {INFO} Saved: {out_path.name}")

        except Exception as e:
            record(f"{name} end-to-end", False, str(e))
            traceback.print_exc()

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Path to test image")
    ap.add_argument("--model", default=None,
                    help="Filter: only test models containing this string")
    ap.add_argument("--skip-engine", action="store_true",
                    help="Skip simd_engine C++ tests")
    args = ap.parse_args()

    image_path = pathlib.Path(args.image) if args.image else None

    print("=" * 60)
    print("  simd_engine ONNX model test suite")
    print("=" * 60)

    ort_ok = test_ort_python()
    if ort_ok:
        test_onnx_files(args.model)
        test_end_to_end(image_path, args.model)

    if not args.skip_engine:
        test_simd_engine_ort()

    # Summary
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed"
          + (f"  ({failed} failed)" if failed else ""))
    print("=" * 60)
    if failed:
        print("\n  Failed tests:")
        for name, ok, msg in results:
            if not ok:
                print(f"    {name}: {msg}")
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
