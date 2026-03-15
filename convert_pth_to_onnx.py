"""
convert_pth_to_onnx.py
Converts RealESRGAN_x4plus.pth -> realesrgan.onnx

Requirements:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install basicsr

Usage:
    python convert_pth_to_onnx.py
    python convert_pth_to_onnx.py --pth path/to/RealESRGAN_x4plus.pth --out models/realesrgan.onnx
    python convert_pth_to_onnx.py --pth RealESRGAN_x4plus_anime_6B.pth --anime --out models/realesrgan_anime.onnx
"""
from __future__ import annotations
import argparse, pathlib, sys, subprocess

ROOT = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
def ensure_deps() -> bool:
    missing = []
    try:
        import torch  # noqa
    except ImportError:
        missing.append(("torch", ["torch", "--index-url",
                                   "https://download.pytorch.org/whl/cpu"]))
    try:
        import onnx  # noqa
    except ImportError:
        missing.append(("onnx", ["onnx"]))
    try:
        import basicsr  # noqa
    except ImportError:
        missing.append(("basicsr", ["basicsr"]))

    if not missing:
        return True

    for name, pkg in missing:
        print(f"Installing {name} ...")
        ret = subprocess.call([sys.executable, "-m", "pip", "install"] + pkg + ["-q"])
        if ret != 0:
            print(f"[FAIL] Could not install {name}")
            return False
    return True


# ---------------------------------------------------------------------------
# Minimal RRDBNet matching xinntao's exact weight keys.
# Source: https://github.com/xinntao/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
# Reproduced here so basicsr is optional (we try basicsr first, fall back to this).
# ---------------------------------------------------------------------------
def build_rrdbnet_manual(num_in_ch=3, num_out_ch=3,
                          num_feat=64, num_block=23, num_grow_ch=32, scale=4):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat,              num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch,   num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2*num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3*num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4*num_grow_ch, num_feat,    3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = ResidualDenseBlock()
            self.rdb2 = ResidualDenseBlock()
            self.rdb3 = ResidualDenseBlock()
        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu     = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        def forward(self, x):
            feat      = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat      = feat + body_feat
            feat      = self.lrelu(self.conv_up1(
                            F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat      = self.lrelu(self.conv_up2(
                            F.interpolate(feat, scale_factor=2, mode='nearest')))
            out       = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    return RRDBNet()


def build_rrdbnet_anime(num_in_ch=3, num_out_ch=3,
                         num_feat=64, num_block=6, num_grow_ch=32, scale=4):
    """6-block anime variant (RealESRGAN_x4plus_anime_6B)."""
    return build_rrdbnet_manual(num_in_ch, num_out_ch, num_feat, num_block,
                                 num_grow_ch, scale)


def load_model(pth_path: pathlib.Path, anime: bool):
    import torch

    # Try basicsr first (exact same arch, cleaner API)
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        if anime:
            model = RRDBNet(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        print("  [arch] basicsr.archs.rrdbnet_arch.RRDBNet")
    except Exception:
        print("  [arch] manual RRDBNet (basicsr unavailable)")
        if anime:
            model = build_rrdbnet_anime()
        else:
            model = build_rrdbnet_manual()

    # Load weights -- pth may be a dict with 'params_ema', 'params', or bare state_dict
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "params_ema" in ckpt:
            state = ckpt["params_ema"]
        elif "params" in ckpt:
            state = ckpt["params"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    # Strip "module." prefix if model was saved with DataParallel
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def convert(pth_path: pathlib.Path, onnx_path: pathlib.Path,
            anime: bool, tile: int, opset: int) -> bool:
    import torch

    print(f"  Loading {pth_path.name} ...")
    try:
        model = load_model(pth_path, anime)
    except Exception as e:
        print(f"  [FAIL] load: {e}")
        import traceback; traceback.print_exc()
        return False

    print(f"  Exporting to {onnx_path.name}  (opset={opset}, tile={tile}x{tile}) ...")
    dummy = torch.zeros(1, 3, tile, tile)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # torch 2.5+ uses onnxscript by default; force legacy TorchScript exporter
    import inspect
    kw = dict(
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False

    try:
        torch.onnx.export(model, dummy, str(onnx_path), **kw)
    except Exception as e:
        print(f"  [FAIL] export: {e}")
        import traceback; traceback.print_exc()
        return False

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  [OK]   {onnx_path}  ({size_mb:.1f} MB)")

    # Quick shape sanity check
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path),
                                     providers=["CPUExecutionProvider"])
        import numpy as np
        x = np.zeros((1, 3, 64, 64), dtype=np.float32)
        y = sess.run(None, {"input": x})[0]
        expected_h = 64 * 4
        if y.shape[2] == expected_h:
            print(f"  [TEST] ORT run OK: {x.shape} -> {y.shape}")
        else:
            print(f"  [WARN] Unexpected output shape: {y.shape}")
    except ImportError:
        print("  [INFO] onnxruntime not installed -- skipping ORT test run")
    except Exception as e:
        print(f"  [WARN] ORT test: {e}")

    return True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Convert RealESRGAN .pth to .onnx")
    ap.add_argument("--pth",   default=None,
                    help="Path to .pth file (default: auto-detect in cwd)")
    ap.add_argument("--out",   default=None,
                    help="Output .onnx path (default: models/<name>.onnx)")
    ap.add_argument("--anime", action="store_true",
                    help="Use 6-block anime architecture")
    ap.add_argument("--tile",  type=int, default=64,
                    help="Dummy input tile size for tracing (default: 64)")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (default: 17)")
    args = ap.parse_args()

    print("=" * 60)
    print("  RealESRGAN .pth -> .onnx converter")
    print("=" * 60)

    # Resolve pth path
    if args.pth:
        pth_path = pathlib.Path(args.pth)
    else:
        # Auto-detect in cwd
        candidates = sorted(pathlib.Path(".").glob("*.pth"))
        if not candidates:
            print("\n[ERROR] No .pth file found in current directory.")
            print("Download from:")
            print("  https://github.com/xinntao/Real-ESRGAN/releases/download/"
                  "v0.1.0/RealESRGAN_x4plus.pth")
            sys.exit(1)
        pth_path = candidates[0]
        print(f"\n  Auto-detected: {pth_path}")

    if not pth_path.exists():
        print(f"\n[ERROR] {pth_path} not found")
        sys.exit(1)

    # Resolve output path
    if args.out:
        onnx_path = pathlib.Path(args.out)
    else:
        name = "realesrgan_anime.onnx" if args.anime else "realesrgan.onnx"
        onnx_path = ROOT / "models" / name

    print(f"\n  Input : {pth_path}")
    print(f"  Output: {onnx_path}")
    print(f"  Arch  : {'anime 6B' if args.anime else 'x4plus 23B'}")
    print(f"  Opset : {args.opset}")

    # Install deps
    print("\n[1/2] Checking dependencies ...")
    if not ensure_deps():
        sys.exit(1)
    print("  [OK]")

    # Convert
    print("\n[2/2] Converting ...")
    ok = convert(pth_path, onnx_path, args.anime, args.tile, args.opset)
    if ok:
        print(f"\n  Done. Place {onnx_path} in the models/ directory next to server.py")
    else:
        print("\n  [FAIL] Conversion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
