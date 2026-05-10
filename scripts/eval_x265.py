"""x265 baseline for the RD comparison.

For each pair of consecutive Aria frames (x_prev, x_curr), encode the *pair* as a 2-frame
video with libx265 at multiple CRF values, and measure (bpp, PSNR) for the P-frame
reconstruction. This gives a fair industry baseline to plot against pframe-imu /
pframe-noimu on the same axes.

Why 2-frame pairs and not full sequences:
  Our learned codec is evaluated on consecutive (prev, curr) pairs (one P-frame predicted
  from one I-frame). To match this, x265 is invoked with --keyint=2 so the second frame is
  forced to be a P-frame off the first frame. Numbers from longer GOPs would understate
  x265 since x265 is much better at long-context coding than our ablation harness.

Requires ffmpeg with libx265 in $PATH (`brew install ffmpeg` or `apt install ffmpeg`).

Usage:
    python scripts/eval_x265.py --data data/aria_proc_val --out figures/x265_baseline.json \\
        --crfs 18 23 28 33 --max-pairs 200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ego_codec.data.aria_loader import AriaPairDataset  # noqa: E402


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH. Install ffmpeg (with libx265).")
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    if "libx265" not in out.stdout:
        raise SystemExit("ffmpeg is installed but missing libx265 support.")


def encode_pair_x265(prev: np.ndarray, curr: np.ndarray, crf: int) -> tuple[float, float]:
    """Encode (prev, curr) as a 2-frame x265 stream. Return (bpp_for_pframe, psnr_pframe)."""
    h, w, _ = prev.shape
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        Image.fromarray(prev).save(td / "f0.png")
        Image.fromarray(curr).save(td / "f1.png")
        encoded = td / "out.265"
        # Force P-frame on frame 1: keyint=2, no scenecut.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "10",
            "-i", str(td / "f%d.png"),
            "-c:v", "libx265",
            "-preset", "medium",
            "-x265-params", f"crf={crf}:keyint=2:min-keyint=2:scenecut=0:bframes=0:lossless=0",
            "-pix_fmt", "yuv420p",
            str(encoded),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Measure compressed size, but we want only the P-frame bits.
        # Trick: encode only-I (keyint=1) at same CRF as a delta for I-frame size, subtract.
        encoded_i = td / "out_i.265"
        cmd_i = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(td / "f0.png"),
            "-c:v", "libx265",
            "-preset", "medium",
            "-x265-params", f"crf={crf}:keyint=1:lossless=0",
            "-pix_fmt", "yuv420p",
            str(encoded_i),
        ]
        subprocess.run(cmd_i, check=True, capture_output=True)

        total_bits = encoded.stat().st_size * 8
        i_bits = encoded_i.stat().st_size * 8
        p_bits = max(total_bits - i_bits, 1)
        bpp = p_bits / (h * w)

        # Decode the 2-frame stream and measure PSNR on frame 1.
        decoded = td / "decoded_%d.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(encoded), str(decoded)],
            check=True, capture_output=True,
        )
        recon = np.asarray(Image.open(td / "decoded_2.png").convert("RGB")).astype(np.float32)
        ref = curr.astype(np.float32)
        mse = float(np.mean((recon - ref) ** 2))
        if mse < 1e-6:
            mse = 1e-6
        psnr = 10.0 * math.log10((255.0**2) / mse)
        return bpp, psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures/x265_baseline.json")
    ap.add_argument("--crfs", type=int, nargs="+", default=[18, 23, 28, 33])
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--max-pairs", type=int, default=200)
    args = ap.parse_args()

    _check_ffmpeg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    ds = AriaPairDataset(args.data, crop_size=args.crop, imu_samples=50)
    indices = list(range(min(len(ds), args.max_pairs)))

    results = {}
    for crf in args.crfs:
        bpps, psnrs = [], []
        for i in indices:
            x_prev, x_curr, _ = ds[i]
            prev_np = (x_prev.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            curr_np = (x_curr.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            try:
                bpp, psnr = encode_pair_x265(prev_np, curr_np, crf)
            except subprocess.CalledProcessError:
                continue
            bpps.append(bpp)
            psnrs.append(psnr)
        if bpps:
            mean_bpp = sum(bpps) / len(bpps)
            mean_psnr = sum(psnrs) / len(psnrs)
            results[f"x265-crf{crf}"] = {"bpp": mean_bpp, "psnr": mean_psnr, "n": len(bpps)}
            print(f"x265 CRF={crf}: bpp={mean_bpp:.3f} psnr={mean_psnr:.2f} n={len(bpps)}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
