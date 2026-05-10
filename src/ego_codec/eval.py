"""Evaluate a sweep of checkpoints and emit RD-curve data + plots.

Loads multiple {tag, ckpt_path} entries, runs them on a held-out Aria split, computes
mean PSNR and bpp per checkpoint, and plots all conditions on one figure.

Usage:
    python -m ego_codec.eval --data data/aria_proc_val --out figures \\
        --condition iframe-aria runs/iframe-l0.0018/best.pt \\
        --condition iframe-aria runs/iframe-l0.0067/best.pt \\
        --condition iframe-aria runs/iframe-l0.0250/best.pt \\
        --condition pframe-imu  runs/pframe-imu-l0.0018/best.pt \\
        --condition pframe-imu  runs/pframe-imu-l0.0067/best.pt \\
        --condition pframe-noimu runs/pframe-noimu-l0.0067/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from ego_codec.data import AriaFrameDataset, AriaPairDataset
from ego_codec.models import IMUConditionedPCodec, MeanScaleHyperprior


@torch.no_grad()
def eval_iframe(model: MeanScaleHyperprior, loader, device) -> tuple[float, float]:
    model.eval()
    psnrs, bpps = [], []
    for x in loader:
        x = x.to(device)
        out = model(x)
        mse = torch.mean((out.x_hat.clamp(0, 1) - x) ** 2).item()
        if mse < 1e-12:
            mse = 1e-12
        psnrs.append(10 * math.log10(1.0 / mse))
        bpps.append(out.bpp.item())
    return sum(psnrs) / len(psnrs), sum(bpps) / len(bpps)


@torch.no_grad()
def eval_pframe(model: IMUConditionedPCodec, loader, device, use_imu: bool) -> tuple[float, float]:
    model.eval()
    psnrs, bpps = [], []
    for x_prev, x_curr, imu in loader:
        x_prev = x_prev.to(device)
        x_curr = x_curr.to(device)
        imu = imu.to(device)
        out, _ = model(x_prev, x_curr, imu, use_imu=use_imu)
        mse = torch.mean((out.x_hat - x_curr) ** 2).item()
        if mse < 1e-12:
            mse = 1e-12
        psnrs.append(10 * math.log10(1.0 / mse))
        bpps.append(out.bpp.item())
    return sum(psnrs) / len(psnrs), sum(bpps) / len(bpps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--condition",
        action="append",
        nargs=2,
        metavar=("TAG", "CKPT"),
        help="Repeatable. TAG must be one of: iframe-aria, iframe-generic, pframe-imu, pframe-noimu",
        default=[],
    )
    ap.add_argument(
        "--x265-json",
        default=None,
        help="Optional path to x265 baseline JSON from scripts/eval_x265.py — overlaid on the RD plot.",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iframe_ds = AriaFrameDataset(args.data, crop_size=256)
    pframe_ds = AriaPairDataset(args.data, crop_size=256)
    iframe_loader = DataLoader(iframe_ds, batch_size=args.batch_size, num_workers=args.workers)
    pframe_loader = DataLoader(pframe_ds, batch_size=args.batch_size, num_workers=args.workers)

    results: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for tag, ckpt_path in args.condition:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Read n, m, imu_samples from the checkpoint's saved training args so the model
        # is instantiated to match the trained weight shapes.
        ck_args = ckpt.get("args", {}) or {}
        n = ck_args.get("n", 128)
        m = ck_args.get("m", 192)
        if tag.startswith("iframe"):
            model = MeanScaleHyperprior(n=n, m=m).to(device)
            model.load_state_dict(ckpt["model"])
            psnr, bpp = eval_iframe(model, iframe_loader, device)
        else:
            imu_window = ck_args.get("imu_samples", 50)
            model = IMUConditionedPCodec(n=n, m=m, imu_window=imu_window).to(device)
            model.load_state_dict(ckpt["model"])
            use_imu = tag == "pframe-imu"
            psnr, bpp = eval_pframe(model, pframe_loader, device, use_imu=use_imu)
        print(f"{tag}: bpp={bpp:.3f} psnr={psnr:.2f}  ({ckpt_path})")
        results[tag].append((bpp, psnr))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    style = {
        "iframe-generic": dict(marker="o", linestyle="--", label="I-frame, trained on COCO"),
        "iframe-aria":    dict(marker="o", linestyle="-",  label="I-frame, trained on Aria"),
        "pframe-noimu":   dict(marker="s", linestyle="--", label="P-frame, no IMU (zero motion)"),
        "pframe-imu":     dict(marker="s", linestyle="-",  label="P-frame, IMU-conditioned (ours)"),
    }
    for tag, points in results.items():
        points.sort()
        bpps, psnrs = zip(*points)
        ax.plot(bpps, psnrs, **style.get(tag, {"label": tag}))

    # Overlay x265 baseline if provided.
    if args.x265_json:
        x265_data = json.loads(Path(args.x265_json).read_text())
        x265_points = sorted((v["bpp"], v["psnr"]) for v in x265_data.values())
        if x265_points:
            bpps, psnrs = zip(*x265_points)
            ax.plot(bpps, psnrs, marker="x", linestyle=":", color="black", label="x265 (libx265)")
    ax.set_xlabel("Bits per pixel (bpp)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Egocentric-native compression — rate-distortion")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rd_curve.png", dpi=160)
    fig.savefig(out_dir / "rd_curve.pdf")

    (out_dir / "rd_results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_dir / 'rd_curve.png'}")


if __name__ == "__main__":
    main()
