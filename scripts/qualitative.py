"""Side-by-side qualitative reconstructions + IMU-predicted flow overlay.

For one Aria frame pair, runs all three codecs at the mid-rate operating point and
produces:
    figures/qualitative.png   — frames panel
    figures/imu_flow.png      — flow visualization

Usage:
    python scripts/qualitative.py --data data/aria_proc --pair-idx 200 \\
        --iframe-ckpt runs/iframe-l0.0067/best.pt \\
        --pframe-imu-ckpt runs/pframe-imu-l0.0067/best.pt \\
        --pframe-noimu-ckpt runs/pframe-noimu-l0.0067/best.pt \\
        --out figures
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ego_codec.data import AriaPairDataset
from ego_codec.models import IMUConditionedPCodec, MeanScaleHyperprior
from ego_codec.models.blocks import warp_with_flow


def load_iframe(ckpt_path: str, device) -> MeanScaleHyperprior:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck.get("args", {}) or {}
    m = MeanScaleHyperprior(n=a.get("n", 128), m=a.get("m", 192)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def load_pframe(ckpt_path: str, device) -> IMUConditionedPCodec:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck.get("args", {}) or {}
    m = IMUConditionedPCodec(
        n=a.get("n", 128), m=a.get("m", 192), imu_window=a.get("imu_samples", 50)
    ).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) float in [0,1] -> (H,W,C) uint8."""
    arr = t.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = float(((x - y) ** 2).mean())
    if mse < 1e-10:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def flow_to_rgb(flow: torch.Tensor) -> np.ndarray:
    """(2,H,W) flow in pixels -> HSV-encoded RGB. Magnitude->value, angle->hue."""
    fx = flow[0].cpu().numpy()
    fy = flow[1].cpu().numpy()
    mag = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx)
    h = (ang + math.pi) / (2 * math.pi)  # [0,1]
    v = np.clip(mag / max(mag.max(), 1e-6), 0, 1)
    s = np.ones_like(h)
    import colorsys
    rgb = np.stack(np.vectorize(colorsys.hsv_to_rgb)(h, s, v), axis=-1)
    return (rgb * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pair-idx", type=int, default=200)
    ap.add_argument("--iframe-ckpt", required=True)
    ap.add_argument("--pframe-imu-ckpt", required=True)
    ap.add_argument("--pframe-noimu-ckpt", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = AriaPairDataset(args.data, crop_size=args.crop, imu_samples=50)
    x_prev, x_curr, imu = ds[args.pair_idx]
    x_prev = x_prev.unsqueeze(0).to(device)
    x_curr = x_curr.unsqueeze(0).to(device)
    imu = imu.unsqueeze(0).to(device)

    iframe = load_iframe(args.iframe_ckpt, device)
    pframe_imu = load_pframe(args.pframe_imu_ckpt, device)
    pframe_noimu = load_pframe(args.pframe_noimu_ckpt, device)

    with torch.no_grad():
        # I-frame on x_curr
        iframe_out = iframe(x_curr)
        x_iframe = iframe_out.x_hat.clamp(0, 1)
        bpp_iframe = float(iframe_out.bpp)
        psnr_iframe = psnr(x_iframe[0], x_curr[0])

        # P-frame, IMU
        pi_out, pi_pred = pframe_imu(x_prev, x_curr, imu, use_imu=True)
        x_pi = pi_out.x_hat.clamp(0, 1)
        bpp_pi = float(pi_out.bpp)
        psnr_pi = psnr(x_pi[0], x_curr[0])

        # IMU-predicted flow (for visualization)
        flow = pframe_imu.warp_predictor(imu, target_hw=x_curr.shape[-2:])
        x_pred_imu = warp_with_flow(x_prev, flow).clamp(0, 1)

        # P-frame, no IMU
        pn_out, _ = pframe_noimu(x_prev, x_curr, imu, use_imu=False)
        x_pn = pn_out.x_hat.clamp(0, 1)
        bpp_pn = float(pn_out.bpp)
        psnr_pn = psnr(x_pn[0], x_curr[0])

    # === Frames panel ===
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    panels = [
        (x_prev[0], "x_{t-1} (previous)", None),
        (x_curr[0], "x_t (target)", None),
        (x_iframe[0], "I-frame (intra only)", (bpp_iframe, psnr_iframe)),
        (x_pn[0], "P-frame, no IMU", (bpp_pn, psnr_pn)),
        (x_pi[0], "P-frame, IMU (ours)", (bpp_pi, psnr_pi)),
    ]
    for ax, (img, title, metrics) in zip(axes, panels):
        ax.imshow(to_uint8(img))
        if metrics is not None:
            bpp, ps = metrics
            ax.set_title(f"{title}\n{bpp:.3f} bpp · {ps:.2f} dB")
        else:
            ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Egocentric-native compression — qualitative reconstructions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {out_dir / 'qualitative.png'}")

    # === IMU-flow panel ===
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4.4))
    axes2[0].imshow(to_uint8(x_prev[0]));  axes2[0].set_title("x_{t-1}");        axes2[0].axis("off")
    axes2[1].imshow(to_uint8(x_curr[0]));  axes2[1].set_title("x_t");            axes2[1].axis("off")
    axes2[2].imshow(flow_to_rgb(flow[0])); axes2[2].set_title("IMU-predicted flow\n(hue=direction, value=magnitude)"); axes2[2].axis("off")
    axes2[3].imshow(to_uint8(x_pred_imu[0])); axes2[3].set_title("warp(x_{t-1}, flow)\n— pre-residual prediction"); axes2[3].axis("off")
    fig2.suptitle("IMU-conditioned motion compensation", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(out_dir / "imu_flow.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {out_dir / 'imu_flow.png'}")


if __name__ == "__main__":
    main()
