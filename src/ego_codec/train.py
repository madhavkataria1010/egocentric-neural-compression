"""Training entry point — tuned to saturate a single A40 (46 GB).

Three modes:

    python -m ego_codec.train iframe        # train MeanScaleHyperprior on AriaFrameDataset
    python -m ego_codec.train pframe        # train IMUConditionedPCodec (loads I-frame ckpt)
    python -m ego_codec.train pframe-noimu  # ablation: pframe with use_imu=False

Defaults are sized for A40-class GPUs:
- bf16 autocast (Ampere+ supports it natively)
- batch 64, crop 256, n=192, m=320 (~17 M-param codec, ~30-40 GB VRAM)
- persistent workers + prefetch to keep GPU fed
- optional torch.compile (--compile) once the model is verified to compile cleanly

Override any of these via CLI flags for smaller GPUs.

Outputs go to runs/<mode>-<lambda>/. Checkpoints are saved as best.pt + last.pt.

Conventions match CompressAI: lambda is the rate-distortion trade-off (higher = lower
distortion, more bits). Sweep [0.0018, 0.0067, 0.0250] to build the RD curve.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ego_codec.data import AriaFrameDataset, AriaPairDataset
from ego_codec.models import IMUConditionedPCodec, MeanScaleHyperprior
from ego_codec.models.baseline import rate_distortion_loss


def build_optimizers(model: torch.nn.Module, lr: float, aux_lr: float):
    """CompressAI convention: entropy-bottleneck CDF params get a separate optimizer.

    Without this, the bottleneck's quantile parameters (which define the discrete CDF
    used at inference) drift slowly and the model never reaches its rate target.
    """
    aux_params, main_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".quantiles"):
            aux_params.append(p)
        else:
            main_params.append(p)
    main_opt = torch.optim.Adam(main_params, lr=lr)
    aux_opt = torch.optim.Adam(aux_params, lr=aux_lr)
    return main_opt, aux_opt


def autocast_ctx(device: torch.device, bf16: bool):
    if bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_compile(model: torch.nn.Module, enable: bool) -> torch.nn.Module:
    """Try torch.compile; fall back silently if it fails (compressai's entropy bottleneck
    has been known to confuse the inductor)."""
    if not enable:
        return model
    try:
        compiled = torch.compile(model, mode="default")
        # Warm up by touching one parameter — surface compile errors early.
        next(model.parameters())
        return compiled
    except Exception as e:
        print(f"[warn] torch.compile failed, running eager: {e!r}")
        return model


def make_loader(ds, args, device):
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
    )


def train_iframe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AriaFrameDataset(args.data, crop_size=args.crop, length_per_seq=args.length_per_seq)
    loader = make_loader(ds, args, device)

    model = MeanScaleHyperprior(n=args.n, m=args.m).to(device)
    main_opt, aux_opt = build_optimizers(model, args.lr, args.aux_lr)
    train_model = maybe_compile(model, args.compile)

    out = Path(args.out) / f"iframe-l{args.lmbda}"
    out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    log_path = out / "train.log"
    log_f = log_path.open("w")
    print(f"[iframe] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"batch={args.batch_size}  crop={args.crop}  bf16={args.bf16}  compile={args.compile}",
          flush=True)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for x in loader:
            x = x.to(device, non_blocking=True)
            with autocast_ctx(device, args.bf16):
                out_codec = train_model(x)
                loss, mse, bpp = rate_distortion_loss(out_codec, x, args.lmbda)

            main_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            main_opt.step()

            aux = model.aux_loss()
            aux_opt.zero_grad(set_to_none=True)
            aux.backward()
            aux_opt.step()

            if step % 50 == 0:
                psnr = 10 * math.log10(1.0 / max(mse.item(), 1e-12))
                vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                line = (
                    f"epoch={epoch} step={step} loss={loss.item():.3f} bpp={bpp.item():.3f} "
                    f"mse={mse.item():.5f} psnr={psnr:.2f} aux={aux.item():.2f} "
                    f"elapsed={time.time()-t0:.0f}s vram={vram:.1f}GB"
                )
                print(line, flush=True)
                log_f.write(line + "\n")
                log_f.flush()
            step += 1

        ckpt = {
            "model": model.state_dict(),
            "main_opt": main_opt.state_dict(),
            "aux_opt": aux_opt.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, out / "last.pt")
        if loss.item() < best:
            best = loss.item()
            torch.save(ckpt, out / "best.pt")

    log_f.close()
    (out / "summary.json").write_text(json.dumps({"final_loss": loss.item(), "best": best}, indent=2))


def train_pframe(args, use_imu: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AriaPairDataset(
        args.data,
        crop_size=args.crop,
        imu_samples=args.imu_samples,
        frame_stride=args.frame_stride,
        length_per_seq=args.length_per_seq,
    )
    loader = make_loader(ds, args, device)

    model = IMUConditionedPCodec(n=args.n, m=args.m, imu_window=args.imu_samples).to(device)

    if args.iframe_ckpt:
        ckpt = torch.load(args.iframe_ckpt, map_location=device)
        model.residual_codec.load_state_dict(ckpt["model"])
        print(f"Loaded I-frame weights from {args.iframe_ckpt}")

    main_opt, aux_opt = build_optimizers(model, args.lr, args.aux_lr)
    tag = "pframe-imu" if use_imu else "pframe-noimu"
    out = Path(args.out) / f"{tag}-l{args.lmbda}"
    out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    log_f = (out / "train.log").open("w")
    train_model = maybe_compile(model, args.compile)
    print(f"[{tag}] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"batch={args.batch_size}  crop={args.crop}  bf16={args.bf16}  compile={args.compile}",
          flush=True)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for x_prev, x_curr, imu in loader:
            x_prev = x_prev.to(device, non_blocking=True)
            x_curr = x_curr.to(device, non_blocking=True)
            imu = imu.to(device, non_blocking=True)

            with autocast_ctx(device, args.bf16):
                out_codec, _ = train_model(x_prev, x_curr, imu, use_imu=use_imu)
                mse = torch.mean((out_codec.x_hat - x_curr) ** 2)
                distortion = 255.0**2 * mse
                loss = args.lmbda * distortion + out_codec.bpp

            main_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            main_opt.step()

            aux = model.aux_loss()
            aux_opt.zero_grad(set_to_none=True)
            aux.backward()
            aux_opt.step()

            if step % 50 == 0:
                psnr = 10 * math.log10(1.0 / max(mse.item(), 1e-12))
                vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                line = (
                    f"[{tag}] epoch={epoch} step={step} loss={loss.item():.3f} "
                    f"bpp={out_codec.bpp.item():.3f} psnr={psnr:.2f} "
                    f"elapsed={time.time()-t0:.0f}s vram={vram:.1f}GB"
                )
                print(line, flush=True)
                log_f.write(line + "\n")
                log_f.flush()
            step += 1

        ckpt = {"model": model.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(ckpt, out / "last.pt")
        if loss.item() < best:
            best = loss.item()
            torch.save(ckpt, out / "best.pt")
    log_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["iframe", "pframe", "pframe-noimu"])
    ap.add_argument("--data", default="data/aria_proc")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--iframe-ckpt", default=None, help="for pframe: load residual codec weights")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--aux-lr", type=float, default=1e-3)
    ap.add_argument("--lmbda", type=float, default=0.0067)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--m", type=int, default=448)
    ap.add_argument("--imu-samples", type=int, default=50)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--length-per-seq", type=int, default=2000,
                    help="Pairs/frames sampled per sequence (controls steps/epoch).")
    ap.add_argument("--bf16", action="store_true", default=True, help="bf16 autocast (default on)")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="torch.compile the model (use after a clean eager run)")
    args = ap.parse_args()

    # Tune torch backend for max throughput on A40 (TF32 + Flash SDPA where applicable).
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if args.mode == "iframe":
        train_iframe(args)
    else:
        train_pframe(args, use_imu=(args.mode == "pframe"))


if __name__ == "__main__":
    main()
