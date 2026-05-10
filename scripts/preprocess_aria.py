"""Extract RGB frames + IMU from a Project Aria VRS file into the layout
expected by AriaFrameDataset / AriaPairDataset.

    <out_root>/<sequence>/frames/000000.jpg
                          frame_timestamps.npy   # (N,) int64 us
                          imu.npy                # (M, 7) float64

Usage:
    python scripts/preprocess_aria.py --vrs /path/to/sequence.vrs --out data/aria_proc --seq seq01
    # batch:
    python scripts/preprocess_aria.py --vrs-dir /path/to/vrs_dir --out data/aria_proc

Requires `projectaria-tools` (install with: pip install -e ".[aria]").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _import_aria():
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except ImportError as e:
        sys.stderr.write("projectaria-tools not installed. pip install -e \".[aria]\"\n")
        raise SystemExit(1) from e
    return data_provider, StreamId


def extract_one(vrs_path: Path, out_dir: Path, fps: int = 10) -> None:
    data_provider, StreamId = _import_aria()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Could not open {vrs_path}")

    rgb_id = StreamId("214-1")  # Aria RGB camera
    imu_id = StreamId("1202-1")  # IMU-right (Aria has two IMUs; right is closer to RGB)

    # Frames
    n_rgb = provider.get_num_data(rgb_id)
    rgb_period_us = int(1e6 / fps)
    last_t = -10**18
    frame_ts = []
    saved = 0
    for i in range(n_rgb):
        rec = provider.get_image_data_by_index(rgb_id, i)
        img = rec[0].to_numpy_array()
        t_us = rec[1].capture_timestamp_ns // 1000
        if t_us - last_t < rgb_period_us:
            continue
        Image.fromarray(img).save(frames_dir / f"{saved:06d}.jpg", quality=92)
        frame_ts.append(t_us)
        last_t = t_us
        saved += 1
    np.save(out_dir / "frame_timestamps.npy", np.asarray(frame_ts, dtype=np.int64))

    # IMU: stack (t_us, gx, gy, gz, ax, ay, az)
    n_imu = provider.get_num_data(imu_id)
    imu = np.zeros((n_imu, 7), dtype=np.float64)
    for i in range(n_imu):
        rec = provider.get_imu_data_by_index(imu_id, i)
        imu[i, 0] = rec.capture_timestamp_ns / 1000.0
        imu[i, 1:4] = rec.gyro_radsec
        imu[i, 4:7] = rec.accel_msec2
    np.save(out_dir / "imu.npy", imu)
    print(f"[{out_dir.name}] saved {saved} frames + {n_imu} IMU samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrs", type=Path, help="Single VRS file")
    ap.add_argument("--vrs-dir", type=Path, help="Directory of VRS files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq", type=str, default=None)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    if args.vrs:
        seq_name = args.seq or args.vrs.stem
        extract_one(args.vrs, args.out / seq_name, fps=args.fps)
    elif args.vrs_dir:
        for vrs in sorted(args.vrs_dir.glob("*.vrs")):
            extract_one(vrs, args.out / vrs.stem, fps=args.fps)
    else:
        ap.error("Provide --vrs or --vrs-dir")


if __name__ == "__main__":
    main()
