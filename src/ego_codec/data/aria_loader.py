"""Aria data loaders.

Two formats supported:

1. **Preprocessed directory** (preferred for fast iteration):
       <root>/<sequence>/frames/<%06d>.jpg          # decoded RGB
       <root>/<sequence>/imu.npy                    # (N_imu, 7): [t_us, gx, gy, gz, ax, ay, az]
       <root>/<sequence>/frame_timestamps.npy       # (N_frames,) int64 microseconds

2. **VRS direct** (set ARIA_USE_VRS=1, requires `projectaria-tools`):
       <root>/<sequence>.vrs

For week-1, scripts/preprocess_aria.py extracts (1) from (2). The training loop
operates on (1) only — keeps the dataloader hot path free of VRS dependencies.

Synthetic mode (set ARIA_SYNTHETIC=1) yields random tensors so the training loop and
shape tests run without any data download. Useful for CI / sanity checking on a laptop.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def _is_synthetic() -> bool:
    return os.environ.get("ARIA_SYNTHETIC", "0") == "1"


@dataclass
class Sequence:
    name: str
    frame_paths: list[Path]
    frame_ts_us: np.ndarray  # (N,)
    imu: np.ndarray          # (M, 7) [t_us, gx, gy, gz, ax, ay, az]


def _scan_sequences(root: Path) -> list[Sequence]:
    seqs: list[Sequence] = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        frames_dir = seq_dir / "frames"
        ts_path = seq_dir / "frame_timestamps.npy"
        imu_path = seq_dir / "imu.npy"
        if not (frames_dir.exists() and ts_path.exists() and imu_path.exists()):
            continue
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            continue
        seqs.append(
            Sequence(
                name=seq_dir.name,
                frame_paths=frame_paths,
                frame_ts_us=np.load(ts_path),
                imu=np.load(imu_path),
            )
        )
    return seqs


def _fast_load_jpg(path: Path, target: int) -> Image.Image:
    """Open a JPEG with sub-decoding to ~target pixels per side. Aria RGB is 1408×1408,
    target=288 -> JPEG decoder uses native 1/4 scale (352×352) which is ~16× faster
    than full decode + downscale."""
    img = Image.open(path)
    img.draft("RGB", (target, target))
    return img.convert("RGB")


def _crop_imu_window(imu: np.ndarray, t_start_us: int, t_end_us: int, n_samples: int) -> np.ndarray:
    """Return n_samples IMU readings between t_start_us and t_end_us, padded if needed."""
    mask = (imu[:, 0] >= t_start_us) & (imu[:, 0] <= t_end_us)
    window = imu[mask, 1:7]  # drop timestamp column
    if window.shape[0] == 0:
        return np.zeros((n_samples, 6), dtype=np.float32)
    if window.shape[0] >= n_samples:
        idx = np.linspace(0, window.shape[0] - 1, n_samples).astype(np.int64)
        return window[idx].astype(np.float32)
    # Pad by repeating last sample
    pad = np.tile(window[-1:], (n_samples - window.shape[0], 1))
    return np.concatenate([window, pad], axis=0).astype(np.float32)


class AriaFrameDataset(Dataset):
    """Yields random-cropped RGB frames for I-frame training.

    Aria RGB is 1408×1408 fisheye. We downsample to `resize_to` first (default 288),
    then random-crop to `crop_size` (default 256). This (a) makes JPEG decode an order
    of magnitude faster, (b) matches Build AI's 456×256 deployment resolution, and
    (c) keeps a small ±16px jitter for augmentation.
    """

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 256,
        resize_to: int = 288,
        length_per_seq: int = 200,
    ):
        self.crop_size = crop_size
        self.resize_to = resize_to
        if _is_synthetic():
            self.synthetic_len = 1024
            self.sequences: list[Sequence] = []
            self.index: list[tuple[int, int]] = []
        else:
            self.sequences = _scan_sequences(Path(root))
            if not self.sequences:
                raise RuntimeError(f"No Aria sequences found under {root}. Run scripts/preprocess_aria.py first.")
            self.index = []
            for si, seq in enumerate(self.sequences):
                step = max(1, len(seq.frame_paths) // length_per_seq)
                for fi in range(0, len(seq.frame_paths), step):
                    self.index.append((si, fi))
        self.transform = T.Compose([
            T.Resize(resize_to, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomCrop(crop_size, pad_if_needed=True, padding_mode="reflect"),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return self.synthetic_len if _is_synthetic() else len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if _is_synthetic():
            return torch.rand(3, self.crop_size, self.crop_size)
        si, fi = self.index[idx]
        img = _fast_load_jpg(self.sequences[si].frame_paths[fi], self.resize_to)
        return self.transform(img)


class AriaPairDataset(Dataset):
    """Yields (x_prev, x_curr, imu) triples for P-frame training.

    Both frames are downsampled to `resize_to` and random-cropped at the *same*
    spatial location so the warp is geometrically meaningful. We deliberately do not
    apply random horizontal flips here — flipping inverts the IMU's left/right axes
    and would require co-flipping the gyro/accel signs to remain consistent.
    """

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 256,
        resize_to: int = 288,
        imu_samples: int = 50,
        frame_stride: int = 1,
        length_per_seq: int = 200,
    ):
        self.crop_size = crop_size
        self.resize_to = resize_to
        self.imu_samples = imu_samples
        self.frame_stride = frame_stride
        self.resize = T.Resize(resize_to, interpolation=T.InterpolationMode.BILINEAR)

        if _is_synthetic():
            self.synthetic_len = 1024
            self.sequences = []
            self.pair_index: list[tuple[int, int]] = []
            return

        self.sequences = _scan_sequences(Path(root))
        if not self.sequences:
            raise RuntimeError(f"No Aria sequences found under {root}.")
        self.pair_index = []
        for si, seq in enumerate(self.sequences):
            n_pairs = len(seq.frame_paths) - frame_stride
            if n_pairs <= 0:
                continue
            step = max(1, n_pairs // length_per_seq)
            for fi in range(0, n_pairs, step):
                self.pair_index.append((si, fi))

    def __len__(self) -> int:
        return self.synthetic_len if _is_synthetic() else len(self.pair_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _is_synthetic():
            x_prev = torch.rand(3, self.crop_size, self.crop_size)
            # Synthetic "next frame" = small shift of x_prev — gives the warp something real to learn
            x_curr = torch.roll(x_prev, shifts=(2, 3), dims=(-2, -1))
            imu = torch.randn(self.imu_samples, 6) * 0.1
            return x_prev, x_curr, imu

        si, fi = self.pair_index[idx]
        seq = self.sequences[si]
        prev = _fast_load_jpg(seq.frame_paths[fi], self.resize_to)
        curr = _fast_load_jpg(seq.frame_paths[fi + self.frame_stride], self.resize_to)
        prev = self.resize(prev)
        curr = self.resize(curr)

        # Same random crop for both frames so spatial alignment is preserved.
        i, j, h, w = T.RandomCrop.get_params(prev, output_size=(self.crop_size, self.crop_size))
        prev = T.functional.crop(prev, i, j, h, w)
        curr = T.functional.crop(curr, i, j, h, w)
        prev_t = T.functional.to_tensor(prev)
        curr_t = T.functional.to_tensor(curr)

        t_start = int(seq.frame_ts_us[fi])
        t_end = int(seq.frame_ts_us[fi + self.frame_stride])
        imu_window = _crop_imu_window(seq.imu, t_start, t_end, self.imu_samples)
        return prev_t, curr_t, torch.from_numpy(imu_window)
