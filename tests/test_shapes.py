"""Shape and forward-pass sanity tests. No GPU required.

Run with: ARIA_SYNTHETIC=1 pytest -q
"""
import os

os.environ.setdefault("ARIA_SYNTHETIC", "1")

import torch
from torch.utils.data import DataLoader

from ego_codec.data import AriaFrameDataset, AriaPairDataset
from ego_codec.models import IMUConditionedPCodec, IMUWarpPredictor, MeanScaleHyperprior
from ego_codec.models.baseline import rate_distortion_loss
from ego_codec.models.blocks import warp_with_flow


def test_baseline_forward():
    model = MeanScaleHyperprior(n=64, m=96)  # smaller for CPU speed
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    assert out.x_hat.shape == x.shape
    assert out.bpp.ndim == 0
    assert torch.isfinite(out.bpp)


def test_baseline_loss_backward():
    model = MeanScaleHyperprior(n=64, m=96)
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    loss, mse, bpp = rate_distortion_loss(out, x, lmbda=0.01)
    loss.backward()
    # At least one main parameter should now have a gradient.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_imu_warp_predictor():
    pred = IMUWarpPredictor(imu_window=50, imu_dim=6)
    imu = torch.randn(4, 50, 6) * 0.1
    flow = pred(imu, target_hw=(64, 64))
    assert flow.shape == (4, 2, 64, 64)


def test_warp_with_flow_identity():
    img = torch.rand(1, 3, 32, 32)
    zero_flow = torch.zeros(1, 2, 32, 32)
    out = warp_with_flow(img, zero_flow)
    assert torch.allclose(out, img, atol=1e-4)


def test_pcodec_forward():
    model = IMUConditionedPCodec(n=64, m=96, imu_window=50)
    x_prev = torch.rand(2, 3, 64, 64)
    x_curr = torch.rand(2, 3, 64, 64)
    imu = torch.randn(2, 50, 6) * 0.1
    out, x_pred = model(x_prev, x_curr, imu, use_imu=True)
    assert out.x_hat.shape == x_curr.shape
    assert x_pred.shape == x_curr.shape
    assert torch.isfinite(out.bpp)


def test_pcodec_no_imu_path():
    model = IMUConditionedPCodec(n=64, m=96, imu_window=50)
    x_prev = torch.rand(1, 3, 64, 64)
    x_curr = torch.rand(1, 3, 64, 64)
    imu = torch.randn(1, 50, 6) * 0.1
    out, x_pred = model(x_prev, x_curr, imu, use_imu=False)
    assert torch.allclose(x_pred, x_prev)


def test_synthetic_frame_dataset():
    ds = AriaFrameDataset(root="UNUSED", crop_size=64)
    assert len(ds) > 0
    x = ds[0]
    assert x.shape == (3, 64, 64)


def test_synthetic_pair_dataset_loader():
    ds = AriaPairDataset(root="UNUSED", crop_size=64, imu_samples=50)
    loader = DataLoader(ds, batch_size=4)
    x_prev, x_curr, imu = next(iter(loader))
    assert x_prev.shape == (4, 3, 64, 64)
    assert x_curr.shape == (4, 3, 64, 64)
    assert imu.shape == (4, 50, 6)
