"""IMU-conditioned P-frame codec.

Pipeline for a frame pair (x_prev_hat, x_curr) with IMU window in between:

    flow      = IMUWarpPredictor(imu, x_curr.shape[-2:])
    x_pred    = warp(x_prev_hat, flow)             # IMU-conditioned motion compensation
    residual  = x_curr - x_pred
    r_hat,bpp = MeanScaleHyperprior(residual)      # code residual with the I-frame codec
    x_hat     = x_pred + r_hat

The IMU is treated as zero-cost side info: it's already streamed alongside Aria video and
costs ~kbps versus Mbps for video.

Two ablations the eval script measures:
    1. Drop IMU (zero flow). Tests whether IMU adds value at all.
    2. Replace IMU-warp with optical-flow-warp where flow bits ARE counted in bpp.
       Tests whether IMU truly is "free" relative to learned motion encoders.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ego_codec.models.baseline import CodecOutput, MeanScaleHyperprior
from ego_codec.models.blocks import warp_with_flow
from ego_codec.models.imu_warp import IMUWarpPredictor


class IMUConditionedPCodec(nn.Module):
    def __init__(
        self,
        n: int = 128,
        m: int = 192,
        imu_window: int = 50,
        imu_dim: int = 6,
        share_iframe_weights: MeanScaleHyperprior | None = None,
    ):
        super().__init__()
        # Either re-use a pretrained I-frame codec for residual coding, or train fresh.
        self.residual_codec = share_iframe_weights or MeanScaleHyperprior(n=n, m=m)
        self.warp_predictor = IMUWarpPredictor(imu_window=imu_window, imu_dim=imu_dim)

    def forward(
        self,
        x_prev_hat: torch.Tensor,
        x_curr: torch.Tensor,
        imu: torch.Tensor,
        use_imu: bool = True,
    ) -> tuple[CodecOutput, torch.Tensor]:
        """Returns (CodecOutput on residual, predicted-frame x_pred).

        The x_hat field on the returned CodecOutput is overwritten with x_pred + r_hat so
        callers can use it directly for distortion loss against x_curr.
        """
        h, w = x_curr.shape[-2:]
        if use_imu:
            flow = self.warp_predictor(imu, (h, w))
            x_pred = warp_with_flow(x_prev_hat, flow)
        else:
            x_pred = x_prev_hat  # zero-flow baseline

        residual = x_curr - x_pred
        out = self.residual_codec(residual)
        out.x_hat = (x_pred + out.x_hat).clamp(0.0, 1.0)
        return out, x_pred

    def aux_loss(self) -> torch.Tensor:
        return self.residual_codec.aux_loss()
