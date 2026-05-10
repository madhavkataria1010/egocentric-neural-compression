"""IMU -> dense flow predictor.

Given an IMU window between two video frames, predict a dense optical-flow field that
explains the global camera-induced motion.

Why this is the egocentric story:
- In ego-video, most pixel motion is camera motion (head turns, walks). Object motion
  is sparse on top of that.
- IMU is sampled at 1 kHz on Project Aria; transmitting the full window costs ~1-3 kbps,
  far below typical video bitrates (1-10 Mbps). For our purposes IMU is "free side info".
- A learned codec normally has to either (a) send a motion vector field as part of the
  bitstream (DCVC-style) or (b) bake camera motion implicitly into latents. Both spend
  bits on something the IMU already knows.

Design: small temporal conv net over the IMU window emits a low-dimensional motion code,
which a tiny decoder upsamples into a dense flow field. We do NOT enforce a strict
homography; the network is free to learn parallax-like residuals from the IMU dynamics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUWarpPredictor(nn.Module):
    """Maps an IMU window (B, T, 6) -> dense flow (B, 2, H, W) for the target image size.

    Channels in the IMU input:
        [0:3] gyroscope (rad/s)
        [3:6] accelerometer (m/s^2)

    Pose is *implicit* — the network integrates as needed. We deliberately don't preprocess
    into rotations/translations because the relevant statistics (head sway, walking gait)
    are easier learned end-to-end on Aria.
    """

    def __init__(
        self,
        imu_window: int = 50,  # ~50ms at 1 kHz
        imu_dim: int = 6,
        hidden: int = 64,
        flow_low_res: tuple[int, int] = (16, 16),
    ):
        super().__init__()
        self.flow_low_res = flow_low_res

        # Temporal encoder over the IMU window: 1D conv stack.
        self.temporal = nn.Sequential(
            nn.Conv1d(imu_dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Project pooled motion code into a low-res 2-channel flow field, then upsample.
        h, w = flow_low_res
        self.to_flow = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 2 * h * w),
        )

        # Small refiner that takes upsampled flow + nothing else and smooths it.
        self.refiner = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(self, imu: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """imu: (B, T, 6); target_hw: (H, W) of the frame to warp.

        Returns flow in pixel units, shape (B, 2, H, W).
        """
        # (B, T, 6) -> (B, 6, T)
        x = imu.transpose(1, 2)
        feat = self.temporal(x).squeeze(-1)  # (B, hidden)

        h, w = self.flow_low_res
        flow_low = self.to_flow(feat).view(-1, 2, h, w)

        H, W = target_hw
        flow = F.interpolate(flow_low, size=(H, W), mode="bilinear", align_corners=True)
        flow = flow + self.refiner(flow)  # residual refine
        return flow
