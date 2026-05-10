"""Mean-Scale Hyperprior image codec (Minnen et al., 2018) — used as our I-frame baseline.

Implementation kept deliberately compact: ~150 LOC total. We rely on CompressAI for the
entropy bottleneck and Gaussian conditional, since rolling our own range coder is out of
scope for a 1-week artifact and orthogonal to the egocentric-native story.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from ego_codec.models.blocks import (
    AnalysisTransform,
    HyperAnalysis,
    HyperSynthesis,
    SynthesisTransform,
)


@dataclass
class CodecOutput:
    x_hat: torch.Tensor
    bpp_y: torch.Tensor
    bpp_z: torch.Tensor

    @property
    def bpp(self) -> torch.Tensor:
        return self.bpp_y + self.bpp_z


class MeanScaleHyperprior(nn.Module):
    """I-frame codec.

    Forward returns reconstructions and per-batch bpp (bits per pixel) so callers can build
    rate-distortion losses without re-deriving the pixel count.
    """

    def __init__(self, n: int = 128, m: int = 192):
        super().__init__()
        self.g_a = AnalysisTransform(in_ch=3, n=n, m=m)
        self.g_s = SynthesisTransform(out_ch=3, n=n, m=m)
        self.h_a = HyperAnalysis(n=n, m=m)
        self.h_s = HyperSynthesis(n=n, m=m)
        self.entropy_bottleneck = EntropyBottleneck(n)
        self.gaussian_conditional = GaussianConditional(None)
        self.m = m

    def forward(self, x: torch.Tensor) -> CodecOutput:
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gauss_params = self.h_s(z_hat)
        scales, means = gauss_params.chunk(2, dim=1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means=means)
        x_hat = self.g_s(y_hat)

        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        bpp_y = -torch.log2(y_likelihoods).sum() / num_pixels
        bpp_z = -torch.log2(z_likelihoods).sum() / num_pixels
        return CodecOutput(x_hat=x_hat, bpp_y=bpp_y, bpp_z=bpp_z)

    def aux_loss(self) -> torch.Tensor:
        """Auxiliary loss for the entropy bottleneck's CDF parameters.

        CompressAI requires this be backpropped through a separate optimizer; see train.py.
        """
        return self.entropy_bottleneck.loss()


def rate_distortion_loss(
    out: CodecOutput, x: torch.Tensor, lmbda: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard L = lambda * 255^2 * MSE + bpp. The 255^2 factor matches CompressAI's
    convention so reported lambdas align with the published checkpoints."""
    mse = torch.mean((out.x_hat - x) ** 2)
    distortion = 255.0**2 * mse
    loss = lmbda * distortion + out.bpp
    return loss, mse, out.bpp
