import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import GDN


def conv(in_ch: int, out_ch: int, kernel: int = 5, stride: int = 2) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)


def deconv(in_ch: int, out_ch: int, kernel: int = 5, stride: int = 2) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_ch,
        out_ch,
        kernel,
        stride=stride,
        padding=kernel // 2,
        output_padding=stride - 1,
    )


class AnalysisTransform(nn.Module):
    """4-stage analysis (encoder) g_a: x -> y. 16x spatial downsample."""

    def __init__(self, in_ch: int = 3, n: int = 128, m: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            conv(in_ch, n), GDN(n),
            conv(n, n),    GDN(n),
            conv(n, n),    GDN(n),
            conv(n, m),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SynthesisTransform(nn.Module):
    """4-stage synthesis (decoder) g_s: y_hat -> x_hat."""

    def __init__(self, out_ch: int = 3, n: int = 128, m: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            deconv(m, n), GDN(n, inverse=True),
            deconv(n, n), GDN(n, inverse=True),
            deconv(n, n), GDN(n, inverse=True),
            deconv(n, out_ch),
        )

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        return self.net(y_hat)


class HyperAnalysis(nn.Module):
    """h_a: y -> z. 4x further spatial downsample."""

    def __init__(self, n: int = 128, m: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            conv(m, n, kernel=3, stride=1),
            nn.LeakyReLU(inplace=True),
            conv(n, n),
            nn.LeakyReLU(inplace=True),
            conv(n, n),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.abs(y))


class HyperSynthesis(nn.Module):
    """h_s: z_hat -> (mu, sigma) params for y. Outputs 2*M channels."""

    def __init__(self, n: int = 128, m: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            deconv(n, n),
            nn.LeakyReLU(inplace=True),
            deconv(n, n * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(n * 3 // 2, m * 2, kernel=3, stride=1),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)


def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp `image` (B, C, H, W) by per-pixel flow (B, 2, H, W) in pixel units."""
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    grid_x = xx.unsqueeze(0).expand(b, -1, -1) + flow[:, 0]
    grid_y = yy.unsqueeze(0).expand(b, -1, -1) + flow[:, 1]
    grid_x = 2.0 * grid_x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
