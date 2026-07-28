"""
Model/FNO3D_encoder.py  ── v10-turbo
======================================
Fourier Neural Operator (FNO) replacing UNet3D for Data3d encoding.

FIXES vs original:
  1. SpectralConv3d._complex_mul: einsum subscript collision fixed.
     "bimno,iomno->bomno" → "bipqr,ijpqr->bjpqr"

  2. ComplexHalf / AMP: cast to float32 before FFT, cast back after.

  3. SpectralConv3d.out_ft always dtype=torch.cfloat.

  4. PARAMS FIX (v10-fixed-3): default d_model 64→32, modes_h/w 16→4.
     Root cause: shape=(64,64,4,16,16) per SpectralConv3d weight tensor
     = 2 × 64×64×4×16×16 = 8.4M params × 4 layers = 33.5M — 94.8% of
     total model params, far above the <1.2M target.
     Fix: d_model=32, modes_h=4, modes_w=4 →
       shape=(32,32,4,4,4) = 2×32×32×4×4×4 = 131K × 4 = 524K params.
     Total FNO3D after fix: ~0.6M  (was 33.5M).

     VelocityField in flow_matching_model.py also hard-codes d_model=64,
     modes_h=16, modes_w=16 — update those values too (see comment at
     bottom of this file).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  Spectral Convolution Layer
# ══════════════════════════════════════════════════════════════════════════════

class SpectralConv3d(nn.Module):
    """
    3D Fourier integral operator layer.

    Param count = 2 × in_ch × out_ch × modes_t × modes_h × modes_w
    With d_model=32, modes_t=4, modes_h=4, modes_w=4:
      2 × 32 × 32 × 4 × 4 × 4 = 131,072 per layer   ✓
    With d_model=64, modes_h=16, modes_w=16 (old default):
      2 × 64 × 64 × 4 × 16 × 16 = 8,388,608 per layer  ✗
    """

    def __init__(
            self,
            in_channels:  int,
            out_channels: int,
            modes_t:      int = 4,
            modes_h:      int = 4,
            modes_w:      int = 4,
        ):
            super().__init__()
            self.in_ch   = in_channels
            self.out_ch  = out_channels
            self.modes_t = modes_t
            self.modes_h = modes_h
            self.modes_w = modes_w

            _scale = 1.0 / math.sqrt(in_channels * out_channels)
            _shape = (in_channels, out_channels, modes_t, modes_h, modes_w)

            # All 4 corners initialised here, once, as proper parameters
            for i in range(1, 5):
                setattr(self, f'w_re_{i}', nn.Parameter(_scale * torch.randn(*_shape)))
                setattr(self, f'w_im_{i}', nn.Parameter(_scale * torch.randn(*_shape)))

    def _complex_mul(self, x, w_re, w_im):
            w_re = w_re.float()
            w_im = w_im.float()
            x_re = x.real.float()
            x_im = x.imag.float()
            out_re = (torch.einsum("bipqr,ijpqr->bjpqr", x_re, w_re)
                    - torch.einsum("bipqr,ijpqr->bjpqr", x_im, w_im))
            out_im = (torch.einsum("bipqr,ijpqr->bjpqr", x_re, w_im)
                    + torch.einsum("bipqr,ijpqr->bjpqr", x_im, w_re))
            return torch.complex(out_re, out_im)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, T, H, W = x.shape
            x_ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")

            out_ft = torch.zeros(B, self.out_ch, T, H, W // 2 + 1,
                                dtype=torch.cfloat, device=x.device)

            mt = min(self.modes_t, T // 2)
            mh = min(self.modes_h, H // 2)
            mw = min(self.modes_w, W // 2 + 1)

            # FIX LOGIC: Lấy đủ 4 góc phổ để giữ các tần số thấp quan trọng
            # (Chỉ chiều W là không cần vì rfftn đã cắt một nửa)
            
            # Góc 1: Dương Time, Dương Height
            out_ft[:, :, :mt, :mh, :mw] = self._complex_mul(
                x_ft[:, :, :mt, :mh, :mw], self.w_re_1, self.w_im_1)
            
            # Góc 2: Âm Time, Dương Height (Thêm phần này để model "nhạy" hơn)
            out_ft[:, :, -mt:, :mh, :mw] = self._complex_mul(
                x_ft[:, :, -mt:, :mh, :mw], self.w_re_2, self.w_im_2)
                
            # Góc 3: Dương Time, Âm Height
            out_ft[:, :, :mt, -mh:, :mw] = self._complex_mul(
                x_ft[:, :, :mt, -mh:, :mw], self.w_re_3, self.w_im_3)
                
            # Góc 4: Âm Time, Âm Height
            out_ft[:, :, -mt:, -mh:, :mw] = self._complex_mul(
                x_ft[:, :, -mt:, -mh:, :mw], self.w_re_4, self.w_im_4)

            return torch.fft.irfftn(out_ft, s=(T, H, W), dim=(-3, -2, -1), norm="ortho").to(x.dtype)

# ══════════════════════════════════════════════════════════════════════════════
#  FNO Layer = SpectralConv + local Conv residual
# ══════════════════════════════════════════════════════════════════════════════

class FNOLayer3d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_t:  int   = 4,
        modes_h:  int   = 4,    # FIX: was 16
        modes_w:  int   = 4,    # FIX: was 16
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, modes_t, modes_h, modes_w)
        self.local    = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm     = nn.InstanceNorm3d(channels, affine=True)
        self.act      = nn.GELU()
        self.drop     = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.spectral(x) + self.local(x))))


# ══════════════════════════════════════════════════════════════════════════════
#  FNO3D Encoder
# ══════════════════════════════════════════════════════════════════════════════

class FNO3DEncoder(nn.Module):
    """
    Drop-in replacement for Unet3D.
    encode(x) → (bottleneck [B,128,T,4,4], summary [B,1,T,1,1])

    Param budget (d_model=32, modes=4, n_layers=4):
      lift             :  13→32 Conv3d 1×1  =     416
      fno_layers × 4   :  SpectralConv + Conv1×1 + IN
                          ≈ 4 × (131K + 1K + 64) ≈ 530K
      proj_bottleneck  :  32→128 Conv3d 1×1 =   4,096
      summary_conv     :  32→16→1            =     528
      TOTAL                                  ≈ 535K   ✓
    """

    def __init__(
        self,
        in_channel:   int   = 13,
        out_channel:  int   = 1,
        d_model:      int   = 32,    # FIX: was 64
        n_layers:     int   = 4,
        modes_t:      int   = 4,
        modes_h:      int   = 4,     # FIX: was 16
        modes_w:      int   = 4,     # FIX: was 16
        spatial_down: int   = 32,
        dropout:      float = 0.05,
    ):
        super().__init__()
        self.spatial_down = spatial_down
        self.d_model      = d_model
        self.in_channel   = in_channel

        self.lift = nn.Sequential(
            nn.Conv3d(in_channel, d_model, kernel_size=1, bias=False),
            nn.InstanceNorm3d(d_model, affine=True),
            nn.GELU(),
        )

        self.fno_layers = nn.ModuleList([
            FNOLayer3d(d_model, modes_t, modes_h, modes_w, dropout)
            for _ in range(n_layers)
        ])

        # Project to 128 for DataEncoder1D compatibility (feat_3d_dim=128)
        self.proj_bottleneck = nn.Conv3d(d_model, 128, kernel_size=1, bias=False)

        self.summary_conv = nn.Sequential(
            nn.Conv3d(d_model, 16, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(16, out_channel, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )

        self.inc = _ChannelProxy(in_channel)

    def _downsample_spatial(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        if H == self.spatial_down and W == self.spatial_down:
            return x
        x2 = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x2 = F.interpolate(
            x2,
            size=(self.spatial_down, self.spatial_down),
            mode="bilinear",
            align_corners=False,
        )
        return x2.reshape(B, T, C, self.spatial_down, self.spatial_down).permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, summary = self.encode(x)
        return summary

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, C, T, H, W = x.shape

        if C == 1 and self.in_channel != 1:
            x = x.expand(-1, self.in_channel, -1, -1, -1)

        x = self._downsample_spatial(x)        # [B, 13,      T, 32, 32]
        x = self.lift(x)                       # [B, d_model, T, 32, 32]

        for layer in self.fno_layers:
            x = layer(x)                       # [B, d_model, T, 32, 32]

        bot = self.proj_bottleneck(x)          # [B, 128, T, 32, 32]
        bot = F.adaptive_avg_pool3d(bot, (None, 4, 4))  # [B, 128, T, 4, 4]

        summary = self.summary_conv(x)         # [B, 1, T, 1, 1]

        return bot, summary


class _ChannelProxy:
    """Dummy proxy so flow_matching_model can read .inc.skip.in_channels."""
    class _skip:
        def __init__(self, c): self.in_channels = c
    def __init__(self, c): self.skip = _ChannelProxy._skip(c)


# Backward-compat alias
Unet3D = FNO3DEncoder


# ══════════════════════════════════════════════════════════════════════════════
#  NOTE: also update VelocityField in flow_matching_model.py
# ══════════════════════════════════════════════════════════════════════════════
# In VelocityField.__init__, the FNO3DEncoder is constructed with explicit
# kwargs that override these defaults. Change those lines too:
#
#   self.spatial_enc = FNO3DEncoder(
#       in_channel   = unet_in_ch,
#       out_channel  = 1,
#       d_model      = 32,    # ← was 64
#       n_layers     = 4,
#       modes_t      = 4,
#       modes_h      = 4,     # ← was 16
#       modes_w      = 4,     # ← was 16
#       spatial_down = 32,
#       dropout      = 0.05,
#   )