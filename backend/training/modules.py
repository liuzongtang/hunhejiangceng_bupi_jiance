"""
Custom neural-network modules for the RT-DETR fabric-defect variant.

Three neck upgrades, injected into ``ultralytics.nn.tasks`` at runtime:

1. ``CARAFE`` — content-aware upsampling (replaces the two ``nn.Upsample`` ops).
   Channel-preserving, so it is referenced by a *new* name in the YAML; the
   ``parse_model`` ``else`` branch (``c2 = ch[f]``) accounts for it correctly.
2. ``DeformableAIFI`` — deformable intra-scale attention (replaces ``AIFI``).
   Channel-preserving, referenced by a new name for the same reason.
3. ``ContextFusion`` — adaptive cross-scale fusion (replaces ``RepC3`` in the
   CCFM neck). It *changes* the channel count (2C -> C), which the ``else``
   branch cannot account for, so it is injected by monkeypatching
   ``ultralytics.nn.tasks.RepC3``; the YAML still writes ``RepC3`` and the
   ``base_modules`` accounting (``c1, c2 = ch[f], args[0]``) stays correct.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import RepC3
from ultralytics.nn.modules.transformer import AIFI, MSDeformAttn


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures (CARAFE) upsampling operator.

    Replaces nearest-neighbour/bilinear upsampling with content-aware
    reassembly: a small convolution predicts a per-position ``k_up x k_up``
    recombination kernel, which is then used to weight the source-feature
    neighbourhood. This yields sharper, more accurate boundaries than fixed
    interpolation, which matters for thin defect structures (broken warp,
    weft shrink, etc.).

    Args:
        c1: Input channel count.
        c2: Output channel count.
        scale: Upsampling factor (default 2).
        k_up: Recombination kernel size (default 3).
        c_m: Intermediate channels for kernel prediction (default 64).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        scale: int = 2,
        k_up: int = 3,
        c_m: int = 64,
    ):
        super().__init__()
        self.scale = scale
        self.k_up = k_up
        self.compress = nn.Conv2d(c1, c_m, 1)
        self.encoder = nn.Conv2d(c_m, scale * scale * k_up * k_up, 3, padding=1)
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` by ``scale`` using content-aware reassembly."""
        b, c, h, w = x.shape
        s, k = self.scale, self.k_up

        # 1) Content-aware kernel prediction, softmax-normalised over the
        #    k_up x k_up recombination neighbourhood.
        weight = self.encoder(self.compress(x))  # [B, s^2*k^2, H, W]
        weight = weight.reshape(b, s * s, k * k, h, w).softmax(dim=2)

        # 2) Unfold the source feature neighbourhood (reflect-padded).
        x_pad = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
        x_unf = F.unfold(x_pad, k, padding=0).reshape(b, c, k * k, h, w)

        # 3) Weighted recombination per target position (sum over kernel dim).
        out = torch.einsum("bckhw,bskhw->bcshw", x_unf, weight)  # [B, C, s^2, H, W]

        # 4) Pixel-shuffle into [B, C, sH, sW].
        out = (
            out.reshape(b, c, s, s, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(b, c, s * h, s * w)
        )
        return self.proj(out)


class DeformableAIFI(nn.Module):
    """Deformable-attention AIFI: intra-scale interaction with deformable attention.

    Replaces the vanilla ``AIFI`` (full multi-head self-attention over the
    flattened P5 feature map) with single-level multi-scale deformable attention
    (``n_levels=1``). Each spatial token attends to a fixed number of
    learned-offset sampling points around its reference position instead of every
    other token, giving geometry-aware local context at linear cost rather than
    the quadratic cost of dense self-attention. The feed-forward network and
    post-normalisation residual layout match ``AIFI``.

    Args:
        c1: Input (and output) channel count.
        cm: Feed-forward hidden dimension.
        num_heads: Number of attention heads.
        n_points: Sampling points per attention head.
        dropout: Dropout probability.
        act: Activation for the feed-forward network.
        normalize_before: If True, use pre-normalisation (unused; kept for parity
            with ``AIFI``'s signature).
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        n_points: int = 4,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(
            d_model=c1, n_levels=1, n_heads=num_heads, n_points=n_points
        )
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def _reference_points(h: int, w: int, device, dtype) -> torch.Tensor:
        """Return normalised [0, 1] reference points for each of ``h * w`` tokens."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        ref = torch.stack(
            [(grid_x.reshape(-1) + 0.5) / w, (grid_y.reshape(-1) + 0.5) / h], dim=-1
        )  # [H*W, 2]
        return ref

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply deformable self-attention + feed-forward to ``[B, C, H, W]``."""
        b, c, h, w = x.shape
        src = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        pos = AIFI.build_2d_sincos_position_embedding(w, h, c, device=x.device).to(
            dtype=x.dtype
        )  # [1, H*W, C]
        ref = self._reference_points(h, w, x.device, x.dtype)
        ref = ref.unsqueeze(0).unsqueeze(2).expand(b, -1, 1, -1)  # [B, H*W, 1, 2]

        q = src + pos
        src2 = self.self_attn(q, ref, value=src, value_shapes=[(h, w)])  # [B, H*W, C]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.permute(0, 2, 1).reshape(b, c, h, w).contiguous()


class ContextFusion(nn.Module):
    """Adaptive cross-scale context fusion (BiFPN-style fast-normalized fusion).

    Drop-in replacement for the ``RepC3`` blocks of the CCFM neck. The
    concatenated two-scale input ``[2C, H, W]`` (top-down upsampled feature +
    lateral skip feature) is split into its two ``C``-channel halves and fused
    with learned per-branch weights, then refined by a ``RepC3`` bottleneck. This
    replaces fixed channel concatenation with an adaptive, content-weighted
    fusion of the two scales. Signature matches ``RepC3`` so it can be injected
    by monkeypatching ``ultralytics.nn.tasks.RepC3`` (the YAML still writes
    ``RepC3`` and the channel accounting stays correct).

    Args:
        c1: Input channels (concatenated two-scale feature; ``2 * c2``).
        c2: Output channels (fused feature width).
        n: Number of ``RepConv`` blocks in the refinement bottleneck.
        e: Expansion ratio for the refinement bottleneck.
    """

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        super().__init__()
        if c1 != 2 * c2:
            raise ValueError(
                f"ContextFusion expects c1 == 2 * c2 (concatenated two-scale "
                f"input), got c1={c1}, c2={c2}"
            )
        self.w_high = nn.Parameter(torch.ones(1))
        self.w_low = nn.Parameter(torch.ones(1))
        self.rep = RepC3(c2, c2, n, e)  # refine the fused [C] feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse the two-scale input with learned weights and refine it."""
        high, low = x.chunk(2, dim=1)  # [B, 2C, H, W] -> two [B, C, H, W]
        w = torch.relu(self.w_high) + torch.relu(self.w_low) + 1e-4
        fused = (torch.relu(self.w_high) / w) * high + (
            torch.relu(self.w_low) / w
        ) * low
        return self.rep(fused)
