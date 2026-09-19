"""Detail-conditioned band residual (DCBR), an unvalidated research candidate.

This is a spatial smoothing residual decomposition, not an FFT or orthogonal
wavelet transform. See docs/MODULE_DCBR.md for prior art and limitations.
"""
from __future__ import annotations

import torch
from torch import nn


class DetailConditionedBandResidual(nn.Module):
    """Refine two detail bands while retaining the coarse component explicitly.

    Inputs: concatenated [upsampled semantic, shallow detail], in BCHW order.
    `semantic` ablates local conditioning; `whole` applies the same two gates
    to D/2 instead of separate bands. Both retain the same parameter shapes.
    Zero output projection and isolated RNG preserve the parent P2 start.
    """

    def __init__(self, semantic_channels: int, detail_channels: int,
                 hidden: int = 16, groups: int = 8, gain: float = 0.25,
                 mode: str = "joint"):
        super().__init__()
        if min(semantic_channels, detail_channels, hidden, groups) < 1:
            raise ValueError("channel counts and groups must be positive")
        if detail_channels % groups:
            raise ValueError("detail_channels must be divisible by groups")
        if not 0 < gain < 1:
            raise ValueError("gain must be in (0, 1)")
        if mode not in ("joint", "semantic", "whole"):
            raise ValueError("mode must be joint, semantic or whole")
        self.semantic_channels = semantic_channels
        self.detail_channels = detail_channels
        self.groups = groups
        self.gain = float(gain)
        self.mode = mode
        # Excluding padded cells preserves spatially constant signals at borders.
        self.smooth3 = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.smooth5 = nn.AvgPool2d(5, stride=1, padding=2, count_include_pad=False)
        with torch.random.fork_rng(devices=[]):
            self.context = nn.Sequential(
                nn.Conv2d(semantic_channels + 2 * detail_channels, hidden, 1),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                nn.SiLU(),
            )
            self.projection = nn.Conv2d(hidden, 2 * groups, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def decompose(self, detail):
        """Return L, M, H with D = L + M + H up to floating-point error."""
        smooth = self.smooth3(detail)
        low = self.smooth5(smooth)
        return low, smooth - low, detail - smooth

    def _parts(self, x):
        expected = self.semantic_channels + self.detail_channels
        if x.ndim != 4 or x.shape[1] != expected:
            raise ValueError(f"expected BCHW input with {expected} channels, got {tuple(x.shape)}")
        semantic, detail = x.split((self.semantic_channels, self.detail_channels), dim=1)
        low, middle, high = self.decompose(detail)
        local = torch.cat((low, middle.abs() + high.abs()), dim=1)
        if self.mode == "semantic":
            local = torch.zeros_like(local)
        context = self.context(torch.cat((semantic, local), dim=1))
        gates = self.gain * torch.tanh(self.projection(context))
        mid_gate, high_gate = gates.chunk(2, dim=1)
        mid_gate = mid_gate.repeat_interleave(self.detail_channels // self.groups, dim=1)
        high_gate = high_gate.repeat_interleave(self.detail_channels // self.groups, dim=1)
        if self.mode == "whole":
            residual = (mid_gate + high_gate) * (detail * 0.5)
        else:
            residual = mid_gate * middle + high_gate * high
        return semantic, detail, low, middle, high, gates, residual

    def forward(self, x):
        semantic, detail, _, _, _, _, residual = self._parts(x)
        return torch.cat((semantic, detail + residual), dim=1)

    @torch.no_grad()
    def diagnostics(self, x):
        """Explicit probe only: no retained tensors or sync during normal forward.

        Probe on held-out validation features, stratified by true objects and
        backgrounds, before making any mechanism claim. Whole-image averages
        do not measure object recall or false alarms.
        """
        _, detail, low, middle, high, gates, residual = self._parts(x)
        rms = lambda t: float(t.float().square().mean().sqrt().cpu())
        return {
            "mode": self.mode, "gain": self.gain,
            "gate_mean": float(gates.float().mean().cpu()),
            "gate_abs_max": float(gates.abs().max().cpu()),
            "gate_saturation_fraction": float((gates.abs() >= .95 * self.gain).float().mean().cpu()),
            "low_rms": rms(low), "middle_rms": rms(middle), "high_rms": rms(high),
            "residual_rms": rms(residual),
            "residual_to_detail_rms": rms(residual) / max(rms(detail), 1e-12),
        }
