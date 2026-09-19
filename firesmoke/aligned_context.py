"""Experimental discrete alignment and spatial context routing for the P2 branch."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AlignedContextResidual(nn.Module):
    """Input/output: concatenated [semantic, detail], equal C channels each.

    Nine integer-offset semantic candidates avoid grid_sample CUDA backward.
    Three dilated detail experts test spatial context selection. Zero output
    projections preserve the parent graph at initialization. No class-specific
    interpretation of the experts is imposed or claimed.
    """

    def __init__(self, channels=64, hidden=16, mode="full"):
        super().__init__()
        if channels < 1 or hidden < 1 or mode not in {"full", "no_align", "uniform", "local"}:
            raise ValueError("Invalid channels, hidden size or ablation mode")
        self.channels, self.mode = channels, mode
        with torch.random.fork_rng(devices=[]):
            self.condition = nn.Sequential(nn.Conv2d(2 * channels, hidden, 1), nn.SiLU())
            self.offset_logits = nn.Conv2d(hidden, 9, 1)
            self.scale_logits = nn.Conv2d(hidden, 3, 1)
            self.detail_reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.SiLU())
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d,
                                        groups=hidden), nn.SiLU()) for d in (1, 3, 5)])
            self.semantic_out = nn.Conv2d(channels, channels, 1, bias=False)
            self.detail_out = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.semantic_out.weight)
        nn.init.zeros_(self.detail_out.weight)

    @staticmethod
    def align(semantic, weights):
        """Convex local resampling; constant maps are preserved at boundaries."""
        h, w = semantic.shape[-2:]
        padded = F.pad(semantic, (1, 1, 1, 1), mode="replicate")
        result = torch.zeros_like(semantic)
        for i in range(9):
            dy, dx = divmod(i, 3)
            result = result + weights[:, i:i + 1] * padded[:, :, dy:dy + h, dx:dx + w]
        return result

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 2 * self.channels:
            raise ValueError("Expected BCHW with 2 * channels")
        semantic, detail = x.chunk(2, 1)
        condition = self.condition(x)
        if self.mode == "no_align":
            aligned = semantic
        else:
            weights = self.offset_logits(condition).softmax(1)
            aligned = self.align(semantic, weights)
        local = self.detail_reduce(detail)
        experts = [expert(local) for expert in self.experts]
        if self.mode == "local":
            context = experts[0]
        elif self.mode == "uniform":
            context = sum(experts) / 3
        else:
            routes = self.scale_logits(condition).softmax(1)
            context = sum(routes[:, i:i + 1] * e for i, e in enumerate(experts))
        return torch.cat((semantic + self.semantic_out(aligned - semantic),
                          detail + self.detail_out(context)), 1)
