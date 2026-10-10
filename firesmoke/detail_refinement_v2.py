"""Normalized short stride-4 detail path for the unchanged YOLO11n head."""
from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import Conv, Detect

from .pyramid_refinement import RefinementDetect, CHANNELS


class NormalizedDetailRefinement(nn.Module):
    needs_detail = True
    mode = "detail_norm"

    def __init__(self, hidden=16):
        super().__init__()
        if hidden not in (16, 32):
            raise ValueError("detail_norm supports hidden width16 or32")
        self.hidden = hidden
        self.rearrange = nn.PixelUnshuffle(2)
        self.mix = Conv(4 * CHANNELS[0], hidden, k=1)
        self.local = Conv(hidden, hidden, k=3, g=hidden)
        self.output = nn.Conv2d(hidden, CHANNELS[0], 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features, detail=None):
        if detail is None or detail.shape[-2:] != tuple(2 * n for n in features[0].shape[-2:]):
            raise ValueError("detail_norm requires stride4 detail aligned with stride8 P3")
        delta = self.output(self.local(self.mix(self.rearrange(detail))))
        regression, classification = list(features), list(features)
        regression[0] = regression[0] + delta
        return regression, classification


class NormalizedDetailDetect(RefinementDetect):
    def __init__(self, parent, hidden=16):
        super().__init__(parent, mode="detail", hidden=hidden)
        with torch.random.fork_rng(devices=[]):
            self.refinement = NormalizedDetailRefinement(hidden)
        reference = next(parent.parameters())
        self.refinement.to(device=reference.device, dtype=reference.dtype)
        self.refinement.train(parent.training)
        self.type = "firesmoke.detail_refinement_v2.NormalizedDetailDetect"
        self.np = sum(p.numel() for p in self.parameters())


def install_normalized_detail(model, hidden=16):
    parent = model.model[-1]
    if isinstance(parent, RefinementDetect) or not isinstance(parent, Detect):
        raise ValueError("detail_norm requires an unmodified YOLO11n Detect")
    if parent.i != 23 or parent.f != [16, 19, 22]:
        raise ValueError("detail_norm requires the pinned original YOLO11n graph")
    head = NormalizedDetailDetect(parent, hidden)
    model.model[-1] = head
    model.save = sorted(set(model.save) | {2})
    spec = {"mode": "detail_norm", "hidden": hidden,
            "predictor_channels": list(CHANNELS), "strides": head.stride.tolist(),
            "cached_detail": True,
            "new_parameters": sum(p.numel() for p in head.refinement.parameters()),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "initialization": "Original predictors preserved; zero residual output projection",
            "path": "P2 PixelUnshuffle2 -> ConvBNAct1x1 -> depthwise ConvBNAct3x3 -> output -> P3 regression"}
    model.refinement_spec = spec
    return spec
