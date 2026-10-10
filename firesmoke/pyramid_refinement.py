"""Small residual feature refinements for the pinned three-level YOLO11n head."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect


MODES = ("detail", "context", "full", "local_control", "full_uniform", "full_shared")
CHANNELS = (64, 128, 256)


def project(cin, cout, kernel=1, groups=1, bias=False):
    return nn.Conv2d(cin, cout, kernel, padding=kernel // 2, groups=groups, bias=bias)


def zero_projection(cin, cout):
    layer = project(cin, cout)
    nn.init.zeros_(layer.weight)
    return layer


class DetailRegression(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.reduce = project(64, hidden)
        self.rearrange = nn.PixelUnshuffle(2)
        self.mix = project(4 * hidden, hidden)
        self.local = project(hidden, hidden, 3, groups=hidden)
        self.gate = project(2 * hidden, hidden, bias=True)
        self.output = zero_projection(hidden, 64)

    def forward(self, detail, semantic):
        reduced = F.silu(self.reduce(detail))
        local = F.silu(self.mix(self.rearrange(reduced)))
        gate = self.gate(torch.cat((local, semantic), dim=1)).sigmoid()
        return self.output(gate * F.silu(self.local(local)))


class ContextRefinement(nn.Module):
    def __init__(self, hidden, uniform=False, shared=False):
        super().__init__()
        self.uniform, self.shared = uniform, shared
        self.reduce = nn.ModuleList(project(c, hidden) for c in CHANNELS)
        self.routes = nn.ModuleList(project(3 * hidden, 3, bias=True) for _ in CHANNELS)
        self.local = nn.ModuleList(project(hidden, hidden, 5, groups=hidden) for _ in CHANNELS)
        self.cls_output = nn.ModuleList(zero_projection(hidden, c) for c in CHANNELS)
        self.reg_output = None if shared else nn.ModuleList(zero_projection(hidden, c) for c in CHANNELS)

    @staticmethod
    def resize(source, source_index, target_index):
        if source_index == target_index:
            return source
        if source_index > target_index:
            return F.interpolate(source, scale_factor=2 ** (source_index-target_index), mode="nearest")
        return F.avg_pool2d(source, kernel_size=2 ** (target_index-source_index))

    def forward(self, features):
        reduced = [F.silu(layer(x)) for layer, x in zip(self.reduce, features)]
        regression, classification = [], []
        for i, feature in enumerate(features):
            aligned = [self.resize(z, j, i) for j, z in enumerate(reduced)]
            if self.uniform:
                fused = sum(aligned) / 3
            else:
                weights = self.routes[i](torch.cat(aligned, dim=1)).softmax(dim=1)
                fused = sum(weights[:, j:j+1] * z for j, z in enumerate(aligned))
            context = F.silu(self.local[i](fused))
            cls_delta = self.cls_output[i](context)
            reg_delta = cls_delta if self.shared else self.reg_output[i](context)
            classification.append(feature + cls_delta)
            regression.append(feature + reg_delta)
        return regression, classification, reduced[0]


class LocalControl(nn.Module):
    """Ordinary local bottlenecks, within 5% of r16 full params and conv MACs."""
    def __init__(self):
        super().__init__()
        widths = (27, 8, 13)
        self.reduce = nn.ModuleList(project(c, h) for c, h in zip(CHANNELS, widths))
        self.local = nn.ModuleList(project(h, h, 3) for h in widths)
        self.cls_output = nn.ModuleList(zero_projection(h, c) for c, h in zip(CHANNELS, widths))
        self.reg_output = nn.ModuleList(zero_projection(h, c) for c, h in zip(CHANNELS, widths))

    def forward(self, features):
        regression, classification = [], []
        for i, feature in enumerate(features):
            local = F.silu(self.local[i](F.silu(self.reduce[i](feature))))
            classification.append(feature + self.cls_output[i](local))
            regression.append(feature + self.reg_output[i](local))
        return regression, classification


class PyramidRefinement(nn.Module):
    def __init__(self, mode="full", hidden=16):
        super().__init__()
        if mode not in MODES or hidden not in (16, 32):
            raise ValueError("Unsupported refinement mode or hidden width")
        if mode == "local_control" and hidden != 16:
            raise ValueError("The matched local control is registered only for r16")
        self.mode, self.hidden = mode, hidden
        self.needs_detail = mode in ("detail", "full", "full_uniform", "full_shared")
        self.context = self.detail = self.semantic = self.control = None
        if mode == "local_control":
            self.control = LocalControl()
        else:
            if mode != "detail":
                self.context = ContextRefinement(hidden, uniform=mode == "full_uniform", shared=mode == "full_shared")
            if self.needs_detail:
                self.detail = DetailRegression(hidden)
                if self.context is None:
                    self.semantic = project(64, hidden)

    def forward(self, features, detail=None):
        if self.control is not None:
            return self.control(features)
        if self.context is not None:
            regression, classification, semantic = self.context(features)
        else:
            regression, classification = list(features), list(features)
            semantic = F.silu(self.semantic(features[0]))
        if self.needs_detail:
            if detail is None:
                raise ValueError("The detail path requires the shallow stride-4 feature")
            regression[0] = regression[0] + self.detail(detail, semantic)
        return regression, classification


class RefinementDetect(Detect):
    """Preserve original predictors; provide distinct features to cv2 and cv3."""
    def __init__(self, parent, mode="full", hidden=16):
        if parent.end2end or parent.legacy or parent.nc != 2 or parent.reg_max != 16:
            raise ValueError("Requires the pinned non-legacy, non-end2end two-class YOLO11n head")
        channels = tuple(layer[0].conv.in_channels for layer in parent.cv2)
        if channels != CHANNELS or parent.stride.tolist() != [8., 16., 32.]:
            raise ValueError("Refinement is registered for the original three-level YOLO11n head")
        # New modules never change the parent's/global RNG stream.
        with torch.random.fork_rng(devices=[]):
            super().__init__(nc=parent.nc, reg_max=parent.reg_max, end2end=False, ch=channels)
            self.refinement = PyramidRefinement(mode, hidden)
        self.cv2, self.cv3, self.dfl = parent.cv2, parent.cv3, parent.dfl
        for name in ("stride", "anchors", "strides", "shape", "dynamic", "export", "format",
                     "max_det", "xyxy", "legacy", "i"):
            if hasattr(parent, name):
                setattr(self, name, getattr(parent, name))
        self.f = [2, 16, 19, 22] if self.refinement.needs_detail else [16, 19, 22]
        self.type = "firesmoke.pyramid_refinement.RefinementDetect"
        self.np = sum(p.numel() for p in self.parameters())
        reference = next(parent.parameters())
        self.refinement.to(device=reference.device, dtype=reference.dtype)
        self.train(parent.training)

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        detail, features = (x[0], x[1:]) if self.refinement.needs_detail else (None, x)
        regression, classification = self.refinement(features, detail)
        batch = features[0].shape[0]
        boxes = torch.cat([box_head[i](regression[i]).view(batch, 4 * self.reg_max, -1)
                           for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](classification[i]).view(batch, self.nc, -1)
                            for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=features)


def install_refinement(model, mode="full", hidden=16):
    if isinstance(model.model[-1], RefinementDetect):
        raise ValueError("Refinement is already installed")
    parent = model.model[-1]
    if not isinstance(parent, Detect) or parent.i != 23 or parent.f != [16, 19, 22]:
        raise ValueError("Expected the unmodified YOLO11n Detect at layer23")
    head = RefinementDetect(parent, mode, hidden)
    model.model[-1] = head
    if head.refinement.needs_detail:
        model.save = sorted(set(model.save) | {2})
    spec = {"mode": mode, "hidden": hidden, "predictor_channels": list(CHANNELS),
            "strides": head.stride.tolist(), "cached_detail": head.refinement.needs_detail,
            "new_parameters": sum(p.numel() for p in head.refinement.parameters()),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "initialization": "Original predictors preserved; zero residual output projections"}
    model.refinement_spec = spec
    return spec
