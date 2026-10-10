"""Frozen-parent quality readout and anchor-preserving inference adapter.

No ground truth is used by QualityTap or by the deployed score calculation.
Ground truth is used only by sampled_quality_batch during readout fitting.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class LocalizationQualityReadout(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.reg_reduce = nn.Conv1d(64, 8, 1)
            self.predictor = nn.Sequential(nn.Conv1d(37, 32, 1), nn.SiLU(), nn.Conv1d(32, 2, 1))

    @staticmethod
    def distribution_statistics(box_logits):
        if box_logits.ndim != 3 or box_logits.shape[1] != 64:
            raise ValueError("Expected Bx64xA DFL logits (four sides, 16 bins)")
        probability = box_logits.float().reshape(box_logits.shape[0], 4, 16, -1).softmax(2)
        top4 = probability.topk(4, dim=2).values
        support = torch.arange(16, device=probability.device).view(1, 1, 16, 1)
        mean = (probability * support).sum(2, keepdim=True) / 15
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(2, keepdim=True) / math.log(16)
        return torch.cat((top4, mean, entropy), dim=2).flatten(1, 2)

    def forward(self, box_logits, regression_features, class_logits, level_ids, *, detach_inputs=True):
        b, _, a = box_logits.shape
        if regression_features.shape != (b, 64, a) or class_logits.shape != (b, 2, a):
            raise ValueError("All predictor inputs must use the same anchor order")
        if level_ids.shape not in ((a,), (b, a)) or level_ids.dtype != torch.long:
            raise ValueError("level_ids must be a long vector with one entry per anchor")
        if bool(((level_ids < 0) | (level_ids > 2)).any()):
            raise ValueError("Only stride8/16/32 levels are registered")
        if detach_inputs:
            box_logits = box_logits.detach()
            regression_features = regression_features.detach()
            class_logits = class_logits.detach()
        stats = self.distribution_statistics(box_logits).to(regression_features.dtype)
        reg = F.silu(self.reg_reduce(regression_features))
        if level_ids.ndim == 1:
            level_ids = level_ids.unsqueeze(0).expand(b, -1)
        level = F.one_hot(level_ids, 3).permute(0, 2, 1).to(reg.dtype)
        x = torch.cat((stats, reg, class_logits.sigmoid(), level), dim=1)
        return self.predictor(x)

    @staticmethod
    def score(class_logits, quality_logits, exponent=1.0):
        if class_logits.shape != quality_logits.shape or not 0 <= exponent <= 1:
            raise ValueError("Score shapes must agree; exponent must be in [0,1]")
        return class_logits.sigmoid() * quality_logits.sigmoid().pow(exponent)


class QualityTap:
    """Capture the existing regression tower; optionally rescore its decoded boxes.

    The parent remains the original Detect model. Install on a freshly loaded
    model for each validation mode; close removes all hooks. Readout weights are
    saved separately with the parent SHA256, avoiding checkpoint class rewrites.
    """

    def __init__(self, parent, readout=None, mode="capture", exponent=1.0):
        from ultralytics.nn.modules.head import Detect
        head = parent.model[-1]
        if type(head) is not Detect or head.nc != 2 or head.reg_max != 16 or head.end2end:
            raise ValueError("Quality experiment requires the original two-class YOLO11n Detect")
        if [int(x) for x in head.stride] != [8, 16, 32]:
            raise ValueError("Unexpected detection strides")
        if mode not in ("capture", "disabled", "pre_nms", "post_nms"):
            raise ValueError(mode)
        if mode in ("pre_nms", "post_nms") and readout is None:
            raise ValueError("Rescoring requires a trained readout")
        self.mode, self.readout, self.exponent = mode, readout, exponent
        self.regression = [None] * 3
        self.raw = self.features = self.level_ids = self.quality_logits = None
        self.hooks = []
        for i, tower in enumerate(head.cv2):
            if tower[1].conv.out_channels != 64:
                raise ValueError("Unexpected regression tower width")
            def capture(module, inputs, output, index=i):
                self.regression[index] = output.detach()
            self.hooks.append(tower[1].register_forward_hook(capture))
        self.hooks.append(head.register_forward_hook(self._head_output))

    def _head_output(self, module, inputs, output):
        if module.training or module.export or not isinstance(output, tuple):
            raise RuntimeError("Frozen quality adapter expects eval-mode native Detect output")
        decoded, raw = output
        self.raw = raw
        self.features = torch.cat([x.flatten(2) for x in self.regression], 2)
        self.level_ids = torch.cat([torch.full((x.shape[2] * x.shape[3],), i,
                                              device=x.device, dtype=torch.long)
                                    for i, x in enumerate(raw["feats"])])
        self.quality_logits = None
        if self.mode in ("pre_nms", "post_nms"):
            self.quality_logits = self.readout(raw["boxes"], self.features, raw["scores"], self.level_ids)
            if self.mode == "pre_nms":
                scores = self.readout.score(raw["scores"], self.quality_logits, self.exponent)
                return torch.cat((decoded[:, :4], scores), 1), raw
        return output

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


@torch.no_grad()
def sampled_quality_batch(tap, criterion, batch, topk=128):
    """Union TAL foreground and top scores; same-class max IoU soft targets.

    Pad only selected anchors. The returned mask normalizes each image equally,
    including negative images. Assignment always sees original class scores.
    """
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.metrics import box_iou
    raw = tap.raw
    distr = raw["boxes"].permute(0, 2, 1).contiguous()
    scores = raw["scores"].permute(0, 2, 1).contiguous()
    b, a, _ = scores.shape
    anchors, strides = make_anchors(raw["feats"], criterion.stride, .5)
    size = torch.tensor(batch["img"].shape[2:], device=scores.device, dtype=scores.dtype)
    labels = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    truth = criterion.preprocess(labels, b, scale_tensor=size[[1, 0, 1, 0]])
    gt_cls, gt_boxes = truth.split((1, 4), 2)
    gt_mask = gt_boxes.sum(2, keepdim=True).gt(0)
    boxes = criterion.bbox_decode(anchors, distr) * strides
    _, _, _, foreground, _ = criterion.assigner(
        scores.sigmoid(), boxes, anchors * strides, gt_cls, gt_boxes, gt_mask)
    top = scores.sigmoid().amax(2).topk(min(topk, a), dim=1).indices
    selected = [torch.unique(torch.cat((foreground[i].nonzero().flatten(), top[i])), sorted=True)
                for i in range(b)]
    k = max(len(x) for x in selected)
    indices = torch.zeros((b, k), dtype=torch.long, device=scores.device)
    mask = torch.zeros((b, k), dtype=torch.bool, device=scores.device)
    targets = torch.zeros((b, 2, k), device=scores.device)
    for i, ids in enumerate(selected):
        indices[i, :len(ids)] = ids
        mask[i, :len(ids)] = True
        for c in range(2):
            keep = gt_mask[i, :, 0] & (gt_cls[i, :, 0] == c)
            if keep.any():
                targets[i, c, :len(ids)] = box_iou(boxes[i, ids], gt_boxes[i, keep]).amax(1).clamp(0, 1)
    def gather(x):
        return x.gather(2, indices[:, None].expand(-1, x.shape[1], -1))
    inputs = (gather(raw["boxes"]), gather(tap.features), gather(raw["scores"]), tap.level_ids[indices])
    return inputs, targets, mask


def quality_loss(logits, targets, mask):
    per_anchor = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(1)
    return (per_anchor.mul(mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


def state_fingerprint(model):
    """Includes frozen parameters and BN buffers; excludes transient anchor caches."""
    import hashlib
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()
