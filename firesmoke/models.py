"""Version-pinned YOLO adapter: native P2 graph and training-only interventions."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch import nn
import ultralytics.nn.tasks as ultralytics_tasks
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ROOT as ULTRA_ROOT
from ultralytics.utils.loss import v8DetectionLoss

from .common import NAMES, ROOT, write_json
from .band_refinement import DetailConditionedBandResidual
from .aligned_context import AlignedContextResidual


def architecture(name):
    if name.startswith("p2_acr"):
        modes = {"p2_acr": "full", "p2_acr_no_align": "no_align",
                 "p2_acr_uniform": "uniform", "p2_acr_local": "local"}
        if name not in modes:
            raise ValueError(name)
        return str(ROOT / "configs/models" / ("yolo11n-" + name.replace("_", "-") + ".yaml"))
    if name == "baseline":
        return str(ULTRA_ROOT / "cfg/models/11/yolo11n.yaml")
    if name == "p2":
        return str(ROOT / "configs/models/yolo11n-p2.yaml")
    if name == "p2_srdg":
        return str(ROOT / "configs/models/yolo11n-p2-srdg.yaml")
    if name in ("p2_dcbr", "p2_dcbr_semantic", "p2_dcbr_whole"):
        return str(ROOT / "configs/models" / ("yolo11n-" + name.replace("_", "-") + ".yaml"))
    raise ValueError(name)


class SemanticResidualDetailGate(nn.Module):
    """Use coarse semantic context to modulate shallow detail without erasing it.

    Input channels are ordered ``[semantic, detail]``.  A zero-initialized 1x1
    projection makes the module an exact identity at initialization.  The
    multiplicative factor is bounded to ``[1-gain, 1+gain]`` by tanh, so the P2
    detail path cannot collapse to zero when gain < 1.
    """
    def __init__(self, semantic_channels: int, detail_channels: int, gain: float = 0.5):
        super().__init__()
        if semantic_channels < 1 or detail_channels < 1:
            raise ValueError("semantic_channels and detail_channels must be positive")
        if not 0 < gain < 1:
            raise ValueError("gain must be in (0, 1)")
        self.semantic_channels = semantic_channels
        self.detail_channels = detail_channels
        self.gain = float(gain)
        # Do not consume the global initialization stream. With the same seed,
        # all downstream P2/Detect parameters therefore match the parent P2 arm.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv2d(semantic_channels, detail_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.semantic_channels + self.detail_channels
        if x.ndim != 4 or x.shape[1] != expected:
            raise ValueError(f"expected BCHW input with {expected} channels, got {tuple(x.shape)}")
        semantic, detail = x.split((self.semantic_channels, self.detail_channels), dim=1)
        modulation = 1.0 + self.gain * torch.tanh(self.projection(semantic))
        return torch.cat((semantic, detail * modulation), dim=1)


# Ultralytics resolves YAML module names from ultralytics.nn.tasks globals.  Registering
# the class in memory keeps site-packages unchanged and makes the version-pinned model
# graph reproducible from this repository.
ultralytics_tasks.SemanticResidualDetailGate = SemanticResidualDetailGate
ultralytics_tasks.DetailConditionedBandResidual = DetailConditionedBandResidual
ultralytics_tasks.AlignedContextResidual = AlignedContextResidual


def shared_transfer(model, source):
    """Same initialization policy for both arms: backbone+neck 0..22 only.

    Never silently load a similarly-shaped but semantically different head.
    """
    target = model.state_dict()
    copied = {k: v for k, v in source.float().state_dict().items()
              if k.startswith("model.") and int(k.split(".")[1]) < 23
              and k in target and target[k].shape == v.shape}
    if not copied:
        raise ValueError("No compatible shared YOLO11 weights")
    model.load_state_dict(copied, strict=False)
    return {"policy": "shared_layers_0_to_22_only", "tensors": len(copied),
            "elements": sum(v.numel() for v in copied.values()), "keys": sorted(copied)}


def background_penalty(scores, batch_idx, topk):
    """Mean top-K class logits BCE on *post-augmentation empty* images.

    scores: [B,C,A]. Empty batches and batches without negatives are supported.
    Every class has the same K budget; gradients remain attached to hard logits.
    """
    present = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    present[batch_idx.long().flatten()] = True
    if present.all():
        return scores.sum() * 0
    k = min(topk, scores.shape[-1])
    return F.softplus(scores[~present].float().topk(k, dim=-1).values).mean()


class BackgroundLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.alpha = model.research_alpha
        self.topk = model.research_topk

    def loss(self, preds, batch):
        losses, _ = super().loss(preds, batch)
        # Upstream returns the 3-element loss vector scaled by batch size.
        extra = self.alpha * background_penalty(preds["scores"], batch["batch_idx"], self.topk)
        losses = losses + torch.stack((extra*0, extra*preds["scores"].shape[0], extra*0))
        return losses, losses.detach() / preds["scores"].shape[0]


class ResearchModel(DetectionModel):
    def init_criterion(self):
        if getattr(self, "research_alpha", 0.0) > 0:
            return BackgroundLoss(self)
        return super().init_criterion()


def style_augment(images):
    """Source-only photometric perturbation, not target-domain adaptation.

    No geometry changes; valid boxes remain valid. Torch RNG follows run seed.
    """
    b = images.shape[0]
    shape = (b, 1, 1, 1)
    gamma = torch.empty(shape, device=images.device).uniform_(0.7, 1.5)
    contrast = torch.empty(shape, device=images.device).uniform_(0.7, 1.3)
    transformed = images.clamp(1e-6, 1).pow(gamma)
    mean = transformed.mean((2, 3), keepdim=True)
    transformed = ((transformed-mean)*contrast+mean).clamp(0, 1)
    gray = (transformed * transformed.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    transformed = torch.where(torch.rand(shape, device=images.device) < 0.1, gray, transformed)
    return torch.where(torch.rand(shape, device=images.device) < 0.5, transformed, images)


class ResearchTrainer(DetectionTrainer):
    def __init__(self, *args, research=None, **kwargs):
        self.research = research or {"background_alpha": 0.0, "topk": 32, "style": False}
        super().__init__(*args, **kwargs)
        self.research_images = 0
        self.research_empty = 0
        self.add_callback("on_train_epoch_end", _log_background_exposure)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ResearchModel(cfg, nc=2, ch=3, verbose=verbose)
        model.names = dict(enumerate(NAMES))
        model.research_alpha = self.research["background_alpha"]
        model.research_topk = self.research["topk"]
        if weights is not None:
            report = shared_transfer(model, weights)
            write_json(self.save_dir / "weight_transfer.json", report)
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self.research_images += batch["img"].shape[0]
        self.research_empty += batch["img"].shape[0] - batch["batch_idx"].unique().numel()
        if self.research["style"]:
            batch["img"] = style_augment(batch["img"])
        return batch


def _log_background_exposure(trainer):
    """Mosaic can remove empty images; report actual loss exposure each epoch."""
    with (trainer.save_dir / "background_exposure.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"epoch": trainer.epoch + 1, "images": trainer.research_images,
                                 "post_augmentation_empty_images": trainer.research_empty,
                                 "alpha": trainer.research["background_alpha"]}) + "\n")
    trainer.research_images = trainer.research_empty = 0
