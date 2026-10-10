"""Isolated YOLO11n head-transfer ablations; historical trainers stay unchanged."""
from __future__ import annotations

import json

import torch
from ultralytics.nn.modules.head import Detect

from .common import write_json
from .models import ResearchTrainer


POLICIES = ("shared_layers_0_to_22_only", "regression_head", "all_shape_compatible")


def transfer_head(model, source, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown transfer policy: " + policy)
    target_head, source_head = model.model[-1], source.model[-1]
    for head in (target_head, source_head):
        if type(head) is not Detect or head.i != 23 or head.f != [16, 19, 22]:
            raise ValueError("Head transfer is restricted to the standard YOLO11n graph")
        if head.stride.tolist() != [8., 16., 32.] or head.reg_max != 16:
            raise ValueError("Incompatible prediction semantics")
        if tuple(layer[0].conv.in_channels for layer in head.cv2) != (64, 128, 256):
            raise ValueError("Only n-scale prediction channels are registered")
    if target_head.nc != 2:
        raise ValueError("The target must have two classes")
    original, target = source.float().state_dict(), model.state_dict()
    copied, decisions = {}, []
    for key in sorted(k for k in target if k.startswith("model.23.")):
        permitted = policy == "all_shape_compatible" or (
            policy == "regression_head" and key.startswith(("model.23.cv2.", "model.23.dfl.")))
        reason = ("excluded_by_policy" if not permitted else "absent_from_source" if key not in original
                  else "shape_mismatch" if target[key].shape != original[key].shape else "copied")
        if reason == "copied":
            copied[key] = original[key]
        decisions.append({"key": key, "target_shape": list(target[key].shape),
                          "source_shape": list(original[key].shape) if key in original else None,
                          "decision": reason})
    model.load_state_dict(copied, strict=False)
    if any(not torch.equal(model.state_dict()[k], v) for k, v in copied.items()):
        raise RuntimeError("Head transfer tensor verification failed")
    return {"policy": policy, "extra_head_tensors": len(copied),
            "extra_head_state_elements": sum(v.numel() for v in copied.values()),
            "decisions": decisions, "copied_keys": sorted(copied)}


class TransferExperimentTrainer(ResearchTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if weights is not None and not bool(getattr(self, "resume", False)):
            policy = self.research.get("initialization", POLICIES[0])
            extra = transfer_head(model, weights, policy)
            file = self.save_dir / "weight_transfer.json"
            base = json.loads(file.read_text())
            # The repository writer creates files exclusively. Preserve the
            # parent trainer's original record before creating the final policy.
            file.rename(file.with_name("weight_transfer_shared.json"))
            write_json(file, {"policy": policy, "shared": base, "head": extra})
        return model
