#!/usr/bin/env python3
"""Validate transfer semantics and the quality readout; no Trainer or training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(args.output.parent / "matplotlib")
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO
    from firesmoke.common import digest, offline_runtime
    from firesmoke.models import ResearchModel, architecture, shared_transfer
    from firesmoke.initialization_experiments import POLICIES, transfer_head, TransferExperimentTrainer
    from firesmoke.localization_quality import LocalizationQualityReadout
    offline_runtime()
    torch.set_num_threads(2)
    weights = ROOT / "weights/yolo11n.pt"
    source = YOLO(str(weights)).model.float()
    results = {}
    baseline = None
    for policy in POLICIES:
        torch.manual_seed(0)
        model = ResearchModel(architecture("baseline"), nc=2, verbose=False).float().eval()
        shared = shared_transfer(model, source)
        extra = transfer_head(model, source, policy)
        state = model.state_dict()
        if baseline is None:
            baseline = {k: v.clone() for k, v in state.items()}
        trunk_equal = all(torch.equal(v, baseline[k]) for k, v in state.items()
                          if int(k.split(".")[1]) < 23)
        cls_equal = all(torch.equal(v, baseline[k]) for k, v in state.items()
                        if k.startswith("model.23.cv3."))
        assert trunk_equal, "Transfer ablation changed the shared trunk"
        if policy != "all_shape_compatible":
            assert cls_equal, "Regression-only transfer changed classification initialization"
        with torch.inference_mode():
            prediction = model(torch.zeros(1, 3, 128, 128))[0]
        assert prediction.shape == (1, 6, 336) and torch.isfinite(prediction).all()
        assert sum(p.numel() for p in model.parameters()) == 2590230
        with tempfile.TemporaryDirectory(prefix="fasdd_stage3_transfer_") as directory:
            stub = TransferExperimentTrainer.__new__(TransferExperimentTrainer)
            stub.research = {"background_alpha": 0., "topk": 32, "style": False, "initialization": policy}
            stub.save_dir = Path(directory)
            stub.resume = False
            torch.manual_seed(0)
            actual = stub.get_model(architecture("baseline"), source, verbose=False).float().eval()
            assert all(torch.equal(v, actual.state_dict()[k]) for k, v in state.items())
            assert json.loads((stub.save_dir / "weight_transfer.json").read_text())["policy"] == policy
            stub.resume = True
            restored = stub.get_model(actual.yaml, actual, verbose=False).float().eval()
            assert all(torch.equal(v, restored.state_dict()[k]) for k, v in actual.state_dict().items())
        results[policy] = {"shared_tensors": shared["tensors"],
                           "extra_head_tensors": extra["extra_head_tensors"],
                           "extra_head_state_elements": extra["extra_head_state_elements"],
                           "trunk_equal_to_control": trunk_equal,
                           "classification_equal_to_control": cls_equal,
                           "actual_trainer_get_model_matches_expected": True,
                           "resume_model_state_exact": True,
                           "forward_shape": list(prediction.shape), "finite": True}
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    q = LocalizationQualityReadout()
    assert torch.equal(before, torch.get_rng_state()), "Readout construction changed global RNG"
    boxes = torch.randn(2, 64, 21, requires_grad=True)
    reg = torch.randn(2, 64, 21, requires_grad=True)
    cls = torch.randn(2, 2, 21, requires_grad=True)
    levels = torch.tensor([0] * 16 + [1] * 4 + [2])
    logits = q(boxes, reg, cls, levels)
    assert logits.shape == (2, 2, 21) and torch.isfinite(logits).all()
    assert torch.equal(q.score(cls, logits, 0), cls.sigmoid()), "Disabled quality changes scores"
    loss = F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits))
    loss.backward()
    gradients = {k: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
                 for k, p in q.named_parameters()}
    assert all(gradients.values()), "Inactive quality parameters"
    assert all(x.grad is None for x in (boxes, reg, cls)), "Detached readout changed parent gradients"
    uniform = q.distribution_statistics(torch.zeros(1, 64, 3)).reshape(1, 4, 6, 3)
    assert torch.allclose(uniform[:, :, -1], torch.ones(1, 4, 3), atol=1e-6)
    extreme = torch.full((1, 4, 16, 3), -10000.)
    extreme[:, :, 7, :] = 10000.
    peaked = q.distribution_statistics(extreme.reshape(1, 64, 3)).reshape(1, 4, 6, 3)
    assert torch.isfinite(peaked).all() and torch.allclose(peaked[:, :, -1], torch.zeros(1, 4, 3))
    clone = LocalizationQualityReadout()
    clone.load_state_dict(q.state_dict(), strict=True)
    with torch.inference_mode():
        assert torch.equal(q(boxes, reg, cls, levels), clone(boxes, reg, cls, levels))
    quality = {"parameters": sum(p.numel() for p in q.parameters()), "global_rng_preserved": True,
               "parent_gradients_unchanged_in_detached_mode": True, "gradient_tensors": gradients,
               "uniform_and_peaked_distribution_checks": "passed", "state_roundtrip_equal": True,
               "exponent_zero_equal_to_original_scores": True,
               "integration_status": "Standalone readout only; not installed in Detect, no accuracy claim"}
    assert quality["parameters"] == 1802
    if torch.cuda.is_available():
        q = q.cuda()
        q.zero_grad(set_to_none=True)
        full_boxes = torch.randn(32, 64, 8400, device="cuda")
        full_reg = torch.randn(32, 64, 8400, device="cuda")
        full_cls = torch.randn(32, 2, 8400, device="cuda")
        full_levels = torch.tensor([0] * 6400 + [1] * 1600 + [2] * 400, device="cuda")
        full_logits = q(full_boxes, full_reg, full_cls, full_levels)
        assert full_logits.shape == (32, 2, 8400) and torch.isfinite(full_logits).all()
        F.binary_cross_entropy_with_logits(full_logits, torch.rand_like(full_logits)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in q.parameters())
        quality["cuda_fp32_batch32_8400_anchor_forward_backward"] = "passed"
    else:
        quality["cuda_fp32_batch32_8400_anchor_forward_backward"] = "unavailable"
    report = {"weights_sha256": digest(weights), "transfer": results, "quality": quality,
              "training_started": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
