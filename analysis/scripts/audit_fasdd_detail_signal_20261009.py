#!/usr/bin/env python3
"""Measure detail branch activations on a fixed, balanced val diagnostic sample."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "true")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.tasks import load_checkpoint
    from firesmoke.models import ResearchTrainer

    torch.set_num_threads(2)
    checkpoint = ROOT / "results/fasdd_cv_clean_v1/fasdd-n-refinement-v1/detail/seed0/weights/best.pt"
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, _ = load_checkpoint(str(checkpoint), device="cuda:0", fuse=False)
    model = model.float().eval()
    branch = model.model[-1].refinement
    groups = {name: [] for name in ("background", "fire", "smoke", "both")}
    for line in (ROOT / "data/prepared/fasdd_cv_clean_v1/manifest.jsonl").open():
        row = json.loads(line)
        if row["split"] != "val":
            continue
        classes = {int(box[0]) for box in row["boxes"]}
        group = "background" if not classes else "both" if classes == {0, 1} else "fire" if classes == {0} else "smoke"
        groups[group].append(row)
    selected = []
    for group, rows in groups.items():
        for row in sorted(rows, key=lambda item: hashlib.sha256(item["id"].encode()).hexdigest())[:32]:
            selected.append({"group": group, **row})
    measurements, feature_refs, current_stages = [], [], {}
    captured = {}
    def rms(tensor):
        return float(tensor.float().square().mean().sqrt())
    def branch_input_hook(module, inputs):
        features, detail = inputs
        captured["features"], captured["detail"] = features, detail
        feature_refs.append({"p2_rms": rms(detail), "p3_rms": rms(features[0])})
    hooks = [branch.register_forward_pre_hook(branch_input_hook)]
    for name, layer in branch.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            def stage_hook(module, inputs, output, name=name):
                current_stages[name] = {"input_rms": rms(inputs[0]), "output_rms": rms(output)}
            hooks.append(layer.register_forward_hook(stage_hook))
    letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="fasdd_detail_reset_reference_") as directory:
        stub = ResearchTrainer.__new__(ResearchTrainer)
        stub.research = {"background_alpha": 0., "topk": 32, "style": False,
                         "refinement": "detail", "refinement_hidden": 16}
        stub.save_dir = Path(directory)
        reset_model = stub.get_model(cfg=model.yaml, weights=model, verbose=False)
        reference = reset_model.model[-1].refinement.to("cuda:0").float().eval()
        del reset_model
    initial_reference = []
    with torch.inference_mode():
        for start in range(0, len(selected), 16):
            images = []
            for row in selected[start:start+16]:
                image = cv2.imread(row["prepared_image"])
                if image is None:
                    raise RuntimeError(row["prepared_image"])
                images.append(np.ascontiguousarray(letterbox(image=image)[:, :, ::-1].transpose(2, 0, 1)))
            batch = torch.from_numpy(np.stack(images)).to("cuda:0").float() / 255
            model(batch)
            measurements.append(dict(current_stages))
            stages = {}
            reference_hooks = []
            for name, layer in reference.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    def reference_hook(module, inputs, output, name=name):
                        stages[name] = {"input_rms": rms(inputs[0]), "output_rms": rms(output)}
                    reference_hooks.append(layer.register_forward_hook(reference_hook))
            reference(captured["features"], captured["detail"])
            initial_reference.append(stages)
            for hook in reference_hooks:
                hook.remove()
    for hook in hooks:
        hook.remove()
    parameters = {}
    with torch.no_grad():
        for name, parameter in branch.named_parameters():
            initial = dict(reference.named_parameters())[name]
            fitted_scale = float((parameter * initial).sum() / initial.square().sum()) if bool(initial.square().sum() > 0) else None
            cosine = float(torch.nn.functional.cosine_similarity(parameter.flatten(), initial.flatten(), dim=0)) if fitted_scale is not None else None
            parameters[name] = {"trained_l2": float(parameter.norm()),
                                "fresh_reference_l2": float(initial.norm()),
                                "trained_rms": rms(parameter),
                                "cosine_with_reset_reference": cosine,
                                "best_fit_scale_vs_reset_reference": fitted_scale,
                                "relative_error_from_scaled_reference": rms(parameter - fitted_scale * initial) / max(rms(parameter), 1e-12) if fitted_scale is not None else None}
    report = {"sample": [{"id": row["id"], "group": row["group"]} for row in selected],
              "sample_policy": "32 per background/fire-only/smoke-only/mixed, SHA256(id) ordered; diagnostic sample, not a prevalence estimate.",
              "preprocessing": "640 square LetterBox RGB /255, unfused float32 model, no gradients or optimizer steps",
              "fresh_reference_note": "Seed0 old-get_model reconstruction on the same backbone features. This is a reconstruction, not an archived pause/init tensor snapshot.",
              "feature_rms": feature_refs, "trained_stage_rms": measurements,
              "fresh_reference_stage_rms": initial_reference, "parameters": parameters,
              "checkpoint_sha256_before": before,
              "checkpoint_sha256_after": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    if report["checkpoint_sha256_before"] != report["checkpoint_sha256_after"]:
        raise RuntimeError("Checkpoint changed")
    (args.output / "signal.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"parameters": parameters, "features_first_batch": feature_refs[0],
                      "trained_first_batch": measurements[0], "fresh_reference_first_batch": initial_reference[0]}), flush=True)


if __name__ == "__main__":
    main()
