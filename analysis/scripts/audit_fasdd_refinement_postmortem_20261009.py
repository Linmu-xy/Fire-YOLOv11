#!/usr/bin/env python3
"""Read-only checkpoint reconstruction and native/bypass validation diagnostics."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def sha256(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(file, value):
    Path(file).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("detail", "context"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(args.output / "matplotlib")
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.nn.tasks import load_checkpoint
    from firesmoke.models import ResearchTrainer
    from scripts.resume_fasdd_n_refinement import restore_full_model_state

    torch.set_num_threads(2)
    torch.manual_seed(0)
    original_hash = sha256(args.checkpoint)
    report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "method": args.method, "checkpoint": str(args.checkpoint),
              "checkpoint_sha256_before": original_hash,
              "ultralytics_version": ultralytics.__version__,
              "note": "Final-checkpoint reconstruction; not a tensor comparison against the overwritten pause checkpoint.",
              "source": {"get_model": inspect.getsource(ResearchTrainer.get_model),
                         "setup_model": inspect.getsource(BaseTrainer.setup_model),
                         "load_checkpoint_state": inspect.getsource(BaseTrainer._load_checkpoint_state)}}
    source, _ = load_checkpoint(str(args.checkpoint), device="cpu", fuse=False)
    source = source.float().eval()
    state = source.state_dict()
    with tempfile.TemporaryDirectory(prefix="fasdd_resume_audit_") as directory:
        stub = ResearchTrainer.__new__(ResearchTrainer)
        stub.research = {"background_alpha": 0., "topk": 32, "style": False,
                         "refinement": args.method, "refinement_hidden": 16}
        stub.save_dir = Path(directory)
        rebuilt = stub.get_model(cfg=source.yaml, weights=source, verbose=False).float().eval()
        observed = rebuilt.state_dict()
        groups = {
            "backbone_neck_0_to_22": lambda key: int(key.split(".")[1]) < 23,
            "predictor_head": lambda key: key.startswith("model.23.") and ".refinement." not in key,
            "new_refinement": lambda key: key.startswith("model.23.refinement.")}
        report["old_resume_reconstruction"] = {}
        for group, predicate in groups.items():
            keys = [key for key in state if predicate(key)]
            unequal = [key for key in keys if not torch.equal(state[key], observed[key])]
            report["old_resume_reconstruction"][group] = {
                "tensors": len(keys), "different_tensors": len(unequal),
                "elements": sum(state[key].numel() for key in keys),
                "different_tensor_elements": sum(state[key].numel() for key in unequal),
                "different_keys": unequal}
        projections = {key: {"checkpoint_l2": float(tensor.float().norm()),
                             "old_reconstruction_l2": float(observed[key].float().norm())}
                       for key, tensor in state.items() if "refinement" in key and "output" in key}
        report["output_projection_norms"] = projections
        report["corrected_restore"] = restore_full_model_state(rebuilt, source)
        sample = torch.rand(1, 3, 128, 128)
        with torch.inference_mode():
            expected, actual = source(sample)[0], rebuilt(sample)[0]
        report["corrected_restore"].update(
            prediction_max_abs_error=float((expected - actual).abs().max()),
            predictions_exactly_equal=bool(torch.equal(expected, actual)))
        if not report["corrected_restore"]["predictions_exactly_equal"]:
            raise RuntimeError("Restored predictor output differs")
    write(args.output / "reconstruction.json", report)
    print("RECONSTRUCTION " + json.dumps({k: v for k, v in report.items() if k not in {"source", "output_projection_norms"}}, ensure_ascii=False), flush=True)
    del rebuilt, source, state, observed
    if args.validate:
        data = ROOT / "data/prepared/fasdd_cv_clean_v1/data.yaml"
        for variant in ("native", "bypass"):
            wrapper = YOLO(str(args.checkpoint))
            branch = wrapper.model.model[-1].refinement
            measurements, gates, routes = [], [], []
            hooks = []
            if variant == "native":
                # AutoBackend performs a synthetic warmup before validation.
                # Capture real val batches only; warmup can use torch.empty input.
                sampling = {"active": False}
                wrapper.add_callback("on_val_batch_start", lambda validator: sampling.update(active=True))
                wrapper.add_callback("on_val_batch_end", lambda validator: sampling.update(active=False))
                def hook(module, inputs, output):
                    if not sampling["active"] or len(measurements) >= 8:
                        return
                    features = inputs[0]
                    regression, classification = output
                    def ratios(values):
                        return [float((value - feature).float().square().mean().sqrt() /
                                      feature.float().square().mean().sqrt().clamp_min(1e-12))
                                for value, feature in zip(values, features)]
                    measurements.append({"batch_size": int(features[0].shape[0]),
                                         "reg_delta_over_feature_rms": ratios(regression),
                                         "cls_delta_over_feature_rms": ratios(classification)})
                hooks.append(branch.register_forward_hook(hook))
                if branch.detail is not None:
                    def gate_hook(module, inputs, output):
                        if sampling["active"] and len(gates) < 8:
                            gates.append(float(output.sigmoid().mean()))
                    hooks.append(branch.detail.gate.register_forward_hook(gate_hook))
                if branch.context is not None:
                    for level, route in enumerate(branch.context.routes):
                        def route_hook(module, inputs, output, level=level):
                            if sampling["active"] and sum(item["level"] == level for item in routes) < 8:
                                weights = output.softmax(dim=1)
                                routes.append({"level": level,
                                               "mean_source_weights": weights.mean((0, 2, 3)).tolist(),
                                               "normalized_entropy": float(-(weights * weights.clamp_min(1e-12).log()).sum(1).mean() / __import__("math").log(3))})
                        hooks.append(route.register_forward_hook(route_hook))
            else:
                def bypass(self, features, detail=None):
                    return list(features), list(features)
                branch.forward = types.MethodType(bypass, branch)
            metrics = wrapper.val(data=str(data), split="val", imgsz=640, batch=64,
                                  device="0", workers=8, rect=True, half=False,
                                  conf=.001, iou=.7, max_det=300, augment=False,
                                  plots=False, save_json=False, verbose=False,
                                  project=str(args.output), name=variant, exist_ok=False)
            names = metrics.names
            per_class = []
            for row, cls in enumerate(metrics.box.ap_class_index):
                per_class.append({"class_id": int(cls), "name": names[int(cls)],
                                  "precision": float(metrics.box.p[row]),
                                  "recall": float(metrics.box.r[row]),
                                  "AP50": float(metrics.box.all_ap[row, 0]),
                                  "AP75": float(metrics.box.all_ap[row, 5]),
                                  "AP50_95": float(metrics.box.all_ap[row].mean())})
            result = {"variant": variant, "metrics": metrics.results_dict,
                      "per_class": per_class, "sampled_batches": measurements,
                      "detail_gate_mean": gates, "context_routes": routes,
                      "speed": metrics.speed,
                      "protocol": {"split": "val", "imgsz": 640, "batch": 64,
                                   "rect": True, "conf": .001, "iou": .7,
                                   "max_det": 300, "half": False, "augment": False},
                      "interpretation": "Posthoc bypass in a jointly trained model; not a separately trained no-module ablation."}
            write(args.output / f"{variant}.json", result)
            print("VALIDATION " + json.dumps({"variant": variant, "metrics": metrics.results_dict, "per_class": per_class}), flush=True)
            for hook in hooks:
                hook.remove()
            del wrapper, branch, metrics
            torch.cuda.empty_cache()
    report["checkpoint_sha256_after"] = sha256(args.checkpoint)
    report["checkpoint_unchanged"] = report["checkpoint_sha256_after"] == original_hash
    if not report["checkpoint_unchanged"]:
        raise RuntimeError("Checkpoint changed during read-only audit")
    write(args.output / "reconstruction.json", report)


if __name__ == "__main__":
    main()
