#!/usr/bin/env python3
"""CUDA checks, measured budgets, optional ONNX parity and real-data trainer pilot."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--pilot", choices=("detail", "context", "full", "local_control"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(args.output / "matplotlib")
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import get_flops, init_seeds
    from firesmoke.models import ResearchTrainer
    from firesmoke.pyramid_refinement import install_refinement
    init_seeds(0, deterministic=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device = torch.device("cuda:" + args.device if args.device.isdigit() else args.device)
    parent = YOLO(str(args.checkpoint)).model.float().cpu().eval()
    report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "checkpoint": str(args.checkpoint),
              "device": str(device), "checks": [], "pilot": None,
              "note": "Engineering validation only; pilot accuracy is not a benchmark result."}
    def save():
        (args.output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    for mode in ("baseline", "detail", "context", "full", "local_control"):
        network = copy.deepcopy(parent)
        spec = None if mode == "baseline" else install_refinement(network, mode)
        item = {"mode": mode, "refinement": spec,
                "parameters": sum(p.numel() for p in network.parameters()), "gflops_640": get_flops(network, 640)}
        network = network.to(device).eval().fuse(verbose=False)
        x = torch.rand(1, 3, 640, 640, device=device)
        with torch.inference_mode():
            for _ in range(20):
                network(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(100):
                started = time.perf_counter()
                prediction = network(x)[0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append((time.perf_counter()-started)*1000)
            item.update(inference_shape=list(prediction.shape), inference_finite=bool(torch.isfinite(prediction).all()),
                        latency_median_ms=statistics.median(timings), latency_p95_ms=sorted(timings)[94],
                        latency_protocol="fused FP32, batch1, 640 square, 20 warmup, 100 sequential forwards, CUDA sync",
                        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        if not item["inference_finite"] or item["inference_shape"] != [1, 6, 8400]:
            raise RuntimeError("Invalid prediction in " + mode)
        # Exercise exact screening batch32 and the actual detection loss on CUDA.
        train_model = copy.deepcopy(parent)
        if mode != "baseline":
            install_refinement(train_model, mode)
        train_model = train_model.to(device).train()
        # Official strip_optimizer freezes parameters in completed checkpoints.
        # A training check must restore trainability, keeping DFL's fixed basis frozen.
        train_model.requires_grad_(True)
        train_model.model[-1].dfl.requires_grad_(False)
        train_model.args = get_cfg()
        images = torch.rand(32, 3, 640, 640, device=device)
        batch = {"img": images, "batch_idx": torch.arange(32, device=device).float(),
                 "cls": (torch.arange(32, device=device) % 2).view(-1, 1).float(),
                 "bboxes": torch.tensor([.5, .5, .2, .25], device=device).repeat(32, 1)}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses, _ = v8DetectionLoss(train_model)(train_model(images), batch)
        losses.sum().backward()
        item.update(batch32_loss=[float(v) for v in losses.detach()],
                    batch32_gradients_finite=all(bool(torch.isfinite(p.grad).all()) for p in train_model.parameters() if p.grad is not None),
                    peak_training_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        if not torch.isfinite(losses).all() or not item["batch32_gradients_finite"]:
            raise RuntimeError("Invalid batch32 loss/gradient in " + mode)
        del train_model, images, batch, losses, network, prediction, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.export and mode != "baseline":
            import numpy as np
            import onnxruntime as ort
            export_network = copy.deepcopy(parent).cpu().eval()
            install_refinement(export_network, mode)
            # Nonzero outputs force the exported graph to exercise refinements.
            for name, parameter in export_network.model[-1].refinement.named_parameters():
                if "output" in name:
                    torch.nn.init.normal_(parameter, std=.005)
            export_network.args = get_cfg()
            checkpoint = args.output / f"export_{mode}.pt"
            torch.save({"model": export_network, "train_args": vars(export_network.args), "version": "8.4.42"}, checkpoint)
            wrapper = YOLO(str(checkpoint))
            exported = wrapper.export(format="onnx", imgsz=(384, 640), batch=1, device="cpu",
                                      simplify=False, dynamic=False, opset=17, verbose=False)
            session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
            sample = torch.randn(1, 3, 384, 640)
            with torch.inference_mode():
                expected = wrapper.model.cpu().eval()(sample)
                if isinstance(expected, tuple):
                    expected = expected[0]
            observed = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
            delta = np.abs(observed - expected.numpy())
            agrees = bool(np.allclose(observed, expected.numpy(), rtol=1e-3, atol=1e-3))
            item.update(onnx=str(exported), onnx_shape=list(observed.shape),
                        onnx_max_abs_error=float(delta.max()), onnx_parity=agrees)
            if not agrees:
                raise RuntimeError("ONNX parity failed in " + mode)
            del session, wrapper, export_network, sample
        report["checks"].append(item)
        save()
        print("CHECK " + json.dumps(item), flush=True)
    if args.pilot:
        manifest = ROOT / "data/prepared/fasdd_cv_clean_v1/manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        fixture = args.output / "real_data_fixture"
        selection = []
        for split in ("train", "val"):
            counts = {"background": 0, "fire": 0, "smoke": 0, "both": 0}
            for row in rows:
                if row["split"] != split:
                    continue
                classes = {int(box[0]) for box in row["boxes"]}
                group = "background" if not classes else "both" if classes == {0, 1} else "fire" if classes == {0} else "smoke"
                if counts[group] >= 4:
                    continue
                counts[group] += 1
                image = fixture / "images" / split / Path(row["prepared_image"]).name
                label = fixture / "labels" / split / (image.stem + ".txt")
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.symlink_to(row["prepared_image"])
                label.write_text(Path(row["prepared_label"]).read_text())
                selection.append({"id": row["id"], "split": split, "group": group,
                                  "source_image": row["prepared_image"], "fixture_image": str(image)})
            if list(counts.values()) != [4, 4, 4, 4]:
                raise RuntimeError("Could not form balanced engineering fixture")
        import yaml
        data = fixture / "data.yaml"
        data.write_text(yaml.safe_dump({"path": str(fixture), "train": "images/train", "val": "images/val",
                                       "names": {0: "fire", 1: "smoke"}}, sort_keys=False))
        (fixture / "selection.json").write_text(json.dumps(selection, indent=2))
        overrides = {"model": str(Path(__import__('ultralytics').__file__).parent / "cfg/models/11/yolo11n.yaml"),
                     "pretrained": str(ROOT / "weights/yolo11n.pt"), "data": str(data), "device": args.device,
                     "epochs": 1, "imgsz": 128, "batch": 4, "workers": 0, "amp": False,
                     "deterministic": True, "seed": 0, "optimizer": "SGD", "lr0": .01,
                     "mosaic": 0., "close_mosaic": 0, "plots": False, "cache": False,
                     "project": str(args.output), "name": "trainer_pilot", "exist_ok": False, "verbose": False}
        trainer = ResearchTrainer(overrides=overrides, research={"background_alpha": 0., "topk": 32,
                   "style": False, "refinement": args.pilot, "refinement_hidden": 16})
        trainer.train()
        pilot_checkpoint = Path(trainer.save_dir) / "weights/best.pt"
        reloaded = YOLO(str(pilot_checkpoint)).model.to(device).eval()
        with torch.inference_mode():
            prediction = reloaded(torch.rand(1, 3, 128, 128, device=device))[0]
        report["pilot"] = {"mode": args.pilot, "training_finished": True, "checkpoint": str(pilot_checkpoint),
                           "reloaded_prediction_finite": bool(torch.isfinite(prediction).all()),
                           "fixture_train_images": 16, "fixture_val_images": 16, "epochs": 1, "imgsz": 128,
                           "purpose": "Trainer/optimizer/EMA/save/strip/final-validator integration, not accuracy measurement"}
        if not report["pilot"]["reloaded_prediction_finite"]:
            raise RuntimeError("Stripped pilot checkpoint failed reload")
        save()
    report["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["status"] = "passed"
    save()
    print("VALIDATION_PASSED " + str(args.output / "validation.json"), flush=True)


if __name__ == "__main__":
    main()
