"""Explicit training gate and reproducible run configuration."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .common import digest, offline_runtime, path, provenance, write_json
from .data import load_prepared


def plan(cfg, dataset, method, seed, tag="v1"):
    if dataset not in cfg["datasets"] or method not in cfg["methods"]:
        raise ValueError("Unknown dataset/method")
    if seed not in cfg["seeds"]:
        raise ValueError(f"Seed must belong to registered set {cfg['seeds']}")
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("tag must contain only letters, digits, underscores or hyphens")
    out = path(cfg["results_root"]) / tag / dataset / method / f"seed{seed}"
    return {
        "protocol": cfg["protocol"], "dataset": dataset, "method": method, "seed": seed,
        "tag": tag, "research": {**cfg["methods"][method], "topk": cfg["background_topk"]},
        "training": {**cfg["training"], "seed": seed},
        "data": str(path(cfg["prepared_root"]) / dataset / "data.yaml"),
        "weights": str(path(cfg["weights"])), "output": str(out),
        "training_started": False,
        "initialization": "shared YOLO11 layers 0..22; all detection heads initialized from scratch",
        "selection": "best validation mAP50-95; never select on test",
    }


def train(cfg, dataset, method, seed, tag, execute=False):
    spec = plan(cfg, dataset, method, seed, tag)
    if not execute:
        return spec
    if "," in str(spec["training"]["device"]):
        raise ValueError("v1 supports CPU/single GPU only; DDP is not validated")
    alpha = spec["research"]["background_alpha"]
    if alpha < 0 or spec["research"]["topk"] < 1:
        raise ValueError("Invalid background loss settings")
    if Path(spec["output"]).exists():
        raise FileExistsError("Run already exists. Use a new tag; v1 never silently resumes/overwrites.")
    if not Path(spec["weights"]).is_file():
        raise FileNotFoundError(spec["weights"])
    out, metadata, _ = load_prepared(cfg, dataset)
    offline_runtime()
    from .models import ResearchTrainer, architecture
    overrides = {
        **spec["training"], "model": architecture(spec["research"]["architecture"]),
        "data": str(out / "data.yaml"), "pretrained": spec["weights"],
        "project": str(Path(spec["output"]).parent), "name": Path(spec["output"]).name,
        "exist_ok": False, "resume": False, "task": "detect", "mode": "train", "split": "val",
    }
    # Construction is deliberately inside the --execute branch.
    trainer = ResearchTrainer(overrides=overrides, research=spec["research"])
    spec.update({"training_started": True, "provenance": provenance(), "data_metadata": metadata,
                 "pretrained_sha256": digest(spec["weights"]), "resolved_overrides": overrides})
    write_json(trainer.save_dir / "run.json", spec)
    (trainer.save_dir / "protocol.yaml").write_text(yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False))
    trainer.train()
    return {"output": str(trainer.save_dir), "training_finished": True}


def check_model(cfg, method, output, imgsz=128):
    """CPU-only random-tensor forward check; never constructs an optimizer."""
    if imgsz < 64 or imgsz % 32:
        raise ValueError("imgsz must be a multiple of 32 and at least 64")
    offline_runtime()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, init_seeds
    from .models import ResearchModel, architecture, shared_transfer
    torch.set_num_threads(2)
    init_seeds(0, deterministic=True)
    source = path(cfg["weights"])
    if not source.is_file():
        raise FileNotFoundError(source)
    method_cfg = cfg["methods"][method]
    model = ResearchModel(architecture(method_cfg["architecture"]), nc=2, verbose=False).cpu().eval()
    report = shared_transfer(model, YOLO(str(source)).model)
    with torch.inference_mode():
        output_tensor = model(torch.zeros(1, 3, imgsz, imgsz))
    prediction = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
    result = {"method": method, "parameters": sum(p.numel() for p in model.parameters()),
              "gflops_at_640": get_flops(model, 640), "strides": model.stride.tolist(),
              "forward_shape": list(prediction.shape), "check_imgsz": imgsz,
              "finite": bool(torch.isfinite(prediction).all()), "weight_transfer": report,
              "training_started": False, "provenance": provenance()}
    if not result["finite"]:
        raise RuntimeError("Non-finite model output")
    write_json(path(output), result)
    return result
