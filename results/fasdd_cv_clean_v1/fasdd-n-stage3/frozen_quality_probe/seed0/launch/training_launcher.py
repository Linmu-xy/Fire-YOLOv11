#!/usr/bin/env python3
"""Five-epoch frozen YOLO11n quality experiment; preview by default.

--check verifies real train batches without creating an experiment.
--execute fits on train-only hash folds then evaluates selected weights on val.
Foreground execution supports a detached external process and stdout capture.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from firesmoke.common import digest, offline_runtime, provenance, stable_digest, write_json
from firesmoke.data import load_prepared, load_protocol

CONFIG = ROOT / "configs/protocol-fasdd-n-quality-stage3.yaml"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def prepare(cfg, runtime):
    import torch
    from torch.utils.data import DataLoader, Subset
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset, seed_worker
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.loss import v8DetectionLoss
    from types import SimpleNamespace
    import firesmoke.models  # historical checkpoint registration
    from firesmoke.localization_quality import LocalizationQualityReadout, QualityTap, state_fingerprint

    data_cfg = load_protocol(ROOT / cfg["data_protocol"])
    prepared, metadata, rows = load_prepared(data_cfg, cfg["dataset"])
    version = json.loads((prepared / "version.json").read_text())
    if not (version["status"] == "built_and_integrity_verified" and
            version["portable_content_sha256"] == metadata["portable_content_sha256"] == cfg["portable_content_sha256"] and
            version["protocol_sha256"] == digest(ROOT / cfg["data_protocol"])):
        raise RuntimeError("Frozen data integrity/version mismatch")
    parent_file = ROOT / cfg["parent"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU0 is required")
    parent = YOLO(str(parent_file)).model.float().cuda().eval()
    if parent.names != {0: "fire", 1: "smoke"} or sum(p.numel() for p in parent.parameters()) != 2590230:
        raise RuntimeError("This experiment requires the exact original n baseline")
    parent.requires_grad_(False)
    parent.args = SimpleNamespace(**parent.args)
    criterion = v8DetectionLoss(parent)
    readout = LocalizationQualityReadout().cuda()
    tap = QualityTap(parent)
    data = check_det_dataset(str(prepared / "data.yaml"))
    args = get_cfg(overrides={"imgsz": cfg["training"]["imgsz"], "rect": False, "cache": False, "task": "detect"})
    dataset = build_yolo_dataset(args, data["train"], cfg["training"]["batch"], data, mode="val", rect=False)
    train_rows = {str(Path(r["prepared_image"])): r for r in rows if r["split"] == "train"}
    if set(dataset.im_files) != set(train_rows):
        raise RuntimeError("Training images differ from the prepared manifest")
    folds = {"fit": [], "heldout": []}
    fold_records = []
    for i, image in enumerate(dataset.im_files):
        identity = train_rows[image]["id"]
        fold = int(hashlib.sha256(("quality-seed0:" + identity).encode()).hexdigest(), 16) % 10
        partition = "heldout" if fold == 0 else "fit"
        folds[partition].append(i)
        fold_records.append({"id": identity, "fold": fold, "partition": partition})
    if any(not v for v in folds.values()) or len(dataset) != version["verification"]["counts"]["train"]:
        raise RuntimeError("Invalid train-only partition")
    # Subset does not inherit collate_fn. Pass the native function explicitly.
    loaders = {}
    for name, indices in folds.items():
        generator = torch.Generator().manual_seed(cfg["seed"])
        loaders[name] = DataLoader(Subset(dataset, indices), batch_size=cfg["training"]["batch"],
            shuffle=name == "fit", num_workers=cfg["training"]["workers"], pin_memory=True,
            collate_fn=dataset.collate_fn, worker_init_fn=seed_worker, generator=generator,
            persistent_workers=cfg["training"]["workers"] > 0)
    checks = {"parent_sha256": digest(parent_file), "parent_state_sha256": state_fingerprint(parent),
              "parent_parameters": 2590230, "trainable_parameters": sum(p.numel() for p in readout.parameters()),
              "train_only": True, "images": {name: len(ids) for name, ids in folds.items()},
              "split_sha256": stable_digest(fold_records), "portable_content_sha256": cfg["portable_content_sha256"],
              "gpu": torch.cuda.get_device_name(0)}
    if checks["trainable_parameters"] != 1802:
        raise RuntimeError("Readout parameter budget changed")
    write_json(runtime / "partition.json", fold_records)
    return parent, readout, tap, criterion, loaders, prepared, checks


def cuda_batch(batch):
    import torch
    return {k: v.cuda(non_blocking=True).float().div(255) if k == "img"
            else v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def check_integration(parent, readout, tap, criterion, loader, cfg):
    import torch
    from firesmoke.localization_quality import sampled_quality_batch, quality_loss, state_fingerprint
    from firesmoke.quality_validation import QualityValidator
    from ultralytics.models.yolo.detect.val import DetectionValidator
    frozen = state_fingerprint(parent)
    batch = cuda_batch(next(iter(loader)))
    with torch.no_grad():
        original = parent(batch["img"])[0].clone()
        inputs, targets, mask = sampled_quality_batch(tap, criterion, batch, cfg["training"]["topk"])
    assert targets.min() >= 0 and targets.max() <= 1 and mask.all(1).any()
    logits = readout(*inputs)
    loss = quality_loss(logits, targets, mask)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in readout.parameters())
    assert all(p.grad is None and not p.requires_grad for p in parent.parameters())
    assert not any(m.training for m in parent.modules())
    assert frozen == state_fingerprint(parent)
    original_readout = {k: v.detach().clone() for k, v in readout.state_dict().items()}
    probe_optimizer = torch.optim.AdamW(readout.parameters(), lr=cfg["training"]["lr"])
    probe_optimizer.step()
    assert any(not torch.equal(original_readout[k], v) for k, v in readout.state_dict().items())
    assert frozen == state_fingerprint(parent)
    readout.load_state_dict(original_readout)
    readout.zero_grad(set_to_none=True)
    # Capture-only and disabled outputs must be bit-identical on the same batch.
    tap.mode, tap.readout = "disabled", readout
    with torch.no_grad():
        disabled = parent(batch["img"])[0].clone()
        assert torch.equal(original, disabled)
        tap.mode, tap.exponent = "pre_nms", 0.0
        zero = parent(batch["img"])[0].clone()
        assert torch.equal(original, zero)
        tap.exponent = cfg["score_exponent"]
        enabled = parent(batch["img"])[0].clone()
        assert torch.equal(original[:, :4], enabled[:, :4])
        assert (enabled[:, 4:] <= original[:, 4:]).all()
        tap.mode = "post_nms"
        post = parent(batch["img"])[0].clone()
        assert torch.equal(original, post)
        validator_args = {**cfg["evaluation"], "task": "detect", "mode": "val"}
        native = DetectionValidator(args=validator_args, save_dir=Path(os.environ["MPLCONFIGDIR"]).parent / "check_native")
        native.nc, native.end2end = 2, False
        quality = QualityValidator(tap, args=validator_args, save_dir=Path(os.environ["MPLCONFIGDIR"]).parent / "check_quality")
        quality.nc, quality.end2end = 2, False
        before = native.postprocess(original.clone())
        after = quality.postprocess(post.clone())
        for x, y in zip(before, after):
            assert torch.equal(x["bboxes"], y["bboxes"]) and torch.equal(x["cls"], y["cls"])
            assert (y["conf"] <= x["conf"]).all()
    tap.mode, tap.readout = "capture", None
    # Exercise native AutoBackend fusion and metric collection on eight actual
    # train-internal examples; these are implementation checks, not val scores.
    from torch.utils.data import DataLoader, Subset
    from ultralytics import YOLO
    from firesmoke.localization_quality import QualityTap
    smoke_loader = DataLoader(Subset(loader.dataset, range(8)), batch_size=8,
                              collate_fn=loader.collate_fn, num_workers=0)
    native_modes = {}
    for mode in cfg["evaluation_order"]:
        smoke_parent = YOLO(str(ROOT / cfg["parent"])).model.float().cuda().eval()
        smoke_tap = QualityTap(smoke_parent, readout.eval(), mode, cfg["score_exponent"])
        smoke_validator = QualityValidator(smoke_tap, args={**validator_args, "data": str(ROOT / "data/prepared" / cfg["dataset"] / "data.yaml")},
            dataloader=smoke_loader, save_dir=Path(os.environ["MPLCONFIGDIR"]).parent / ("check_" + mode))
        try:
            smoke_validator(model=smoke_parent)
            smoke_summary = smoke_validator.summary()
            assert smoke_summary["images"] == 8
            native_modes[mode] = {"status": "passed", "images": 8}
        finally:
            smoke_tap.close()
        del smoke_parent, smoke_tap, smoke_validator
    readout.train()
    return {"status": "passed", "real_batch": len(batch["img"]), "loss": float(loss.detach()),
            "parent_frozen_state_identical": True, "disabled_tensor_exact": True,
            "zero_exponent_tensor_exact": True, "pre_nms_boxes_exact": True,
            "post_nms_boxes_classes_exact": True, "finite_readout_gradients": True,
            "optimizer_changed_only_readout": True, "native_validator_smoke": native_modes,
            "selected_anchors_per_image": mask.sum(1).tolist()}


def evaluate(cfg, output, prepared, parent_sha):
    import torch
    from ultralytics import YOLO
    from firesmoke.localization_quality import LocalizationQualityReadout, QualityTap
    from firesmoke.quality_validation import QualityValidator
    selected = torch.load(output / "weights/best_readout.pt", map_location="cpu", weights_only=False)
    if selected["parent_sha256"] != parent_sha or digest(ROOT / cfg["parent"]) != parent_sha:
        raise RuntimeError("Parent checkpoint changed before evaluation")
    results = {}
    for mode in cfg["evaluation_order"]:
        parent = YOLO(str(ROOT / cfg["parent"])).model.float().cuda().eval()
        readout = LocalizationQualityReadout().cuda().eval()
        readout.load_state_dict(selected["readout"], strict=True)
        tap = QualityTap(parent, readout, mode, cfg["score_exponent"])
        args = {**cfg["evaluation"], "data": str(prepared / "data.yaml"), "task": "detect", "mode": "val"}
        validator = QualityValidator(tap, args=args, save_dir=output / "evaluation" / mode)
        try:
            validator(model=parent)
            summary = validator.summary()
        finally:
            tap.close()
        results[mode] = summary
        write_json(output / "evaluation" / mode / "metrics.json", summary)
        print("VALIDATION " + json.dumps(summary), flush=True)
        if mode == "disabled" and abs(summary["mAP50_95"] - cfg["disabled_reference_mAP50_95"]) > cfg["disabled_reference_tolerance"]:
            raise RuntimeError("Disabled quality failed to reproduce historical baseline; stop interpretation")
        del parent, readout, validator, tap
        torch.cuda.empty_cache()
    gain = 100 * (results["pre_nms"]["mAP50_95"] - results["disabled"]["mAP50_95"])
    write_json(output / "evaluation/comparison.json", {"selected_epoch": selected["epoch"],
        "heldout_BCE": selected["best_heldout_BCE"], "gain_pp_pre_nms": gain,
        "expand_budget_gate_passed": gain >= cfg["minimum_gain_pp_to_expand"], "modes": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / "results" / cfg["dataset"] / cfg["tag"] / cfg["method"] / f"seed{cfg['seed']}"
    if not (args.check or args.execute):
        print(json.dumps({"protocol": cfg, "output": str(output)}, indent=2))
        return 0
    if args.execute and output.exists():
        raise FileExistsError("Preserve existing experiment: " + str(output))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime = ROOT.parent / (ROOT.name + "-runtime") / cfg["dataset"] / cfg["tag"] / cfg["method"] / f"seed{cfg['seed']}" / (("check_" if args.check else "fit_") + stamp)
    runtime.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    offline_runtime()
    import numpy as np
    import torch
    from firesmoke.localization_quality import sampled_quality_batch, quality_loss, state_fingerprint
    torch.set_num_threads(4)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.use_deterministic_algorithms(True, warn_only=False)
    parent, readout, tap, criterion, loaders, prepared, checks = prepare(cfg, runtime)
    validation = check_integration(parent, readout, tap, criterion, loaders["heldout"], cfg)
    write_json(runtime / "integration_validation.json", validation)
    print("PREFLIGHT " + json.dumps({"checks": checks, "integration": validation, "runtime": str(runtime)}), flush=True)
    if args.check:
        tap.close()
        return 0
    output.mkdir(parents=True, exist_ok=False)
    launch = output / "launch"
    launch.mkdir()
    (output / "weights").mkdir()
    hashes = {str(f.relative_to(ROOT)): digest(f) for pattern in ("firesmoke/*.py", "scripts/*.py", "configs/**/*.yaml")
              for f in ROOT.glob(pattern) if f.is_file()}
    with tarfile.open(launch / "source_snapshot.tar.gz", "w:gz") as bundle:
        for relative in sorted(hashes):
            bundle.add(ROOT / relative, arcname=relative, recursive=False)
    shutil.copy2(CONFIG, output / "protocol.yaml")
    shutil.copy2(__file__, launch / "training_launcher.py")
    shutil.copy2(runtime / "partition.json", launch / "train_only_partition.json")
    record = {"started_at": now(), "pid": os.getpid(), "protocol": cfg, "preflight": checks,
              "integration_validation": validation, "runtime": str(runtime), "output": str(output),
              "snapshot_source_sha256": hashes, "provenance": provenance(), "official_test_used": False}
    write_json(output / "run.json", record)
    write_json(runtime / "launch.json", record)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    best = float("inf")
    status = {"status": "failed", "started_at": now(), "pid": os.getpid()}
    try:
        with (output / "results.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "fit_BCE", "heldout_BCE", "fit_images", "heldout_images", "seconds"])
            writer.writeheader()
            for epoch in range(1, cfg["training"]["epochs"] + 1):
                start = time.monotonic()
                row = {"epoch": epoch}
                for partition in ("fit", "heldout"):
                    readout.train(partition == "fit")
                    total, count = 0., 0
                    for step, cpu in enumerate(loaders[partition], 1):
                        batch = cuda_batch(cpu)
                        with torch.no_grad():
                            parent(batch["img"])
                            inputs, targets, mask = sampled_quality_batch(tap, criterion, batch, cfg["training"]["topk"])
                        with torch.set_grad_enabled(partition == "fit"):
                            loss = quality_loss(readout(*inputs), targets, mask)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite quality loss")
                        if partition == "fit":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                        n = len(batch["img"])
                        total += float(loss.detach()) * n
                        count += n
                        if step == 1 or step % 100 == 0:
                            print(f"PROGRESS epoch={epoch}/5 partition={partition} step={step}/{len(loaders[partition])} images={count} BCE={total/count:.6f}", flush=True)
                    row[partition + "_BCE"], row[partition + "_images"] = total / count, count
                if state_fingerprint(parent) != checks["parent_state_sha256"] or any(p.grad is not None for p in parent.parameters()):
                    raise RuntimeError("Frozen parent state or gradients changed")
                row["seconds"] = time.monotonic() - start
                writer.writerow(row)
                stream.flush()
                improved = row["heldout_BCE"] < best
                best = min(best, row["heldout_BCE"])
                checkpoint = {"epoch": epoch, "readout": readout.state_dict(), "optimizer": optimizer.state_dict(),
                    "parent_sha256": checks["parent_sha256"], "parent_path": cfg["parent"], "protocol": cfg,
                    "best_heldout_BCE": best, "source_sha256": hashes, "split_sha256": checks["split_sha256"],
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                    "loader_generator_states": {k: v.generator.get_state() for k, v in loaders.items()}}
                temporary = runtime / "checkpoint.tmp"
                torch.save(checkpoint, temporary)
                os.replace(temporary, output / "weights/last_readout.pt")
                if improved:
                    shutil.copy2(output / "weights/last_readout.pt", output / "weights/best_readout.pt")
                print("EPOCH " + json.dumps(row), flush=True)
        if digest(ROOT / cfg["parent"]) != checks["parent_sha256"]:
            raise RuntimeError("Parent checkpoint file changed")
        write_json(output / "frozen_parent_verification.json", {"status": "passed",
            "parent_file_sha256": checks["parent_sha256"], "parent_state_sha256": state_fingerprint(parent),
            "trainable_parameters": 1802, "epochs": cfg["training"]["epochs"]})
        tap.close()
        del tap, parent, criterion, loaders, readout, optimizer, inputs, targets, mask, loss, checkpoint, batch, cpu
        torch.cuda.empty_cache()
        evaluate(cfg, output, prepared, checks["parent_sha256"])
        status["status"] = "completed"
    except BaseException as error:
        status["error"] = repr(error)
        raise
    finally:
        status["ended_at"] = now()
        write_json(output / "launch/status.json", status)
        write_json(runtime / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
