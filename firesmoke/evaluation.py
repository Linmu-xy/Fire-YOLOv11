"""Frozen checkpoint evaluation, forward latency, and seed aggregation."""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .common import NAMES, digest, fresh_dir, offline_runtime, path, provenance, read_json, stable_digest, write_json
from .data import load_prepared


def load_model(weights):
    offline_runtime()
    from ultralytics import YOLO
    weights = path(weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights))
    names = [model.names[i] for i in range(len(model.names))]
    if names != NAMES:
        raise ValueError(f"Checkpoint names {names} != canonical {NAMES}. Legacy weights require explicit migration, not relabeling.")
    return model


def evaluate(cfg, weights, dataset, split, output, device="cpu", imgsz=640, batch=16):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    model = load_model(weights)
    source_run = path(weights).parent.parent / "run.json"
    if not source_run.is_file():
        raise ValueError("Missing run.json next to checkpoint's weights directory; use the registered training entry point")
    run = read_json(source_run)
    data, metadata, rows = load_prepared(cfg, dataset)
    selected = [r for r in rows if r["split"] == split]
    out = fresh_dir(path(output))
    settings = {"imgsz": imgsz, "confidence_floor": 0.001, "nms_iou": 0.7, "max_det": 300,
                "augment": False, "half": False, "rect": False}
    metrics = model.val(data=str(data / "data.yaml"), split=split, imgsz=imgsz, batch=batch,
                        device=device, conf=settings["confidence_floor"], iou=settings["nms_iou"],
                        max_det=300, augment=False, half=False, rect=False, plots=True, workers=0,
                        project=str(out), name="detection", exist_ok=False, save_json=False)
    header = {"schema": 1, "dataset": dataset, "split": split, "names": NAMES,
              "checkpoint_sha256": digest(path(weights)), "manifest_sha256": metadata["manifest_sha256"],
              "training_dataset": run["dataset"], "method": run["method"], "seed": run["seed"],
              "tag": run["tag"], "settings": settings, "provenance": provenance(),
              "training_recipe_sha256": stable_digest({k: v for k, v in run["training"].items() if k != "seed"}),
              "training_manifest_sha256": run["data_metadata"]["manifest_sha256"],
              "research_settings": run["research"]}
    images = []
    # Separate predict pass provides stable image-level artifacts, including empty images.
    for start in range(0, len(selected), batch):
        chunk = selected[start:start+batch]
        results = model.predict(source=[r["prepared_image"] for r in chunk], imgsz=imgsz, batch=batch,
                                device=device, conf=0.001, iou=0.7, max_det=300, augment=False,
                                half=False, rect=False, verbose=False, save=False, stream=True)
        produced = 0
        for row, result in zip(chunk, results):
            boxes = result.boxes
            predictions = [[int(c), *[float(v) for v in xy], float(p)]
                           for c, xy, p in zip(boxes.cls.cpu().tolist(), boxes.xywhn.cpu().tolist(), boxes.conf.cpu().tolist())]
            images.append({"id": row["id"], "group": row["group"], "truth": row["boxes"], "predictions": predictions})
            produced += 1
        if produced != len(chunk):
            raise RuntimeError("Predictor returned an incomplete batch")
    write_json(out / "predictions.json", {**header, "images": images})
    per_class = {}
    for i, c in enumerate(metrics.box.ap_class_index):
        per_class[NAMES[int(c)]] = dict(zip(("precision", "recall", "AP50", "AP50_95"),
                                          map(float, metrics.box.class_result(i))))
    result = {**header, "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
              "per_class": per_class, "image_count": len(images),
              "note": "Detection AP is uncalibrated. Image alarm calibration is reported separately."}
    write_json(out / "metrics.json", result)
    return result


def benchmark(weights, output, device="cpu", imgsz=640, warmup=50, iterations=200, threads=2):
    if warmup < 1 or iterations < 2 or threads < 1 or imgsz < 64 or imgsz % 32:
        raise ValueError("Invalid benchmark settings")
    model = load_model(weights)
    import torch
    torch.set_num_threads(threads)
    device = torch.device("cuda:"+device if device.isdigit() else device)
    network = model.model.to(device).float().eval().fuse(verbose=False)
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            network(x)
        sync()
        for _ in range(iterations):
            sync()
            start = time.perf_counter()
            network(x)
            sync()
            times.append((time.perf_counter()-start)*1000)
    ordered = sorted(times)
    result = {"checkpoint_sha256": digest(path(weights)), "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "measurement": "FP32 fused model forward only; excludes preprocessing, NMS, IO and alarms",
              "batch": 1, "imgsz": imgsz, "threads": threads, "warmup": warmup, "iterations": iterations,
              "mean_ms": statistics.mean(times), "std_ms": statistics.stdev(times),
              "median_ms": statistics.median(times), "p95_ms": ordered[int(0.95*(iterations-1))],
              "forward_fps": 1000/statistics.mean(times), "samples_ms": times,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "provenance": provenance()}
    write_json(path(output), result)
    return result


def summarize(files, output, expected_seeds=(0, 1, 2)):
    groups = defaultdict(list)
    for filename in files:
        row = read_json(path(filename))
        if row["split"] != "test":
            raise ValueError("Main result table accepts test metrics only")
        key = (row["tag"], row["training_dataset"], row["dataset"], row["method"],
               row["manifest_sha256"], json.dumps(row["settings"], sort_keys=True),
               row.get("training_recipe_sha256"), row.get("training_manifest_sha256"),
               json.dumps(row.get("research_settings"), sort_keys=True))
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        seeds = [r["seed"] for r in rows]
        if sorted(seeds) != sorted(expected_seeds):
            raise ValueError(f"Expected exactly seeds {expected_seeds}, got {seeds} for {key[:4]}")
        if len({r["checkpoint_sha256"] for r in rows}) != len(rows):
            raise ValueError("Same checkpoint reused as independent seeds")
        names = set(rows[0]["metrics"])
        if any(set(r["metrics"]) != names for r in rows):
            raise ValueError("Incompatible metric keys")
        stats = {name: {"mean": statistics.mean(r["metrics"][name] for r in rows),
                        "std_ddof1": statistics.stdev(r["metrics"][name] for r in rows)} for name in sorted(names)}
        result.append({"tag": key[0], "source": key[1], "target": key[2], "method": key[3],
                       "seeds": seeds, "n": len(rows), "metrics": stats})
    write_json(path(output), {"groups": result, "note": "Seed SD and test-sampling uncertainty are distinct."})
    return result
