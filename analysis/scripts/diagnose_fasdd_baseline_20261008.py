#!/usr/bin/env python3
"""Validate a completed FASDD baseline at fixed inference resolutions; never train.

AP comes from the pinned Ultralytics validator. Additional object diagnostics
use a fixed confidence threshold and are reported separately from AP.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[640, 800, 960])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--rect", action="store_true", help="Use rectangular validation batches")
    parser.add_argument("--audit-conf", type=float, default=0.25)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    runtime = repo.parent / (repo.name + "-runtime") / "baseline-diagnosis"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    from firesmoke.common import offline_runtime
    from firesmoke.data import load_prepared, load_protocol
    import firesmoke.models  # register checkpoint classes
    offline_runtime()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou

    if any(s < 64 or s % 32 for s in args.sizes):
        raise ValueError("Resolution must be a positive multiple of 32")
    args.output.mkdir(parents=True, exist_ok=False)
    source_run = json.loads((args.checkpoint.parent.parent / "run.json").read_text())
    if source_run["dataset"] != "fasdd_cv_clean_v1":
        raise ValueError("This analysis is registered for fasdd_cv_clean_v1")
    cfg = load_protocol(args.checkpoint.parent.parent / "protocol.yaml")
    prepared, metadata, rows = load_prepared(cfg, source_run["dataset"])
    baseline_finished = args.checkpoint.parent.parent / "launch" / "status.json"
    if not baseline_finished.is_file() or json.loads(baseline_finished.read_text())["status"] != "completed":
        raise RuntimeError("Baseline launcher has not completed")
    names = ["fire", "smoke"]
    size_names = ["lt16", "16to32", "32to64", "64to128", "ge128"]
    manifest_val = {Path(r["prepared_image"]).name: r for r in rows if r["split"] == "val"}

    def pairs(iou, pred_cls, gt_cls, threshold):
        correct = (gt_cls[:, None] == pred_cls[None, :]) & (iou >= threshold)
        ij = np.argwhere(correct)
        if len(ij) == 0:
            return np.empty((0, 2), dtype=int)
        scores = iou[ij[:, 0], ij[:, 1]]
        matched = ij[np.argsort(scores)[::-1]]
        matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
        matched = matched[np.unique(matched[:, 0], return_index=True)[1]]
        return matched

    class AuditValidator(DetectionValidator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.audit_rows = []
            self.audit_classes = {name: collections.Counter() for name in names}
            self.audit_sizes = np.zeros((2, 5, 3), dtype=np.int64)
            self.negative_images = 0
            self.negative_alarm_images = 0
            self.negative_class_alarms = [0, 0]

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                truth = self._prepare_batch(si, batch)
                filtered = {k: v[pred["conf"] >= args.audit_conf] for k, v in pred.items()}
                gt_cls = truth["cls"].cpu().numpy().astype(int)
                pred_cls = filtered["cls"].cpu().numpy().astype(int)
                gt_boxes = truth["bboxes"]
                pred_boxes = filtered["bboxes"]
                iou = box_iou(gt_boxes, pred_boxes).cpu().numpy()
                match = {t: pairs(iou, pred_cls, gt_cls, t) for t in (0.5, 0.75, 0.9, 0.95)}
                missed = sorted(set(range(len(gt_cls))) - set(match[0.5][:, 0].tolist()))
                fps = sorted(set(range(len(pred_cls))) - set(match[0.5][:, 1].tolist()))
                gt_native = ops.scale_boxes(truth["imgsz"], gt_boxes.clone(), truth["ori_shape"], ratio_pad=truth["ratio_pad"]).cpu().numpy()
                pred_native = ops.scale_boxes(truth["imgsz"], pred_boxes.clone(), truth["ori_shape"], ratio_pad=truth["ratio_pad"]).cpu().numpy()
                manifest_boxes = np.asarray(manifest_val[Path(truth["im_file"]).name]["boxes"], dtype=float).reshape(-1, 5)
                if len(manifest_boxes) != len(gt_cls) or not np.array_equal(manifest_boxes[:, 0], gt_cls):
                    raise ValueError("Validator label order differs from registered manifest")
                h, w = truth["ori_shape"]
                # Bin membership comes from original manifest geometry, so it
                # cannot change with integer resize rounding across resolutions.
                long640 = np.maximum(manifest_boxes[:, 3]*w, manifest_boxes[:, 4]*h) * 640 / max(h, w)
                bins = np.digitize(long640, [16, 32, 64, 128])
                categories = {}
                for c, name in enumerate(names):
                    counter = self.audit_classes[name]
                    counter["gt"] += int((gt_cls == c).sum())
                    counter["predictions"] += int((pred_cls == c).sum())
                    for t, matched in match.items():
                        counter[f"tp_iou{t:g}"] += int((gt_cls[matched[:, 0]] == c).sum())
                    for b in range(5):
                        self.audit_sizes[c, b, 0] += int(((gt_cls == c) & (bins == b)).sum())
                        for j, t in enumerate((0.5, 0.75), 1):
                            ids = match[t][:, 0]
                            self.audit_sizes[c, b, j] += int(((gt_cls[ids] == c) & (bins[ids] == b)).sum())
                for gi in missed:
                    same = iou[gi, pred_cls == gt_cls[gi]]
                    other = iou[gi, pred_cls != gt_cls[gi]]
                    sm = float(same.max()) if same.size else 0.0
                    om = float(other.max()) if other.size else 0.0
                    if sm >= 0.5:
                        reason = "competing_sameclass_candidate"
                    elif om >= 0.5:
                        reason = "wrongclass_overlap"
                    elif sm >= 0.1:
                        reason = "weak_localization_candidate"
                    else:
                        reason = "no_close_sameclass_candidate"
                    categories[str(gi)] = {"category": reason, "max_sameclass_iou": sm, "max_otherclass_iou": om}
                    self.audit_classes[names[int(gt_cls[gi])]][reason] += 1
                if len(gt_cls) == 0:
                    self.negative_images += 1
                    self.negative_alarm_images += int(len(pred_cls) > 0)
                    for c in range(2):
                        self.negative_class_alarms[c] += int((pred_cls == c).any())
                self.audit_rows.append({"image": truth["im_file"], "original_shape_hw": list(truth["ori_shape"]),
                    "gt_classes": gt_cls.tolist(), "gt_xyxy": gt_native.tolist(),
                    "pred_classes": pred_cls.tolist(), "pred_xyxy": pred_native.tolist(),
                    "pred_conf": filtered["conf"].cpu().tolist(), "missed_gt": missed,
                    "false_prediction_indices": fps, "missed_candidates": categories})

        def get_stats(self):
            result = super().get_stats()
            self.metrics.baseline_audit = {"conf": args.audit_conf,
                "classes": {k: dict(v) for k, v in self.audit_classes.items()},
                "size_bins_standardized_longside_at640": size_names,
                "size_counts_gt_tp50_tp75": self.audit_sizes.tolist(),
                "negative_images": self.negative_images,
                "negative_alarm_images": self.negative_alarm_images,
                "negative_class_alarm_images": self.negative_class_alarms,
                "note": "Fixed-conf matching diagnostics, not AP. Missed-candidate categories are geometric clues, not causal error labels."}
            with (self.save_dir / "audit_images.jsonl").open("x", encoding="utf-8") as stream:
                for row in self.audit_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            return result

    settings = {"split": "val", "conf": 0.001, "iou": 0.7, "max_det": 300,
                "half": False, "rect": args.rect, "augment": False, "batch": args.batch,
                "workers": 8, "device": "0"}
    header = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "training_seed": source_run["seed"], "training_imgsz": source_run["training"]["imgsz"],
        "dataset": source_run["dataset"], "portable_content_sha256": metadata["portable_content_sha256"],
        "ultralytics": ultralytics.__version__, "settings": settings,
        "validation_images": sum(r["split"] == "val" for r in rows),
        "purpose": "Inference-resolution sensitivity of the SAME 640-trained checkpoint; this does not replace a high-resolution training experiment."}
    write_json(args.output / "registration.json", header)
    summaries = []
    for size in args.sizes:
        print("START_SIZE", size, flush=True)
        model = YOLO(str(args.checkpoint))
        if [model.names[i] for i in range(2)] != names:
            raise ValueError("Checkpoint class order changed")
        torch.cuda.reset_peak_memory_stats()
        m = model.val(validator=AuditValidator, data=str(prepared / "data.yaml"), imgsz=size,
            **settings, project=str(args.output), name=f"val_{size}", exist_ok=False,
            plots=True, save_json=False, verbose=True)
        per_class = {}
        for i, c in enumerate(m.box.ap_class_index):
            per_class[names[int(c)]] = {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                "AP50": float(m.box.all_ap[i, 0]), "AP75": float(m.box.all_ap[i, 5]),
                "AP50_95": float(m.box.all_ap[i].mean()), "AP_by_IoU": m.box.all_ap[i].tolist()}
        summary = {**header, "validation_imgsz": size,
            "metrics": {k: float(v) for k, v in m.results_dict.items()},
            "IoU_thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)],
            "per_class": per_class, "audit": m.baseline_audit,
            "speed_ms_per_image": m.speed,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}
        write_json(args.output / f"metrics_{size}.json", summary)
        summaries.append(summary)
        print("DONE_SIZE", size, json.dumps(per_class), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(args.output / "summary.json", {"evaluations": summaries})
    print("ANALYSIS_COMPLETED", args.output, flush=True)


if __name__ == "__main__":
    main()
