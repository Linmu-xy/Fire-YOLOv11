#!/usr/bin/env python3
"""Read-only, fixed-protocol AP and error audit of completed FASDD checkpoints.

The GT-IoU score experiment is a diagnostic oracle AFTER NMS, never a deployment
metric or an estimate of achievable improvement. It keeps boxes/TP assignment.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(args.output / "matplotlib")
    from firesmoke.common import offline_runtime
    from firesmoke.data import load_prepared, load_protocol
    import firesmoke.models  # noqa: F401; checkpoint class registration
    offline_runtime()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou

    if ultralytics.__version__ != "8.4.42":
        raise RuntimeError("This audit requires the pinned Ultralytics 8.4.42")
    torch.set_num_threads(4)
    run_dir = args.checkpoint.parent.parent
    run = json.loads((run_dir / "run.json").read_text())
    cfg = load_protocol(run_dir / "protocol.yaml")
    prepared, metadata, rows = load_prepared(cfg, run["dataset"])
    val_rows = {Path(r["prepared_image"]).name: r for r in rows if r["split"] == "val"}
    original_hash = sha256(args.checkpoint)
    names = {0: "fire", 1: "smoke"}
    thresholds = (.001, .25)
    size_names = ["lt16", "16to32", "32to64", "64to128", "ge128"]
    counts = {str(t): {"classes": np.zeros((2, 4), dtype=np.int64),
                        "sizes": np.zeros((2, 5, 3), dtype=np.int64),
                        "scenes": {s: np.zeros((2, 3), dtype=np.int64)
                                   for s in ("fire_only", "smoke_only", "mixed", "negative")},
                        "negative_images": 0, "negative_alarm_images": 0}
              for t in thresholds}
    coverage = np.zeros((2, 4), dtype=np.int64)
    score_stats = {name: [] for name in names.values()}
    ap_stats = []
    residuals = []
    image_stream = (args.output / "gt_geometry.jsonl").open("x", encoding="utf-8")

    def pairs(iou, pc, gc, threshold):
        ij = np.argwhere((gc[:, None] == pc[None, :]) & (iou >= threshold))
        if not len(ij):
            return np.empty((0, 2), dtype=int)
        matched = ij[np.argsort(iou[ij[:, 0], ij[:, 1]])[::-1]]
        matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
        return matched[np.unique(matched[:, 0], return_index=True)[1]]

    class AuditValidator(DetectionValidator):
        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, raw in enumerate(preds):
                truth = self._prepare_batch(si, batch)
                pred = self._prepare_pred(raw)
                gc = truth["cls"].cpu().numpy().astype(int)
                pc = pred["cls"].cpu().numpy().astype(int)
                conf = pred["conf"].cpu().numpy()
                iou = box_iou(truth["bboxes"], pred["bboxes"]).cpu().numpy()
                same = (gc[:, None] == pc[None, :])
                same_iou = iou * same
                quality = same_iou.max(axis=0) if len(gc) else np.zeros(len(pc))
                ap_stats.append((self._process_batch(pred, truth)["tp"], conf,
                                 pc, gc, conf * quality))
                scene = ("negative" if not len(gc) else "mixed" if len(set(gc)) == 2
                         else "fire_only" if gc[0] == 0 else "smoke_only")
                manifest = np.asarray(val_rows[Path(truth["im_file"]).name]["boxes"]).reshape(-1, 5)
                if len(manifest) != len(gc) or not np.array_equal(manifest[:, 0], gc):
                    raise RuntimeError("Manifest/validator label order mismatch")
                h, w = truth["ori_shape"]
                long640 = np.maximum(manifest[:, 3] * w, manifest[:, 4] * h) * 640 / max(h, w)
                bins = np.digitize(long640, [16, 32, 64, 128])
                best_iou = same_iou.max(axis=1) if len(pc) else np.zeros(len(gc))
                for c in range(2):
                    coverage[c, 0] += int((gc == c).sum())
                    for col, cutoff in enumerate((.1, .5, .75), 1):
                        coverage[c, col] += int(((gc == c) & (best_iou >= cutoff)).sum())
                image_stream.write(json.dumps({"image": Path(truth["im_file"]).name,
                    "scene": scene, "gt_class": gc.tolist(), "longside640": long640.tolist(),
                    "max_sameclass_post_nms_iou": best_iou.tolist()}, allow_nan=False) + "\n")
                for threshold in thresholds:
                    keep = conf >= threshold
                    current_pc = pc[keep]
                    current_iou = iou[:, keep]
                    item = counts[str(threshold)]
                    matches = {cutoff: pairs(current_iou, current_pc, gc, cutoff)
                               for cutoff in (.5, .75, .9)}
                    for c in range(2):
                        item["classes"][c, 0] += int((gc == c).sum())
                        for col, cutoff in enumerate((.5, .75, .9), 1):
                            ids = matches[cutoff][:, 0]
                            item["classes"][c, col] += int((gc[ids] == c).sum())
                        item["scenes"][scene][c, 0] += int((gc == c).sum())
                        for b in range(5):
                            item["sizes"][c, b, 0] += int(((gc == c) & (bins == b)).sum())
                        for col, cutoff in enumerate((.5, .75), 1):
                            ids = matches[cutoff][:, 0]
                            item["scenes"][scene][c, col] += int((gc[ids] == c).sum())
                            for b in range(5):
                                item["sizes"][c, b, col] += int(((gc[ids] == c) & (bins[ids] == b)).sum())
                    if not len(gc):
                        item["negative_images"] += 1
                        item["negative_alarm_images"] += int(keep.any())
                # IoU distribution of one-to-one close candidates, fixed conf .25.
                keep = conf >= .25
                matched = pairs(iou[:, keep], pc[keep], gc, .1)
                for gi, pi in matched:
                    score_stats[names[int(gc[gi])]].append(
                        [float(conf[keep][pi]), float(iou[:, keep][gi, pi])])

    model = YOLO(str(args.checkpoint))
    if model.names != names:
        raise RuntimeError("Class order changed")
    active = {"real_batch": False}
    model.add_callback("on_val_batch_start", lambda _: active.update(real_batch=True))
    model.add_callback("on_val_batch_end", lambda _: active.update(real_batch=False))
    hooks = []
    branch = getattr(model.model.model[-1], "refinement", None)
    if branch is not None:
        def capture(module, inputs, output):
            if not active["real_batch"] or len(residuals) >= 8:
                return
            features = inputs[0]
            def ratio(values):
                return [float((v - f).float().square().mean().sqrt() /
                              f.float().square().mean().sqrt().clamp_min(1e-12))
                        for v, f in zip(values, features)]
            residuals.append({"reg_ratio": ratio(output[0]), "cls_ratio": ratio(output[1])})
        hooks.append(branch.register_forward_hook(capture))
    settings = dict(split="val", imgsz=640, batch=64, device="0", workers=8,
                    rect=True, half=False, conf=.001, iou=.7, max_det=300,
                    augment=False, plots=False, save_json=False, verbose=False)
    try:
        metrics = model.val(validator=AuditValidator, data=str(prepared / "data.yaml"),
                            **settings, project=str(args.output), name="native", exist_ok=False)
    finally:
        image_stream.close()
        for hook in hooks:
            hook.remove()

    def per_class(ap, indices):
        return {names[int(c)]: {"AP50": float(ap[i, 0]), "AP75": float(ap[i, 5]),
                               "AP50_95": float(ap[i].mean()), "AP_by_IoU": ap[i].tolist()}
                for i, c in enumerate(indices)}
    tp, conf, pc, gc, oracle = [np.concatenate([row[i] for row in ap_stats]) for i in range(5)]
    original_ap = ap_per_class(tp, conf, pc, gc, plot=False, names=names)
    oracle_ap = ap_per_class(tp, oracle, pc, gc, plot=False, names=names)
    # Preserve every TP50/FP50 score slot and the classwise score distribution;
    # reorder ONLY the TP50 confidence values by GT-IoU. This separates box
    # quality ordering from the oracle's removal of unmatched predictions.
    within_tp50 = conf.copy()
    quality = np.divide(oracle, conf, out=np.zeros_like(oracle), where=conf > 0)
    for c in np.unique(pc):
        indices = np.flatnonzero((pc == c) & tp[:, 0])
        slots = np.sort(conf[indices])[::-1]
        order = indices[np.argsort(-quality[indices], kind="stable")]
        within_tp50[order] = slots
    within_ap = ap_per_class(tp, within_tp50, pc, gc, plot=False, names=names)
    fp_suppression_ap = ap_per_class(tp, conf * tp[:, 0], pc, gc, plot=False, names=names)
    if not np.allclose(original_ap[5], metrics.box.all_ap, atol=1e-10):
        raise RuntimeError("Collected prediction stats do not reproduce official AP")
    for item in counts.values():
        item["classes"] = item["classes"].tolist()
        item["sizes"] = item["sizes"].tolist()
        item["scenes"] = {k: v.tolist() for k, v in item["scenes"].items()}
    score_summary = {}
    for name, values in score_stats.items():
        a = np.asarray(values).reshape(-1, 2)
        score_summary[name] = {"count": len(a),
            "mean_iou": float(a[:, 1].mean()) if len(a) else None,
            "pearson_score_iou": float(np.corrcoef(a.T)[0, 1]) if len(a) > 1 else None,
            "fraction_iou_lt75": float((a[:, 1] < .75).mean()) if len(a) else None}
    report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": original_hash,
        "method": run["method"], "seed": run["seed"], "settings": settings,
        "ultralytics": ultralytics.__version__, "validation_images": len(val_rows),
        "portable_content_sha256": metadata["portable_content_sha256"],
        "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "per_class": per_class(metrics.box.all_ap, metrics.box.ap_class_index),
        "fixed_conf_diagnostics": counts,
        "columns_classes": ["gt", "tp50", "tp75", "tp90"],
        "columns_sizes_scenes": ["gt", "tp50", "tp75"], "size_bins": size_names,
        "gt_coverage_counts": coverage.tolist(),
        "gt_coverage_columns": ["gt", "any_sameclass_iou10", "any_sameclass_iou50", "any_sameclass_iou75"],
        "close_candidate_score_iou": score_summary, "residual_sample": residuals,
        "oracle_post_nms_conf_times_gt_iou": {
            "mAP50_95": float(oracle_ap[5].mean()),
            "per_class": per_class(oracle_ap[5], oracle_ap[6]),
            "note": "GT-required diagnostic; fixed post-NMS boxes and original TP assignment. Not a deployable metric or expected module gain."},
        "oracle_quality_order_within_tp50": {
            "mAP50_95": float(within_ap[5].mean()),
            "per_class": per_class(within_ap[5], within_ap[6]),
            "AP50_unchanged": bool(np.allclose(within_ap[5][:, 0], original_ap[5][:, 0], atol=1e-10)),
            "note": "GT-required reordering only within original TP50 score slots; boxes, FP50 scores and the classwise score distribution stay fixed."},
        "oracle_suppress_fp50": {
            "mAP50_95": float(fp_suppression_ap[5].mean()),
            "note": "GT-required suppression of original FP50 scores; diagnostic only."},
        "checkpoint_unchanged": sha256(args.checkpoint) == original_hash}
    if not report["checkpoint_unchanged"]:
        raise RuntimeError("Checkpoint changed during read-only audit")
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "ranking_stats.npz", tp=tp, conf=conf,
                        pred_cls=pc, target_cls=gc, max_sameclass_iou=quality)
    print("AUDIT_COMPLETE", json.dumps({"output": str(args.output), "per_class": report["per_class"],
          "oracle_mAP": report["oracle_post_nms_conf_times_gt_iou"]["mAP50_95"]}), flush=True)


if __name__ == "__main__":
    main()
