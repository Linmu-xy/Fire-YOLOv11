#!/usr/bin/env python3
"""Combine saved official AP and fixed-confidence diagnostics; no model inference."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    for seed in (0, 1):
        payload = json.loads((args.inputs / f"seed{seed}" / "diagnostic_rect_summary.json").read_text())
        evaluations[seed] = {r["validation_imgsz"]: r for r in payload["evaluations"]}
    mean = statistics.mean
    resolution = []
    sizes = (640, 800, 960)
    for size in sizes:
        rows = [evaluations[s][size] for s in (0, 1)]
        row = {"imgsz": size, "mAP50_95": mean(r["metrics"]["metrics/mAP50-95(B)"] for r in rows),
               "mAP50": mean(r["metrics"]["metrics/mAP50(B)"] for r in rows), "classes": {}}
        for name in ("fire", "smoke"):
            row["classes"][name] = {key: mean(r["per_class"][name][key] for r in rows)
                for key in ("AP50", "AP75", "AP50_95", "precision", "recall")}
            row["classes"][name]["AP_by_IoU"] = [mean(r["per_class"][name]["AP_by_IoU"][i] for r in rows) for i in range(10)]
        resolution.append(row)
    for row in resolution:
        row["delta_mAP50_95_pp_vs640"] = 100 * (row["mAP50_95"] - resolution[0]["mAP50_95"])
    fixed = []
    for seed in (0, 1):
        row = evaluations[seed][640]["audit"]
        bins = row["size_bins_standardized_longside_at640"]
        out = {"seed": seed, "conf": row["conf"], "negative_image_alarm_fraction": row["negative_alarm_images"] / row["negative_images"], "classes": {}}
        for cls, name in enumerate(("fire", "smoke")):
            c = row["classes"][name].copy()
            for iou in (0.5, 0.75, 0.9, 0.95):
                c[f"recall_iou{iou}"] = c[f"tp_iou{iou}"] / c["gt"]
            c["size_bins"] = [{"bin": b, "gt": gt, "tp50": tp50, "tp75": tp75,
                "recall50": tp50 / gt, "recall75": tp75 / gt}
                for b, (gt, tp50, tp75) in zip(bins, row["size_counts_gt_tp50_tp75"][cls])]
            out["classes"][name] = c
        fixed.append(out)
    review = json.loads((args.inputs / "seed1" / "error_review.json").read_text())
    context = {}
    for group, counts in review["size_groups_gt_tp50"].items():
        context[group] = [{"bin": b, "gt": gt, "tp50": tp, "recall50": tp / gt if gt else None}
                          for b, (gt, tp) in zip(bins, counts)]
    context_total = {}
    for group, row in review["groups"].items():
        for name in ("fire", "smoke"):
            if row[f"{name}_gt"]:
                context_total[f"{group}/{name}"] = 1 - row[f"{name}_fn"] / row[f"{name}_gt"]
    summary = {"protocol": evaluations[0][640]["settings"], "resolution_probe": resolution,
        "fixed_confidence_640": fixed, "seed1_context_recall50": context_total,
        "seed1_context_size_recall50": context,
        "notes": ["Resolution probe changes inference size only, with both checkpoints trained at 640.",
                  "Official AP is confidence-integrated. Fixed-confidence recall and image alarms are separate diagnostics.",
                  "Long side bins use original image geometry projected to a 640-long-side image; they are not COCO area bins.",
                  "Context differences within broad size bins are associations, not a causal attribution to co-occurrence."]}
    (args.output / "baseline_diagnostic_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ious = evaluations[0][640]["IoU_thresholds"]
    colors = {"fire": "#dc6b28", "smoke": "#46658a"}
    for name in colors:
        axes[0, 0].plot(ious, [100*x for x in resolution[0]["classes"][name]["AP_by_IoU"]], marker="o", color=colors[name], label=name)
    axes[0, 0].set(title="Official AP by IoU: mean of two seeds", xlabel="Matching IoU", ylabel="AP (%)", ylim=(0, 100))
    axes[0, 0].legend()
    for name in colors:
        axes[0, 1].plot(sizes, [100*r["classes"][name]["AP50_95"] for r in resolution], marker="o", color=colors[name], label=name)
    axes[0, 1].plot(sizes, [100*r["mAP50_95"] for r in resolution], marker="s", color="#252525", label="mAP")
    axes[0, 1].set(title="Inference size only: checkpoints trained at 640", xlabel="Validation image size", ylabel="AP50-95 (%)", xticks=sizes)
    axes[0, 1].legend()
    labels = ["<16", "16-32", "32-64", "64-128", ">=128"]
    for key, label in (("recall50", "IoU 0.50"), ("recall75", "IoU 0.75")):
        axes[1, 0].plot(labels, [100*mean(s["classes"]["fire"]["size_bins"][i][key] for s in fixed) for i in range(5)], marker="o", label=label)
    axes[1, 0].set(title="Fire recall by original size: mean of two seeds", xlabel="Box long side projected to 640 (pixels)", ylabel="Recall (%)", ylim=(0, 100))
    axes[1, 0].legend()
    for group, label in (("smoke_only/smoke", "smoke-only images"), ("both/smoke", "fire + smoke images")):
        axes[1, 1].plot(labels, [100*r["recall50"] for r in context[group]], marker="o", label=label)
    axes[1, 1].set(title="Smoke recall by size and context: seed1", xlabel="Box long side projected to 640 (pixels)", ylabel="Recall at IoU 0.50 (%)", ylim=(0, 100))
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
    fig.suptitle("FASDD clean v1 baseline diagnostics | lower panels: confidence >= 0.25", fontsize=15)
    fig.savefig(args.output / "fasdd_baseline_diagnostics.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"resolution_probe": resolution, "context_recall": context_total,
                      "negative_alarm_percent": [100*r["negative_image_alarm_fraction"] for r in fixed]}, indent=2))


if __name__ == "__main__":
    main()
