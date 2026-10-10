#!/usr/bin/env python3
"""Analyze archived baseline curves without changing training results."""
import argparse
import csv
import json
import math
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
    runs = []
    curves = {}
    for seed in (0, 1):
        root = args.inputs / f"seed{seed}"
        with (root / "results.csv").open(newline="") as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        config = json.loads((root / "run.json").read_text(encoding="utf-8"))
        by_epoch = {int(r["epoch"]): r for r in rows}
        metric = "metrics/mAP50-95(B)"
        best = max(rows, key=lambda r: r[metric])
        last = rows[-1]
        exposures = [json.loads(x) for x in (root / "background_exposure.jsonl").read_text().splitlines() if x]
        def exposure(a, b):
            selected = [r for r in exposures if a <= r["epoch"] <= b]
            total = sum(r["images"] for r in selected)
            empty = sum(r["post_augmentation_empty_images"] for r in selected)
            return {"images": total, "empty_images": empty, "fraction": empty / total}
        windows = {}
        for a, b in [(20, 30), (45, 55), (70, 80), (81, 90), (91, 100)]:
            selected = [r for r in rows if a <= r["epoch"] <= b]
            windows[f"{a}_{b}"] = {k: statistics.mean(r[k] for r in selected) for k in
                [metric, "metrics/mAP50(B)", "val/box_loss", "val/cls_loss", "train/box_loss", "train/cls_loss"]}
        changes = {f"{a}_to_{b}_pp": (by_epoch[b][metric] - by_epoch[a][metric]) * 100
                   for a, b in [(60, 80), (80, 90), (90, 100), (95, 100)]}
        summary = {"seed": seed, "epochs": len(rows), "missing_epochs": sorted(set(range(1, 101)) - set(by_epoch)),
            "nonfinite": sum(not math.isfinite(v) for r in rows for v in r.values()),
            "best": best, "last": last, "windows": windows, "changes": changes,
            "training_hours": last["time"] / 3600, "training": config["training"],
            "pretrained_sha256": config["pretrained_sha256"],
            "content_sha256": config["data_metadata"]["portable_content_sha256"],
            "core_source_sha256": {k: v for k, v in config["provenance"]["source_sha256"].items() if k.startswith("firesmoke/")},
            "post_augmentation_empty": {"epochs_1_90": exposure(1, 90), "epochs_91_100": exposure(91, 100)}}
        runs.append(summary)
        curves[seed] = rows
    recipe0 = {k: v for k, v in runs[0]["training"].items() if k != "seed"}
    recipe1 = {k: v for k, v in runs[1]["training"].items() if k != "seed"}
    values = [r["best"]["metrics/mAP50-95(B)"] for r in runs]
    report = {"runs": runs, "same_recipe_except_seed": recipe0 == recipe1,
        "same_core_source": runs[0]["core_source_sha256"] == runs[1]["core_source_sha256"],
        "same_content": runs[0]["content_sha256"] == runs[1]["content_sha256"],
        "same_pretrained": runs[0]["pretrained_sha256"] == runs[1]["pretrained_sha256"],
        "best_mean": statistics.mean(values), "best_sample_sd": statistics.stdev(values),
        "plus5_target_mean": statistics.mean(values) + 0.05,
        "note": "Only two seeds. AP is the best validation value, not test generalization."}
    (args.output / "baseline_curve_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for seed, rows in curves.items():
        ep = [r["epoch"] for r in rows]
        label = f"seed{seed}"
        color = ("#2457a7", "#d87320")[seed]
        axes[0, 0].plot(ep, [100*r["metrics/mAP50-95(B)"] for r in rows], color=color, label=label)
        axes[0, 1].plot(ep, [100*r["metrics/mAP50-95(B)"] for r in rows], color=color, label=label)
        axes[1, 0].plot(ep, [r["val/box_loss"] for r in rows], color=color, label=label)
        axes[1, 1].plot(ep, [r["val/cls_loss"] for r in rows], color=color, label=label)
    axes[0, 0].set(title="FASDD clean v1: validation mAP50-95", ylabel="mAP (%)", xlabel="Epoch")
    axes[0, 1].set(title="Last 30 epochs", xlim=(70, 100), ylim=(51.5, 54), ylabel="mAP (%)", xlabel="Epoch")
    axes[1, 0].set(title="Validation box loss", xlabel="Epoch", ylabel="Loss")
    axes[1, 1].set(title="Validation classification loss", xlabel="Epoch", ylabel="Loss")
    for ax in axes.ravel():
        ax.axvline(90, color="#777777", ls="--", alpha=.6)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Best validation score occurs at epoch 100 for both seeds", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output / "fasdd_baseline_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, ensure_ascii=False))
    for r in runs:
        print("SEED", r["seed"], "BEST", r["best"]["metrics/mAP50-95(B)"], "CHANGES", r["changes"], "EMPTY", r["post_augmentation_empty"])


if __name__ == "__main__":
    main()
