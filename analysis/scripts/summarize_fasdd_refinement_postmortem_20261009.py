#!/usr/bin/env python3
"""Summarize frozen remote CSV copies and plot restart-contaminated curves."""
from __future__ import annotations
import csv
import argparse
import io
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/artifacts/fasdd_refinement_postmortem_20261009"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    sources = {host: json.loads((OUT / host / "collected.json").read_text(encoding="utf-8-sig"))
               for host in ("cmj", "lxy")}
    specs = {"baseline_seed0": ("lxy", "fasdd-cv-clean-v1/baseline/seed0", None),
             "baseline_seed1": ("cmj", "fasdd-cv-clean-v1/baseline/seed1", None),
             "detail_seed0": ("lxy", "fasdd-n-refinement-v1/detail/seed0", 25),
             "context_seed0": ("cmj", "fasdd-n-refinement-v1/context/seed0", 30)}
    curves, summary = {}, {}
    for name, (host, suffix, resumed_at) in specs.items():
        key = f"results/fasdd_cv_clean_v1/{suffix}/results.csv"
        rows = [{k.strip(): float(v.strip()) for k, v in row.items()}
                for row in csv.DictReader(io.StringIO(sources[host][key]))]
        curve = {int(row["epoch"]): row for row in rows}
        curves[name] = curve
        summary[name] = {"host": host, "source": key, "epochs": len(rows),
                         "final": rows[-1],
                         "best": max(rows, key=lambda row: row["metrics/mAP50-95(B)"]),
                         "first_resumed_epoch": resumed_at}
    baseline = curves["baseline_seed0"]
    for name in ("detail_seed0", "context_seed0"):
        curve, resume = curves[name], specs[name][2]
        delta = {e: 100 * (row["metrics/mAP50-95(B)"] - baseline[e]["metrics/mAP50-95(B)"])
                 for e, row in curve.items()}
        recent = list(range(resume - 10, resume))
        minimum = min(range(resume, 101), key=lambda e: curve[e]["metrics/mAP50-95(B)"])
        summary[name].update(final_delta_pp=delta[100],
                             pre_resume_last10_mean_delta_pp=statistics.mean(delta[e] for e in recent),
                             pre_resume_last10_delta_range_pp=[min(delta[e] for e in recent), max(delta[e] for e in recent)],
                             pre_resume_last_epoch_delta_pp=delta[resume - 1],
                             lowest_post_resume_epoch=minimum,
                             lowest_post_resume_map5095=curve[minimum]["metrics/mAP50-95(B)"],
                             lowest_post_resume_delta_pp=delta[minimum],
                             delta_by_epoch_pp=delta)
    summary["observed_baseline_seed_spread_pp"] = 100 * (
        summary["baseline_seed1"]["final"]["metrics/mAP50-95(B)"] - summary["baseline_seed0"]["final"]["metrics/mAP50-95(B)"])
    (OUT / "curve_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: {field: value for field, value in item.items() if field != "delta_by_epoch_pp"}
                      if isinstance(item, dict) else item for key, item in summary.items()}, indent=2))
    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    styles = {"baseline_seed0": ("#374151", "Baseline seed0"),
              "baseline_seed1": ("#a1a1aa", "Baseline seed1 (reference)"),
              "detail_seed0": ("#2563eb", "Detail seed0"),
              "context_seed0": ("#dc2626", "Context seed0")}
    for name, curve in curves.items():
        color, label = styles[name]
        epochs = sorted(curve)
        axes[0].plot(epochs, [100 * curve[e]["metrics/mAP50-95(B)"] for e in epochs],
                     color=color, label=label, lw=2 if name != "baseline_seed1" else 1,
                     linestyle="--" if name == "baseline_seed1" else "-")
        if name in ("detail_seed0", "context_seed0"):
            axes[1].plot(epochs, [summary[name]["delta_by_epoch_pp"][e] for e in epochs], color=color, label=label, lw=2)
    for ax in axes:
        ax.axvline(25, color="#2563eb", ls=":", lw=1.3)
        ax.axvline(30, color="#dc2626", ls=":", lw=1.3)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlim(15, 101)
    axes[0].set_ylim(39, 54)
    axes[0].set_ylabel("Validation mAP50-95 (%)")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].set_title("FASDD clean-v1 / YOLO11n / 640 / 100 epochs\nHead and refinement were reset at resume; backbone/neck were restored", loc="left", fontsize=13)
    axes[0].text(25.8, 53.7, "Detail resume: epoch25", color="#2563eb", fontsize=9, va="top")
    axes[0].text(30.8, 52.95, "Context resume: epoch30", color="#dc2626", fontsize=9, va="top")
    axes[1].axhline(0, color="#374151", lw=1)
    axes[1].set_ylim(-10, 1)
    axes[1].set_ylabel("Difference vs baseline seed0 (pp)")
    axes[1].set_xlabel("Completed epoch")
    fig.tight_layout()
    fig.savefig(OUT / "restart_curves.png", dpi=180)
    fig.savefig(OUT / "restart_curves.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
