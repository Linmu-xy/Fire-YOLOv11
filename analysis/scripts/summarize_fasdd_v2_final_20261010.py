#!/usr/bin/env python3
"""Reproduce V2 final comparisons and ranking diagnostic figure from saved inputs."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "analysis/artifacts/fasdd_v2_final_20261010"
os.environ["MPLCONFIGDIR"] = str(ART / "matplotlib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Produce the JSON summary without matplotlib")
    args = parser.parse_args()
    summaries = {}
    audits = {}
    curves = {}
    for name in ("baseline", "context", "detail_norm"):
        audit_dir = ART / (name + "_rank" if name in ("baseline", "context") else name)
        audit = json.loads((audit_dir / "audit.json").read_text(encoding="utf-8-sig"))
        audits[name] = audit
        rows = list(csv.DictReader((ART / name / "results.csv").open()))
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in rows]
        curves[name] = rows
        assert len(rows) == 100 and [int(r["epoch"]) for r in rows] == list(range(1, 101))
        diag = audit["fixed_conf_diagnostics"]["0.25"]
        fire_sizes = diag["sizes"][0]
        counts = {"fire_16to64_recall75": sum(fire_sizes[b][2] for b in (1, 2)) /
                  sum(fire_sizes[b][0] for b in (1, 2)),
                  "mixed_smoke_recall50": diag["scenes"]["mixed"][1][1] / diag["scenes"]["mixed"][1][0],
                  "negative_image_alarm_rate": diag["negative_alarm_images"] / diag["negative_images"]}
        summaries[name] = {"csv_final": rows[-1],
            "best_epoch": int(max(rows, key=lambda r: r["metrics/mAP50-95(B)"])["epoch"]),
            "last10_mean_map": sum(r["metrics/mAP50-95(B)"] for r in rows[-10:]) / 10,
            "full_val_map": audit["metrics"]["metrics/mAP50-95(B)"],
            "per_class": audit["per_class"], "fixed_conf25": counts,
            "conf001_recall": {cls: {f"recall{threshold}": diag_row[col] / diag_row[0]
                                     for col, threshold in enumerate((50, 75, 90), 1)}
                               for cls, diag_row in zip(("fire", "smoke"), audit["fixed_conf_diagnostics"]["0.001"]["classes"])}}
        if "oracle_quality_order_within_tp50" in audit:
            summaries[name]["quality_order_oracle"] = audit["oracle_quality_order_within_tp50"]
        if audit["residual_sample"]:
            summaries[name]["mean_residual"] = {key: [sum(row[key][i] for row in audit["residual_sample"]) /
                                                      len(audit["residual_sample"]) for i in range(3)]
                                                   for key in ("reg_ratio", "cls_ratio")}
    for name in ("context", "detail_norm"):
        base, candidate = summaries["baseline"], summaries[name]
        candidate["delta_pp"] = {"csv_map": 100 * (candidate["csv_final"]["metrics/mAP50-95(B)"] - base["csv_final"]["metrics/mAP50-95(B)"]),
            "full_val_map": 100 * (candidate["full_val_map"] - base["full_val_map"]),
            "fire_AP75": 100 * (candidate["per_class"]["fire"]["AP75"] - base["per_class"]["fire"]["AP75"]),
            "fire_AP50_95": 100 * (candidate["per_class"]["fire"]["AP50_95"] - base["per_class"]["fire"]["AP50_95"]),
            "smoke_AP50_95": 100 * (candidate["per_class"]["smoke"]["AP50_95"] - base["per_class"]["smoke"]["AP50_95"]),
            "last10_paired_map": 100 * (candidate["last10_mean_map"] - base["last10_mean_map"]),
            **{k: 100 * (v - base["fixed_conf25"][k]) for k, v in candidate["fixed_conf25"].items()}}
    (ART / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    inputs = sorted({*(ART.glob("*/audit.json")), *(ART.glob("*/results.csv"))})
    (ART / "input_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                      for p in inputs}, indent=2) + "\n")
    if args.no_plot:
        print(json.dumps({name: {"map": row["csv_final"]["metrics/mAP50-95(B)"],
                                "fixed_conf25": row["fixed_conf25"], "delta_pp": row.get("delta_pp")}
                          for name, row in summaries.items()}, indent=2))
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {"baseline": "#444444", "context": "#0072B2", "detail_norm": "#D55E00"}
    for name in curves:
        axes[0].plot(range(1, 101), [100 * r["metrics/mAP50-95(B)"] for r in curves[name]],
                     color=colors[name], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Val mAP50-95 (%)", ylim=(40, 55), title="V2 changes little at convergence")
    axes[0].legend(fontsize=9)
    for name in ("context", "detail_norm"):
        axes[1].plot(range(1, 101), [100 * (c["metrics/mAP50-95(B)"] - b["metrics/mAP50-95(B)"])
                     for c, b in zip(curves[name], curves["baseline"])], color=colors[name], label=name)
    axes[1].axhline(0, color="#999999", linewidth=1)
    axes[1].set(xlabel="Epoch", ylabel="Paired gain (percentage points)", ylim=(-.5, 1),
                title="Same seed / same epoch gain")
    axes[1].legend(fontsize=9)
    labels = ["Native", "GT quality\nordering oracle"]
    values = [100 * summaries["baseline"]["full_val_map"],
              100 * summaries["baseline"]["quality_order_oracle"]["mAP50_95"]]
    axes[2].bar(labels, values, color=["#444444", "#009E73"], width=.55)
    axes[2].set(ylabel="Val mAP50-95 (%)", ylim=(0, 70), title="Boxes and FP50 scores are fixed")
    for i, value in enumerate(values):
        axes[2].text(i, value + 1, f"{value:.2f}%", ha="center")
    axes[2].text(.5, .03, "GT REQUIRED: diagnostic, not model gain", transform=axes[2].transAxes,
                 ha="center", fontsize=8, color="#B22222")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(ART / "v2_final_and_ranking.png", dpi=160)
    plt.close(fig)
    print(json.dumps({name: {"map": row["csv_final"]["metrics/mAP50-95(B)"],
                            "fixed_conf25": row["fixed_conf25"], "delta_pp": row.get("delta_pp")}
                      for name, row in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
