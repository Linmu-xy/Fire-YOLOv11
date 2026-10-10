#!/usr/bin/env python3
"""Summarize fixed-conf errors and draw six explicitly selected review examples."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {Path(x["prepared_image"]).name: x for x in
                map(json.loads, args.manifest.read_text().splitlines()) if x["split"] == "val"}
    groups = collections.defaultdict(collections.Counter)
    size_groups = collections.defaultdict(lambda: [[0, 0] for _ in range(5)])
    negatives, missed = [], []
    for line in args.audit.open():
        x = json.loads(line)
        cls = x["gt_classes"]
        group = "background" if not cls else "both" if set(cls) == {0, 1} else "fire_only" if set(cls) == {0} else "smoke_only"
        g = groups[group]
        g["images"] += 1
        for c, name in enumerate(("fire", "smoke")):
            g[name + "_gt"] += cls.count(c)
            g[name + "_fp"] += sum(x["pred_classes"][i] == c for i in x["false_prediction_indices"])
            g[name + "_fn"] += sum(cls[i] == c for i in x["missed_gt"])
        if group == "background":
            g["alarm_images"] += bool(x["pred_classes"])
        h, w = x["original_shape_hw"]
        for i, box in enumerate(manifest[Path(x["image"]).name]["boxes"]):
            long = max(box[3]*w, box[4]*h)*640/max(h, w)
            b = sum(long >= t for t in (16, 32, 64, 128))
            a = size_groups[group + "/" + ("fire", "smoke")[int(box[0])]][b]
            a[0] += 1
            a[1] += i not in x["missed_gt"]
        if not cls and x["pred_conf"]:
            negatives.append(x)
        fire_ids = [i for i in x["missed_gt"] if cls[i] == 0]
        if 1 <= len(cls) <= 8 and fire_ids and x["pred_conf"] and any(
            x["missed_candidates"][str(i)]["category"] == "weak_localization_candidate" for i in fire_ids
        ):
            missed.append(x)
    negatives.sort(key=lambda x: max(x["pred_conf"]), reverse=True)
    missed.sort(key=lambda x: hashlib.sha256(x["image"].encode()).hexdigest())
    selected = [{"kind": "background alarm", **x} for x in negatives[:3]] + [{"kind": "missed fire / weak overlap", **x} for x in missed[:3]]
    result = {"groups": {k: dict(v) for k, v in groups.items()}, "size_groups_gt_tp50": dict(size_groups),
        "selected": selected, "confidence": 0.25, "IoU": 0.5,
        "note": "Three hardest negative-image alarms and three deterministic missed-fire weak-overlap candidates. Images are review examples, not prevalence estimates or proof of label errors."}
    (args.output / "error_review.json").write_text(json.dumps(result, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, row in zip(axes.ravel(), selected):
        ax.imshow(Image.open(row["image"]).convert("RGB"))
        for c, xy in zip(row["gt_classes"], row["gt_xyxy"]):
            x1, y1, x2, y2 = xy
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#20d879", linewidth=1.7))
            ax.text(x1, y1, "GT " + ("fire", "smoke")[c], color="black", fontsize=7, backgroundcolor="#20d879")
        for c, xy, conf in zip(row["pred_classes"], row["pred_xyxy"], row["pred_conf"]):
            x1, y1, x2, y2 = xy
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#ff931e", linewidth=1.2, linestyle="--"))
            ax.text(x1, y2, ("fire", "smoke")[c] + f" {conf:.2f}", color="black", fontsize=7, backgroundcolor="#ff931e")
        ax.set_title(row["kind"] + "\n" + Path(row["image"]).name, fontsize=10)
        ax.axis("off")
    fig.suptitle("FASDD validation seed1: GT green, predictions orange (confidence >= 0.25)", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output / "error_review_examples.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result["groups"]))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
