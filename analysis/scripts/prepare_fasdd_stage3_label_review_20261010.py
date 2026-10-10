#!/usr/bin/env python3
"""Create a train-only review packet; no labels or prepared data are edited."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image
    from firesmoke.data import load_prepared, load_protocol
    cfg = load_protocol(ROOT / "configs/protocol-fasdd-n-stage3.yaml")
    _, metadata, rows = load_prepared(cfg, "fasdd_cv_clean_v1")
    groups = collections.defaultdict(list)
    for row in rows:
        if row["split"] != "train":
            continue
        classes = {int(b[0]) for b in row["boxes"]}
        if not classes:
            groups["negative"].append(row)
        if classes == {0, 1}:
            groups["mixed"].append(row)
        if 0 in classes:
            with Image.open(row["prepared_image"]) as image:
                w, h = image.size
            if any(int(b[0]) == 0 and 16 <= max(b[3] * w, b[4] * h) * 640 / max(w, h) < 64
                   for b in row["boxes"]):
                groups["fire_16to64"].append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    selected, used = [], set()
    for group in ("mixed", "fire_16to64", "negative"):
        candidates = sorted(groups[group], key=lambda r: hashlib.sha256(
            ("fasdd-stage3-review-seed0:" + str(r["id"])).encode()).hexdigest())
        fresh = [r for r in candidates if r["id"] not in used][:100]
        if len(fresh) != 100:
            raise RuntimeError("Insufficient unique train images for " + group)
        for row in fresh:
            used.add(row["id"])
            selected.append({"image_id": row["id"], "group": group,
                "image": row["prepared_image"], "original_boxes": row["boxes"],
                "review_status": "pending", "reviewer": "", "policy_decision": "",
                "proposed_boxes": None, "notes": ""})
    with (args.output / "review_manifest.jsonl").open("x") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["image_id", "group", "image", "review_status", "reviewer", "policy_decision", "notes"]
    with (args.output / "review_sheet.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row[k] for k in fields})
    summary = {"portable_content_sha256": metadata["portable_content_sha256"],
               "split": "train", "unique_images": len(used),
               "groups": dict(collections.Counter(row["group"] for row in selected)),
               "candidate_pool_sizes": {k: len(v) for k, v in groups.items()},
               "review_status": "pending", "labels_changed": False,
               "selection": "Hash order, unique IDs; mixed then small-fire then negative. Group frequencies are not dataset frequencies."}
    (args.output / "registration.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
