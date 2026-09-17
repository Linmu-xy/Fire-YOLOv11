"""Read-only source inventory, canonical labels, and duplicate candidate audit."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .common import NAMES, digest, fresh_dir, path, read_json, stable_digest, write_json

SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_protocol(file):
    cfg = yaml.safe_load(path(file).read_text(encoding="utf-8"))
    if cfg.get("protocol") != "firesmoke-v1":
        raise ValueError("Unknown protocol version")
    return cfg


def read_labels(file, source_names=NAMES, issues=None):
    """Empty labels are valid negatives; missing/malformed labels are errors."""
    boxes = []
    for number, line in enumerate(Path(file).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{file}:{number}: expected class x y w h")
        c, x, y, w, h = map(float, fields)
        if not all(math.isfinite(v) for v in (c, x, y, w, h)):
            raise ValueError(f"{file}:{number}: non-finite label")
        if c != int(c) or not 0 <= int(c) < len(source_names):
            raise ValueError(f"{file}:{number}: invalid class")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            if issues is None:
                raise ValueError(f"{file}:{number}: invalid normalized box")
            issues.append({"line": number, "reason": "invalid_normalized_box", "label": line.strip()})
            continue
        if w == 0 or h == 0:
            if issues is None:
                raise ValueError(f"{file}:{number}: zero-area box")
            issues.append({"line": number, "reason": "zero_area_box", "label": line.strip()})
            continue
        boxes.append([NAMES.index(source_names[int(c)]), x, y, w, h])
    return boxes


def inventory(cfg, dataset):
    spec = cfg["datasets"][dataset]
    root = path(spec["root"])
    records, seen = [], set()
    for split in SPLITS:
        if spec["layout"] == "dfire":
            images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in EXTENSIONS)
            label_dir = root / "labels" / split
            expected = {p.stem for p in images}
            actual = {p.stem for p in label_dir.glob("*.txt")}
            if expected != actual:
                raise ValueError(f"{dataset}/{split}: image/label names do not match")
        elif spec["layout"] == "fasdd":
            ann = root / "annotations" / "YOLO_CV"
            entries = [s.strip() for s in (ann / f"{split}.txt").read_text().splitlines() if s.strip()]
            if any(not s.startswith("./images/") or ".." in Path(s).parts for s in entries):
                raise ValueError(f"Unexpected path in {ann}/{split}.txt")
            images = sorted(root / "images" / Path(s).name for s in entries)
            label_dir = ann / "labels"
        else:
            raise ValueError("Supported source layouts: dfire, fasdd")
        if not images:
            raise ValueError(f"Empty split: {dataset}/{split}")
        for image in images:
            key = str(image.resolve())
            if key in seen:
                raise ValueError(f"Repeated image path within/across splits: {image}")
            seen.add(key)
            if not image.is_file():
                raise FileNotFoundError(image)
            label = label_dir / f"{image.stem}.txt"
            issues = []
            policy = spec.get("invalid_box_policy", "error")
            if policy not in {"error", "exclude_invalid_images"}:
                raise ValueError(f"Unknown invalid_box_policy: {policy}")
            boxes = read_labels(label, spec["source_names"], issues if policy == "exclude_invalid_images" else None)
            records.append({
                "id": f"{dataset}/{split}/{image.name}", "dataset": dataset, "split": split,
                "image": str(image.absolute()), "label": str(label.absolute()), "boxes": boxes,
                # Scene identity cannot safely be inferred from an image filename.
                "group": None,
                "excluded": bool(issues), "label_issues": issues,
            })
    return records


def prepare(cfg, dataset):
    all_records = inventory(cfg, dataset)  # validate all labels before making any outputs
    excluded = [r for r in all_records if r["excluded"]]
    records = [r for r in all_records if not r["excluded"]]
    if any(not any(r["split"] == s for r in records) for s in SPLITS):
        raise ValueError("Label policy would leave an empty split")
    out = fresh_dir(path(cfg["prepared_root"]) / dataset)
    write_json(out / "excluded.json", excluded)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    for r in records:
        image = out / "images" / r["split"] / Path(r["image"]).name
        image.symlink_to(r["image"])
        label = out / "labels" / r["split"] / Path(r["label"]).name
        label.write_text("".join(" ".join(map(str, b)) + "\n" for b in r["boxes"]), encoding="utf-8")
        r["prepared_image"] = str(image)  # do not resolve: preserve canonical label lookup
        r["prepared_label"] = str(label)
    for split in SPLITS:
        (out / f"{split}.txt").write_text("".join(r["prepared_image"]+"\n" for r in records if r["split"] == split))
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    data = {"path": str(out), **{s: f"{s}.txt" for s in SPLITS}, "names": dict(enumerate(NAMES))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    write_json(out / "metadata.json", {
        "dataset": dataset, "names": NAMES, "source": cfg["datasets"][dataset],
        "manifest_sha256": digest(manifest),
        "portable_inventory_sha256": stable_digest([{k: r[k] for k in ("id", "boxes")} for r in records]),
        "counts": dict(Counter(r["split"] for r in records)),
        "excluded_counts": dict(Counter(r["split"] for r in excluded)),
        "excluded_sha256": digest(out / "excluded.json"),
        "label_policy": cfg["datasets"][dataset].get("invalid_box_policy", "error"),
        "image_content_hashed": False,
    })
    return out


def load_prepared(cfg, dataset, verify=True):
    out = path(cfg["prepared_root"]) / dataset
    metadata = read_json(out / "metadata.json")
    if verify and digest(out / "manifest.jsonl") != metadata["manifest_sha256"]:
        raise ValueError("Prepared manifest was modified; create a new prepared version")
    rows = [json.loads(s) for s in (out / "manifest.jsonl").read_text().splitlines()]
    if verify:
        for r in rows:
            if not Path(r["prepared_image"]).is_file():
                raise FileNotFoundError(r["prepared_image"])
            if read_labels(r["prepared_label"]) != r["boxes"]:
                raise ValueError(f"Prepared label changed: {r['id']}")
        for split in SPLITS:
            expected = [r["prepared_image"] for r in rows if r["split"] == split]
            actual = (out / f"{split}.txt").read_text().splitlines()
            if actual != expected:
                raise ValueError(f"Prepared split changed: {out}/{split}.txt")
        data = yaml.safe_load((out / "data.yaml").read_text())
        expected = {"path": str(out), **{s: f"{s}.txt" for s in SPLITS}, "names": dict(enumerate(NAMES))}
        if data != expected:
            raise ValueError("Prepared data.yaml changed")
    return out, metadata, rows


class BKTree:
    """Hamming-distance candidate index; audit only, never automatic deletion."""
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        todo = [self.root] if self.root else []
        while todo:
            node = todo.pop()
            d = (value ^ node[0]).bit_count()
            if d <= radius:
                for i in node[1]:
                    yield i, d
            todo.extend(child for edge, child in node[2].items() if d-radius <= edge <= d+radius)


def image_hashes(file):
    from PIL import Image, ImageOps
    with Image.open(file) as image:
        # Verify decode; record displayed orientation for the duplicate check.
        gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8))
        pixels = list(gray.tobytes())
        dhash = 0
        for y in range(8):
            for x in range(8):
                dhash = (dhash << 1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return digest(file), dhash


def audit(cfg, datasets, output, hashes=False, radius=4):
    if not 0 <= radius <= 8:
        raise ValueError("dHash radius must be 0..8")
    rows = [r for ds in datasets for r in inventory(cfg, ds)]
    out = fresh_dir(path(output))
    counts = defaultdict(Counter)
    for r in rows:
        c = counts[f"{r['dataset']}/{r['split']}"]
        c["images"] += 1
        c["background"] += not r["boxes"] and not r["excluded"]
        c["excluded_images"] += r["excluded"]
        for issue in r["label_issues"]:
            c[issue["reason"]] += 1
        for b in r["boxes"]:
            c[NAMES[b[0]]+"_boxes"] += 1
            _, x, y, w, h = b
            c["boundary_crossing_boxes"] += min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5
    findings = 0
    write_json(out / "label_issues.json", [{"id": r["id"], "issues": r["label_issues"]} for r in rows if r["excluded"]])
    if hashes:
        tree, exact = BKTree(), defaultdict(list)
        with (out / "hashes.jsonl").open("x") as hs, (out / "duplicate_candidates.jsonl").open("x") as pairs:
            for i, r in enumerate(rows):
                sha, dh = image_hashes(r["image"])
                hs.write(json.dumps({"id": r["id"], "sha256": sha, "dhash": f"{dh:016x}"})+"\n")
                # Count all pairs, stream to disk instead of holding pairwise matrices.
                candidates = dict(tree.query(dh, radius))
                candidates.update({j: 0 for j in exact[sha]})
                for j, distance in candidates.items():
                    other = rows[j]
                    if (r["dataset"], r["split"]) == (other["dataset"], other["split"]):
                        continue
                    findings += 1
                    pairs.write(json.dumps({"a": other["id"], "b": r["id"], "distance": distance,
                                            "byte_identical": j in exact[sha]})+"\n")
                exact[sha].append(i)
                tree.add(dh, i)
                if (i+1) % 1000 == 0:
                    print(f"Hashed {i+1}/{len(rows)} images", flush=True)
    report = {"counts": counts, "hash_audit_completed": hashes, "candidate_pairs": findings if hashes else None,
              "dhash_radius": radius if hashes else None, "scene_groups_available": False,
              "warning": "Candidates need manual verification. Metadata-only audit does not prove absence of leakage.",
              "boundary_policy": "Preserve source boxes with valid normalized xywh even if edges exceed image; report counts, never silently clip labels."}
    write_json(out / "audit.json", report)
    return report
