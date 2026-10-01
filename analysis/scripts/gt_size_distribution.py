#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 YOLO 格式数据集的 GT 框尺寸分布，判定 P2 检测层在结构上是否可能有收益。

这是决定「P2 该不该继续投入算力」的唯一前置测量。没有它，任何「P2 应该对小目标有用」
的论断都只是假设。

在 Fire-YOLOv11 仓库根目录运行：

    python analysis/gt_size_distribution.py --data datasets/D-Fire/data.yaml
    python analysis/gt_size_distribution.py --data datasets/D-Fire/data.yaml --imgsz 640 --out analysis/out
    python analysis/gt_size_distribution.py --data datasets/FASDD_CV/data.yaml --split train val test

判读口径
--------
1. COCO 面积分层按 **原图像素** 计算（这是文献可比的唯一口径）：
     small  : area < 32^2  (1024 px^2)
     medium : 32^2 <= area < 96^2
     large  : area >= 96^2
   同时另报 letterbox 缩放到 imgsz 之后的像素尺寸，两者**不得混称**。

2. 可分辨性：目标在某个 stride 的特征图上占多少个格子。设缩放后长边为 s，
   stride k 的格子数 n = s / k。
     n < 1     → 该层完全看不到这个目标（漏检由结构决定）
     1 <= n < 2 → 勉强 1 个格子，定位误差极大
     n >= 2    → 该层有基本定位能力
   关键结论是 P3(stride 8) 已经够用的目标占比。若绝大多数目标在 stride 8 上 n >= 2，
   那 P2 只是给同一批目标多加一层冗余分支，涨点靠的是额外容量而非小目标能力。

3. 分类别统计（fire / smoke）必须分开报。火与烟的尺度分布完全不同，
   总体 mAP 上升可能只来自其中一类。

输出
----
  size_buckets.csv        split × class × COCO 面积段 的数量与占比
  resolvability.csv       split × class × stride 的可分辨性占比
  size_summary.txt        人读摘要 + 结构性结论
  size_hist.png           长边像素直方图（需要 matplotlib）
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_KEYS = ("train", "val", "test")


def parse_data_yaml(path: Path) -> dict:
    """读 ultralytics data.yaml，取 path / train / val / test / names。"""
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass

    out: dict = {"names": {}}
    in_names = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.rstrip()
        if stripped.startswith("names:"):
            in_names = True
            rest = stripped.partition(":")[2].strip()
            if rest:
                for pair in rest.strip("{}").split(","):
                    if ":" not in pair:
                        continue
                    k, _, v = pair.partition(":")
                    try:
                        out["names"][int(k.strip())] = v.strip().strip("'\"")
                    except ValueError:
                        pass
                in_names = False
            continue
        if in_names and stripped.startswith((" ", "\t")) and ":" in stripped:
            k, _, v = stripped.partition(":")
            try:
                out["names"][int(k.strip())] = v.strip().strip("'\"")
            except ValueError:
                pass
            continue
        in_names = False
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            out[key.strip()] = value.strip().strip("'\"")
    return out


def resolve_split_dirs(root: Path, entry) -> list[Path]:
    """data.yaml 的 train/val/test 可能是 str、list，或一个 .txt 清单文件。"""
    if entry is None:
        return []
    candidates: list[Path] = []
    values = entry if isinstance(entry, (list, tuple)) else [entry]
    for value in values:
        if isinstance(value, list):
            candidates.extend(Path(v) for v in value)
            continue
        p = Path(str(value))
        if not p.is_absolute():
            p = root / p
        if p.is_file() and p.suffix == ".txt":
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                q = Path(line)
                candidates.append(q if q.is_absolute() else p.parent / q)
        else:
            candidates.append(p)
    return candidates


def image_to_label_file(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def iter_images(directory: Path):
    """directory 可能是目录, 也可能是清单展开后的单个图像文件路径。"""
    if directory.is_file():
        if directory.suffix.lower() in EXTS:
            yield directory
        return
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in EXTS:
            yield path


def image_size(path: Path) -> tuple[int, int] | None:
    """优先用 PIL 读真实像素；失败则退回 PNG/JPEG 头直接解析。"""
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return im.size
    except Exception:
        pass
    try:
        if path.suffix.lower() == ".png":
            with path.open("rb") as fh:
                head = fh.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return (int.from_bytes(head[16:20], "big"),
                        int.from_bytes(head[20:24], "big"))
    except OSError:
        pass
    return None


def coco_bucket(area_px: float) -> str:
    if area_px < 32 * 32:
        return "small"
    if area_px < 96 * 96:
        return "medium"
    return "large"


def main() -> int:
    parser = argparse.ArgumentParser(description="GT 框尺寸分布与 P2 必要性判定")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml 路径")
    parser.add_argument("--imgsz", type=int, default=640, help="训练输入边长，默认 640")
    parser.add_argument("--split", nargs="*", default=None, help="只统计指定 split")
    parser.add_argument("--out", type=Path, default=Path("analysis/out_size"))
    parser.add_argument("--strides", nargs="*", type=int, default=[4, 8, 16, 32])
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"[error] data.yaml 不存在：{args.data}", file=sys.stderr)
        return 2

    cfg = parse_data_yaml(args.data)
    names = {int(k): str(v) for k, v in (cfg.get("names") or {}).items()}
    root_raw = cfg.get("path")
    if root_raw:
        root = Path(str(root_raw))
        if not root.is_absolute():
            root = (args.data.parent / root).resolve()
    else:
        root = args.data.parent.resolve()

    splits = args.split or [s for s in SPLIT_KEYS if s in cfg]
    if not splits:
        print("[error] data.yaml 里没有 train/val/test 字段", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    # counts[split][class][bucket] 与 lengths[split][class] 存长边像素
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    resolv = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    lengths = defaultdict(lambda: defaultdict(list))
    images_seen = defaultdict(int)
    images_empty = defaultdict(int)
    images_no_size = defaultdict(int)

    for split in splits:
        for directory in resolve_split_dirs(root, cfg.get(split)):
            for img in iter_images(directory):
                size = image_size(img)
                if size is None:
                    images_no_size[split] += 1
                    continue
                images_seen[split] += 1
                width, height = size
                label_file = image_to_label_file(img)
                boxes = 0
                if label_file.is_file():
                    for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            cid = int(float(parts[0]))
                            bw = float(parts[3])
                            bh = float(parts[4])
                        except ValueError:
                            continue
                        cname = names.get(cid, f"class{cid}")
                        w_px, h_px = bw * width, bh * height
                        area = w_px * h_px
                        counts[split][cname][coco_bucket(area)] += 1
                        boxes += 1

                        scale = min(args.imgsz / width, args.imgsz / height)
                        long_side = max(w_px, h_px) * scale
                        lengths[split][cname].append(long_side)
                        for k in args.strides:
                            cells = long_side / k
                            tag = "n_lt_1" if cells < 1 else ("n_lt_2" if cells < 2 else "n_ge_2")
                            resolv[split][cname][f"stride{k}:{tag}"] += 1
                if boxes == 0:
                    images_empty[split] += 1

    if not any(counts.values()):
        print("[error] 未统计到任何 GT 框，请检查 data.yaml 中 path 与标签目录结构",
              file=sys.stderr)
        return 1

    # size_buckets.csv
    with (args.out / "size_buckets.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "class", "bucket", "count", "share_of_class", "share_of_split"])
        for split in splits:
            split_total = sum(sum(v.values()) for v in counts[split].values())
            for cname, buckets in sorted(counts[split].items()):
                cls_total = sum(buckets.values())
                for bucket in ("small", "medium", "large"):
                    n = buckets.get(bucket, 0)
                    writer.writerow([
                        split, cname, bucket, n,
                        f"{n / cls_total:.4f}" if cls_total else "",
                        f"{n / split_total:.4f}" if split_total else "",
                    ])

    # resolvability.csv
    with (args.out / "resolvability.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "class", "stride", "n_ge_2_share", "n_lt_2_share", "n_lt_1_share",
                         "n_ge_2_count", "total"])
        for split in splits:
            for cname in sorted(resolv[split]):
                for k in args.strides:
                    total = sum(resolv[split][cname].get(f"stride{k}:{t}", 0)
                                for t in ("n_lt_1", "n_lt_2", "n_ge_2"))
                    if not total:
                        continue
                    ge2 = resolv[split][cname].get(f"stride{k}:n_ge_2", 0)
                    lt2 = resolv[split][cname].get(f"stride{k}:n_lt_2", 0)
                    lt1 = resolv[split][cname].get(f"stride{k}:n_lt_1", 0)
                    writer.writerow([split, cname, k, f"{ge2/total:.4f}", f"{lt2/total:.4f}",
                                     f"{lt1/total:.4f}", ge2, total])

    # size_summary.txt
    lines: list[str] = []
    lines.append(f"data.yaml : {args.data}")
    lines.append(f"imgsz     : {args.imgsz}  (letterbox 缩放基准)")
    lines.append("")
    for split in splits:
        total = sum(sum(v.values()) for v in counts[split].values())
        lines.append(f"== split: {split} ==")
        lines.append(f"  图像数（有尺寸）      : {images_seen[split]}")
        lines.append(f"  空标签图（背景/None） : {images_empty[split]}")
        if images_no_size[split]:
            lines.append(f"  读不到尺寸而跳过      : {images_no_size[split]}")
        lines.append(f"  GT 框总数             : {total}")
        if not total:
            lines.append("")
            continue

        for cname in sorted(counts[split]):
            buckets = counts[split][cname]
            cls_total = sum(buckets.values())
            small = buckets.get("small", 0)
            medium = buckets.get("medium", 0)
            large = buckets.get("large", 0)
            lines.append(
                f"  [{cname}] n={cls_total}  small={small/cls_total:.1%}  "
                f"medium={medium/cls_total:.1%}  large={large/cls_total:.1%}"
            )

        all_buckets = defaultdict(int)
        for cname in counts[split]:
            for k, v in counts[split][cname].items():
                all_buckets[k] += v
        lines.append(
            f"  [合并] small={all_buckets['small']/total:.1%}  "
            f"medium={all_buckets['medium']/total:.1%}  "
            f"large={all_buckets['large']/total:.1%}"
        )

        # 结构性判读
        for k in args.strides:
            ge2 = sum(resolv[split][c].get(f"stride{k}:n_ge_2", 0) for c in resolv[split])
            tot = sum(
                resolv[split][c].get(f"stride{k}:{t}", 0)
                for c in resolv[split] for t in ("n_lt_1", "n_lt_2", "n_ge_2")
            )
            if tot:
                lines.append(f"  stride {k:>2}: 该层可定位(n>=2 格)占比 = {ge2/tot:.1%}")
        lines.append("")

    # 汇总结论
    lines.append("== 结论 ==")
    for split in splits:
        all_long: list[float] = []
        for cname in lengths[split]:
            all_long.extend(lengths[split][cname])
        if not all_long:
            continue
        all_long.sort()
        def q(p: float) -> float:
            idx = min(len(all_long) - 1, max(0, int(round(p * (len(all_long) - 1)))))
            return all_long[idx]
        lines.append(
            f"[{split}] letterbox 后长边(px): "
            f"p10={q(.10):.1f}  p25={q(.25):.1f}  p50={q(.50):.1f}  "
            f"p75={q(.75):.1f}  p90={q(.90):.1f}"
        )
        frac_small_8 = sum(1 for s in all_long if s / 8 < 2) / len(all_long)
        frac_small_4 = sum(1 for s in all_long if s / 4 < 2) / len(all_long)
        lines.append(
            f"  P3(stride 8) 已不充分的目标准占比 = {frac_small_8:.1%}  "
            f"(这些才是 P2 的作用域)"
        )
        lines.append(
            f"  P2(stride 4) 也已不充分的目标准占比 = {frac_small_4:.1%}"
        )
        if frac_small_8 < 0.15:
            lines.append(
                "  → 判读：P3 已覆盖绝大多数目标，P2 缺少结构性必要性；"
                "若 P2 仍涨点，应归因于额外容量，需用同参数量对照证明。"
            )
        else:
            lines.append(
                "  → 判读：存在可观的小目标尾部，P2 有结构必要性；"
                "下一步必须补尺寸分层 AP，证明收益确实集中在小目标段。"
            )
    (args.out / "size_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 直方图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
        for split in splits:
            merged: list[float] = []
            for cname in lengths[split]:
                merged.extend(lengths[split][cname])
            if merged:
                ax.hist(merged, bins=60, range=(0, 320), alpha=0.55, label=f"{split} (n={len(merged)})")
        for k, color in ((4, "#c0392b"), (8, "#e67e22"), (16, "#2980b9")):
            ax.axvline(2 * k, color=color, linestyle="--", linewidth=1.0,
                       label=f"stride {k} 可定位下限 (~{2*k}px)")
        ax.set_xlabel(f"letterbox to {args.imgsz}px 后的目标长边 (px)", fontsize=9)
        ax.set_ylabel("GT boxes", fontsize=9)
        ax.set_title("Ground-truth object size distribution", fontsize=11)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(args.out / "size_hist.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] 跳过直方图（{exc}）", file=sys.stderr)

    print("\n".join(lines))
    print(f"\n输出目录: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
