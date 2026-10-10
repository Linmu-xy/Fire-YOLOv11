#!/usr/bin/env python
"""Arm B 训练数据准备：D-Fire 全量 + FireSmoke-YOLO train 负样本 ×R（源目录层合并）。

为什么不用 `neg_repeat`
-----------------------
`firesmoke/data.py` 的 `neg_repeat` 会重复 train 划分内**所有**无 GT 框的图像。
D-Fire train 自带 7,043 张背景图，而它们在源域 val 上的触发率实测是 0/790，
即模型早已学会「这不是火」—— 重复它们（`dfire_hn3`）没有信息增量。
真正造成跨域误报的暖色物体（红花 / 红发 / 橙色灯带 / 暖色花盆）在 D-Fire 背景集里不存在。
因此本脚本只重复 FireSmoke-YOLO 的 train 负样本（实测触发 2.42% / 3.33%，
比 D-Fire 背景难约 10 倍），D-Fire 的背景图保持每张一次。

产物布局（`layout: dfire`，供 `configs/protocol.yaml` 的 `dfire_fsneg3` 条目使用）
    <out>/images/{train,val,test}/...
    <out>/labels/{train,val,test}/...
  * D-Fire 三个 split：symlink 到原文件，不复制、不修改源数据集
  * FS train 负样本：hardlink × R（同文件系统零额外空间，失败退回 copy2），空标签
  * val / test 与 `dfire` 逐图一致 → 域内 val 指标可与 `paper-v1` 对照直接比较

类别顺序
--------
D-Fire 是 0=smoke / 1=fire，FireSmoke-YOLO 是 0=fire / 1=smoke。
本脚本只并入 FS 的**空标签**图，不含任何类别标注，因此顺序差异不适用。
将来若要并入 FS 的正样本标注，必须先换到 D-Fire 顺序（见 PROJECT_LAYOUT 坑 2）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
MARKER = ".armb_source.json"


def repo_path(value):
    """protocol 里的路径相对仓库根；已经是绝对路径就原样返回。"""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def images_of(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def duplicate(src, dst):
    """hardlink 优先（同文件系统零额外空间），失败退回复制。"""
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build(dfire_root, fs_root, out, repeat, force):
    for split in SPLITS:
        for sub in ("images", "labels"):
            if not (dfire_root / sub / split).is_dir():
                raise FileNotFoundError(dfire_root / sub / split)
    if not (fs_root / "images" / "train").is_dir():
        raise FileNotFoundError(fs_root / "images" / "train")

    if out.exists():
        if not force:
            raise FileExistsError(f"{out} 已存在；确认要重建请加 --force")
        if not (out / MARKER).is_file():
            raise RuntimeError(f"{out} 不是本脚本生成的目录（缺少 {MARKER}），拒绝删除")
        shutil.rmtree(out)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    counts, train_stems = {}, set()
    for split in SPLITS:
        number = 0
        for image in images_of(dfire_root / "images" / split):
            label = dfire_root / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            (out / "images" / split / image.name).symlink_to(image)
            (out / "labels" / split / label.name).symlink_to(label)
            train_stems.add(image.stem)
            number += 1
        counts[f"dfire_{split}"] = number

    negatives = []
    for image in images_of(fs_root / "images" / "train"):
        label = fs_root / "labels" / "train" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        if label.stat().st_size == 0:
            negatives.append(image)
    if not negatives:
        raise RuntimeError("未找到任何 FireSmoke-YOLO train 负样本（空标签图）")

    modes, added = {}, 0
    for image in negatives:
        for k in range(1, repeat + 1):
            stem = f"fsneg_{image.stem}_r{k}"
            if stem in train_stems:
                raise RuntimeError(f"命名冲突：{stem} 与已有 train 图同名")
            train_stems.add(stem)
            mode = duplicate(image, out / "images" / "train" / f"{stem}{image.suffix.lower()}")
            modes[mode] = modes.get(mode, 0) + 1
            (out / "labels" / "train" / f"{stem}.txt").write_text("", encoding="utf-8")
            added += 1
    counts["fsneg_train"] = added

    card = {
        "generator": "scripts/prepare_armb_dataset.py",
        "purpose": "Arm B: D-Fire 全量 + FireSmoke-YOLO train 负样本 ×R（源目录层合并）",
        "dfire_root": str(dfire_root),
        "fs_root": str(fs_root),
        "repeat": repeat,
        "fs_negative_images": len(negatives),
        "counts": counts,
        "link_modes": modes,
        "note": "val/test 与 dfire 逐图一致；只并入 FS 空标签负样本，类别顺序差异不适用",
    }
    (out / MARKER).write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    return card


def audit_duplicates(fs_images, dfire_train, radius, report_path):
    """sha256 精确重复 + dHash 近重复（FS train 负样本 vs D-Fire train）。"""
    sys.path.insert(0, str(ROOT))
    from firesmoke.data import BKTree, image_hashes

    tree, exact = BKTree(), {}
    print(f"[audit] 哈希 D-Fire train {len(dfire_train)} 张…", flush=True)
    for i, image in enumerate(dfire_train):
        sha, dhash = image_hashes(image)
        exact.setdefault(sha, []).append(f"D-Fire/train/{image.name}")
        tree.add(dhash, i)
        if (i + 1) % 2000 == 0:
            print(f"[audit]   {i + 1}/{len(dfire_train)}", flush=True)

    print(f"[audit] 哈希 FS train 负样本 {len(fs_images)} 张并查重…", flush=True)
    findings = []
    for j, image in enumerate(fs_images):
        sha, dhash = image_hashes(image)
        hits = [{"dfire": exact[s][0], "distance": 0, "byte_identical": True} for s in [sha] if s in exact]
        hits += [{"dfire": f"D-Fire/train/{dfire_train[k].name}", "distance": d, "byte_identical": False}
                 for k, d in tree.query(dhash, radius)]
        if hits:
            findings.append({"fs_negative": image.name, "dfire_candidates": hits[:5]})
        if (j + 1) % 250 == 0:
            print(f"[audit]   {j + 1}/{len(fs_images)}", flush=True)

    report = {
        "fs_negatives_checked": len(fs_images),
        "dfire_train_checked": len(dfire_train),
        "dhash_radius": radius,
        "images_with_candidates": len(findings),
        "policy": "候选仅供人工核验；即使存在近重复也不自动删除，只按结论决定是否接入",
        "candidates": findings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", default=str(ROOT / "configs/protocol.yaml"))
    parser.add_argument("--out", default="data/raw/D-Fire-FSneg", help="合并后的源目录（相对仓库根）")
    parser.add_argument("--repeat", type=int, default=3, help="FS train 负样本的重复份数 R")
    parser.add_argument("--force", action="store_true", help="重建已有产物（仅限本脚本生成的目录）")
    parser.add_argument("--audit", action="store_true", help="额外做近重复审计（接入前的必做项）")
    parser.add_argument("--audit-radius", type=int, default=4)
    parser.add_argument("--audit-report", default="analysis/work/armb_duplicate_audit.json")
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat 必须 >= 1")
    cfg = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    dfire_root = repo_path(cfg["datasets"]["dfire"]["root"]).resolve()
    fs_root = repo_path(cfg["datasets"]["fsyolo"]["root"]).resolve()
    out = repo_path(args.out)

    # 审计时允许复用已生成的源目录：审计本身不改数据，不必重建
    if out.exists() and args.audit and not args.force:
        if not (out / MARKER).is_file():
            raise SystemExit(f"{out} 不是本脚本生成的目录（缺少 {MARKER}）")
        card = json.loads((out / MARKER).read_text(encoding="utf-8"))
        print(f"[skip] 复用已有源目录：{out}")
    else:
        card = build(dfire_root, fs_root, out, args.repeat, args.force)
    print(json.dumps(card, ensure_ascii=False, indent=1))
    print(f"[ok] 源目录已生成：{out}")

    if args.audit:
        fs_images = sorted((out / "images" / "train").glob("fsneg_*_r1*"))
        dfire_train = images_of(dfire_root / "images" / "train")
        report = audit_duplicates(fs_images, dfire_train, args.audit_radius, repo_path(args.audit_report))
        print(f"[audit] {report['images_with_candidates']}/{report['fs_negatives_checked']} 张 FS 负样本在 "
              f"D-Fire train（{report['dfire_train_checked']} 张）里找到候选 → {repo_path(args.audit_report)}")


if __name__ == "__main__":
    main()
