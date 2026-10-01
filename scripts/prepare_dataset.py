#!/usr/bin/env python3
"""把 D-Fire 原始数据集按 9:1 切分并整理为 data/raw/D-Fire（图片 + 标签严格一一对应）。

【历史记录 / 可复现说明】
本脚本用于最初的整理。执行时原始数据位于发起者机器上的 D-Fire(1)/
（train 17221 + test 4306）。整理完成并校验通过后，原始目录及其 zip 已按计划清理，
全部 21527 张图片均已并入 data/raw/D-Fire/，无数据丢失。
如需重跑，请先把原始 D-Fire 数据恢复到 --src 所指位置。

划分规则：
  - 原始 train (17221) -> 按 9:1 随机切分为 train(15499) / val(1722)，seed 固定可复现
  - 原始 test  (4306)  -> 原样作为 test
类别沿用 D-Fire 官方定义：0 = smoke, 1 = fire

用法: python scripts/prepare_dataset.py --src <原始 D-Fire 目录> [--dst data/raw/D-Fire]
说明: 不硬编码任何机器路径；--dst 缺省为 <repo>/data/raw/D-Fire。
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # <repo>/
SRC = ROOT / "data/raw/D-Fire(1)"   # 占位，实际由 --src 指定
DST = ROOT / "data/raw/D-Fire"
SEED = 0
VAL_RATIO = 0.1
IMG_EXT = ".jpg"
LBL_EXT = ".txt"


def paired_stems(split: str) -> list[str]:
    """返回该 split 中图片与标签严格同名的 stem 列表；不一一对应则直接报错。"""
    imgs = {p.stem for p in (SRC / split / "images").glob(f"*{IMG_EXT}")}
    lbls = {p.stem for p in (SRC / split / "labels").glob(f"*{LBL_EXT}")}
    if not imgs:
        raise SystemExit(f"[ERROR] {SRC / split / 'images'} 下没有 {IMG_EXT} 图片")
    only_img, only_lbl = imgs - lbls, lbls - imgs
    if only_img or only_lbl:
        raise SystemExit(
            f"[ERROR] {split}: 图片/标签不一一对应 "
            f"(有图无标签 {len(only_img)}, 有标签无图 {len(only_lbl)}); "
            f"示例 {sorted(only_img)[:3]} {sorted(only_lbl)[:3]}"
        )
    return sorted(imgs)


def copy_split(split_src: str, split_dst: str, names: list[str]) -> None:
    for s in names:
        shutil.copy2(SRC / split_src / "images" / f"{s}{IMG_EXT}",
                     DST / "images" / split_dst / f"{s}{IMG_EXT}")
        shutil.copy2(SRC / split_src / "labels" / f"{s}{LBL_EXT}",
                     DST / "labels" / split_dst / f"{s}{LBL_EXT}")
    print(f"  {split_dst:5s} <- {split_src:5s}: {len(names)} 对 (图+标签)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, type=Path,
                        help="原始 D-Fire 目录（需含 train/ 与 test/，各自含 images/ 与 labels/）")
    parser.add_argument("--dst", type=Path, default=ROOT / "data/raw/D-Fire",
                        help="输出目录，默认 <repo>/data/raw/D-Fire")
    return parser.parse_args()


def main() -> None:
    global SRC, DST
    args = parse_args()
    SRC = args.src.expanduser().resolve()
    DST = args.dst.expanduser().resolve()
    if not SRC.is_dir():
        raise SystemExit(f"[ERROR] --src 不存在: {SRC}")

    for d in ("train", "val", "test"):
        (DST / "images" / d).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / d).mkdir(parents=True, exist_ok=True)

    train_all = paired_stems("train")
    test_all = paired_stems("test")

    rng = random.Random(SEED)
    shuffled = train_all[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * VAL_RATIO)
    val_names = sorted(shuffled[:n_val])
    train_names = sorted(shuffled[n_val:])

    print(f"D-Fire/train 共 {len(train_all)} -> "
          f"train {len(train_names)} / val {len(val_names)} (seed={SEED}, val_ratio={VAL_RATIO})")
    copy_split("train", "train", train_names)
    copy_split("train", "val", val_names)
    copy_split("test", "test", test_all)
    print("done")


if __name__ == "__main__":
    main()
