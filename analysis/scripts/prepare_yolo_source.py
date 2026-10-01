# -*- coding: utf-8 -*-
"""把一份 YOLO 数据集目录补成符合仓库 `layout: dfire` 约定的形态。

问题
----
FireSmoke-YOLO 的负样本（`NoFileSmoke*`，2,199 张）**没有标签文件**，而仓库
`firesmoke/data.py: inventory()` 对 `layout == "dfire"` 会校验

    expected = {图 stem}  ==  actual = {标签 stem}

不相等就 `raise ValueError("image/label names do not match")`。

空的 `.txt` 才是 YOLO 表示"纯背景图"的标准做法，所以这里把缺失的补成空文件。
这是**纯新增**：不修改、不删除任何既有文件。

用法
----
cd <repo>
python analysis/scripts/prepare_yolo_source.py --root datasets/FireSmoke-YOLO --dry-run
python analysis/scripts/prepare_yolo_source.py --root datasets/FireSmoke-YOLO --apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--apply", action="store_true", help="真正写入；不加则只报告（dry-run）")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_dir():
        print("目录不存在:", root)
        return 2

    total_new = 0
    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        img_dir = root / "images" / split
        lab_dir = root / "labels" / split
        if not img_dir.is_dir():
            print(f"[skip] 无 {img_dir}")
            continue
        lab_dir.mkdir(parents=True, exist_ok=True)

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        stems = {p.stem for p in imgs}
        have = {p.stem for p in lab_dir.glob("*.txt")}
        missing = sorted(stems - have)
        orphan = sorted(have - stems)   # 有标签无图：也应报出来
        empty_existing = sum(1 for s in (stems & have) if not (lab_dir / f"{s}.txt").stat().st_size)

        print(f"--- {split}")
        print(f"    图 {len(imgs)}  标签 {len(have)}  缺标签(负样本) {len(missing)}  有标签无图 {len(orphan)}")
        print(f"    已有空标签文件 {empty_existing}")
        if orphan:
            print(f"    !! 有标签无图的 stem（前 5）: {orphan[:5]}")

        if not a.apply:
            continue
        for s in missing:
            (lab_dir / f"{s}.txt").write_text("", encoding="utf-8")
        total_new += len(missing)
        print(f"    已创建 {len(missing)} 个空标签文件")

    if not a.apply:
        print("\n(dry-run，未写入任何文件。加 --apply 执行)")
    else:
        print(f"\n完成：共新增 {total_new} 个空标签文件（未修改/删除任何既有文件）")
        print("回退: 需要删掉的正是那些「0 字节 且 原本没有标签文件」的新增项。")
        print("      注意原数据集里本来就有若干空标签文件(真实背景图)，所以不要用 -size 0 批量删。")
        print("      精确回退（按本次记录的缺失 stem 列表逐个删）: 见上方 --apply 时打印的计数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
