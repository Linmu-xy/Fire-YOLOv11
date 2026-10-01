#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_clean_vs_dup.py
====================
用**已有 checkpoint** 量化"近重复泄漏"对 val 指标的影响 —— 不需要重训任何东西。

原理
----
split_integrity_audit.py --emit-manifests 会产出四类子集:
    val_full   全部 val                       (你一直在报的那个数)
    val_clean  在 train 里没有近重复的 val
    val_dup    在 train 里有近重复的 val      (泄漏子集)
    test_full / test_clean / test_dup        同理

同一组权重分别在这些子集上跑 ultralytics val, 就能拆出:

    泄漏溢价 (leakage premium) = mAP(val_full) - mAP(val_clean)

如果溢价很大, 说明 val 指标里有相当一部分来自"背过的场景";
这种指标对结构改动(比如加 P2)天生不敏感 —— 因为能背的都已经背会了,
改动只能体现在真正泛化的那一小部分图上。

用法
----
python eval_clean_vs_dup.py \
    --audit-out audit_out \
    --weights runs/p2_seed0/weights/best.pt runs/baseline_800/weights/best.pt \
    --imgsz 640 --out eval_out

依赖: ultralytics (和训练时同一版本)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="在 full/clean/dup 子集上评估已有 checkpoint, 量化泄漏溢价")
    ap.add_argument("--audit-out", required=True, help="split_integrity_audit.py 的输出目录 (含 manifests/)")
    ap.add_argument("--weights", nargs="+", required=True, help="一个或多个 .pt 权重路径")
    ap.add_argument("--labels", nargs="*", default=None, help="与 weights 一一对应的显示名, 省略则用文件名")
    ap.add_argument("--splits", default="val", help="要评估的 split, 逗号分隔 (val,test)")
    ap.add_argument("--subsets", default="full,clean,dup", help="子集类型, 逗号分隔")
    ap.add_argument("--imgsz", type=int, default=640, help="评估分辨率, 必须与该 checkpoint 的训练分辨率一致")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="eval_out")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[fatal] 无法导入 ultralytics: {e}", file=sys.stderr)
        print("        本脚本必须在与训练相同的环境里运行。", file=sys.stderr)
        return 2

    man = Path(a.audit_out) / "manifests"
    if not man.is_dir():
        print(f"[fatal] 找不到 {man}; 请先跑 split_integrity_audit.py --emit-manifests", file=sys.stderr)
        return 2

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    subsets = [s.strip() for s in a.subsets.split(",") if s.strip()]

    jobs = []
    for sp in splits:
        for sub in subsets:
            yml = man / f"{sp}_{sub}.yaml"
            txt = man / f"{sp}_{sub}.txt"
            if not yml.exists():
                continue
            n = len([l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]) if txt.exists() else 0
            if n == 0:
                print(f"[skip] {sp}_{sub}: 子集为空 (0 张), 跳过")
                continue
            jobs.append({"split": sp, "subset": sub, "yaml": str(yml), "n_images": n})

    if not jobs:
        print("[fatal] 没有可评估的子集, 请检查 --audit-out 与 --subsets", file=sys.stderr)
        return 2

    names = a.labels if a.labels else [Path(w).parent.parent.name or Path(w).name for w in a.weights]
    if len(names) != len(a.weights):
        print("[fatal] --labels 数量必须与 --weights 一致", file=sys.stderr)
        return 2

    print("评估矩阵:")
    for j in jobs:
        print(f"  {j['split']}_{j['subset']:<6} {j['n_images']:>6} 张  {j['yaml']}")
    print()

    records = []
    for w, nm in zip(a.weights, names):
        wp = Path(w)
        if not wp.exists():
            print(f"[skip] 权重不存在: {w}")
            continue
        for j in jobs:
            print(f"[val] {nm} / {j['split']}_{j['subset']} ...", flush=True)
            try:
                model = YOLO(str(wp))
                kw = dict(data=j["yaml"], split=j["split"], imgsz=a.imgsz,
                          batch=a.batch, plots=False, verbose=False)
                if a.device:
                    kw["device"] = a.device
                r = model.val(**kw)
                box = getattr(r, "box", None)
                rec = {
                    "weights": str(wp), "name": nm, "split": j["split"], "subset": j["subset"],
                    "n_images": j["n_images"], "imgsz": a.imgsz,
                    "mAP50-95": float(getattr(box, "map", float("nan"))),
                    "mAP50": float(getattr(box, "map50", float("nan"))),
                    "mAP75": float(getattr(box, "map75", float("nan"))) if hasattr(box, "map75") else None,
                    "precision": float(getattr(box, "mp", float("nan"))),
                    "recall": float(getattr(box, "mr", float("nan"))),
                    "per_class_map50_95": [float(v) for v in getattr(box, "maps", [])] if hasattr(box, "maps") else [],
                }
                records.append(rec)
                print(f"       mAP50-95={rec['mAP50-95']:.5f}  mAP50={rec['mAP50']:.5f}  "
                      f"P={rec['precision']:.5f}  R={rec['recall']:.5f}")
            except Exception as e:
                print(f"       [error] {type(e).__name__}: {e}")
                records.append({"weights": str(wp), "name": nm, "split": j["split"],
                                "subset": j["subset"], "error": str(e)})

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_subsets.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 报告 -------------------------------------------------------- #
    L = ["# 泄漏溢价评估\n", f"- imgsz: {a.imgsz}", f"- 来源清单: `{man}`\n"]
    by_key = {}
    for r in records:
        if "error" in r:
            continue
        by_key[(r["name"], r["split"], r["subset"])] = r

    L.append("## 原始结果\n")
    L.append("| 权重 | split | 子集 | 图数 | mAP50-95 | mAP50 | P | R |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in records:
        if "error" in r:
            L.append(f"| {r['name']} | {r['split']} | {r['subset']} | - | ERROR | | | |")
            continue
        L.append(f"| {r['name']} | {r['split']} | {r['subset']} | {r['n_images']} | "
                 f"{r['mAP50-95']:.5f} | {r['mAP50']:.5f} | {r['precision']:.5f} | {r['recall']:.5f} |")
    L.append("")

    L.append("## 泄漏溢价 (mAP50-95)\n")
    L.append("| 权重 | split | full | clean | dup | 溢价 full-clean | 溢价 dup-clean |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for nm in names:
        for sp in splits:
            f = by_key.get((nm, sp, "full"))
            c = by_key.get((nm, sp, "clean"))
            d = by_key.get((nm, sp, "dup"))
            if not f:
                continue

            def g(x):
                return f"{x['mAP50-95']:.5f}" if x else "-"

            p1 = f"{f['mAP50-95'] - c['mAP50-95']:+.5f}" if c else "-"
            p2 = f"{d['mAP50-95'] - c['mAP50-95']:+.5f}" if (d and c) else "-"
            L.append(f"| {nm} | {sp} | {g(f)} | {g(c)} | {g(d)} | {p1} | {p2} |")
    L.append("")

    L.append("## 怎么读\n")
    L.append("- **溢价 dup-clean 很大**: val 分数里有一大块来自在 train 里出现过近重复的图。")
    L.append("  这部分分数对结构改动天然不敏感(能背的都背会了), 会稀释掉你想测的效应。")
    L.append("- **clean 上的排序才是可信比较**: 如果 P2 的收益只在 full 上出现、clean 上消失, ")
    L.append("  那它就是在拟合泄漏; 反过来若 clean 上差异更清楚, 说明 clean 子集信噪比更高。")
    L.append("- 两个子集图数不同, 绝对 mAP 不可直接跨数据集比较, 只看同权重下的差值。")
    L.append("- 本评估不改动任何权重, 不构成新的训练结果。\n")

    (out_dir / "eval_subsets.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n输出: {out_dir.resolve()}  (eval_subsets.json / eval_subsets.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
