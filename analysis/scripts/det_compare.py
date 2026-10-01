# -*- coding: utf-8 -*-
"""目标域上的能力对照：在**同一背景误报率**下比较两个源域模型的召回。

为什么必须做这一步
------------------
FireSmoke-YOLO 源域训练的模型在 FASDD 负样本上的告警率是 4.79%，D-Fire 源域是 8.60%
（各自源域 1% FPR 工作点）。但这**不能**直接得出"FS 的负样本更好"，因为：
  * FS 是弱得多的模型（源域 smoke 召回 0.185 vs D-Fire 0.681）
  * **检得少，天然就报得少** —— 漏检与少报是同一个现象的两面

正确做法：扫描工作点，画出「背景误报率 vs 目标域召回」的权衡曲线，
在**相同误报率**下比召回（或反之）。只有在同一误报率下召回也更高，
才能说"误报更少"不是"能力更弱"的副产品。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/scripts/det_compare.py \
  --model "D-Fire:analysis/work/out_zeroshot/tgt_baseline640_s0/predictions.json:0.3634:0.6182" \
  --model "FS:analysis/work/calib/tgt_fs_fasdd/predictions.json:0.3018:0.5111" \
  --out analysis/work/det_compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from firesmoke.reliability import arrays  # noqa: E402

KS = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 6.0]
MATCH_ALARMS = [0.02, 0.05, 0.10]


def curve(tf: float, ts: float, scores, labels):
    """给定一对阈值，返回 (fire 召回, smoke 召回, 背景告警率, 全类召回)。"""
    fire_ids = labels[:, 0] == 1
    smoke_ids = labels[:, 1] == 1
    bg_ids = labels.sum(1) == 0
    any_pos = labels.sum(1) > 0
    rf = float((scores[fire_ids, 0] >= tf).mean()) if fire_ids.any() else float("nan")
    rs = float((scores[smoke_ids, 1] >= ts).mean()) if smoke_ids.any() else float("nan")
    alarm = (scores[:, 0] >= tf) | (scores[:, 1] >= ts)
    neg = float(alarm[bg_ids].mean()) if bg_ids.any() else float("nan")
    # "全类召回": 正样本图里，该类 GT 至少被命中一次
    hit = np.zeros(len(scores), dtype=bool)
    if fire_ids.any():
        hit[fire_ids] |= scores[fire_ids, 0] >= tf
    if smoke_ids.any():
        hit[smoke_ids] |= scores[smoke_ids, 1] >= ts
    rec_any = float(hit[any_pos].mean()) if any_pos.any() else float("nan")
    return rf, rs, neg, rec_any


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="名称:preds.json:fire阈值:smoke阈值，可多次给")
    ap.add_argument("--out", default="analysis/work/det_compare")
    a = ap.parse_args()

    models = []
    for spec in a.model:
        name, path, tf, ts = spec.split(":")
        art = json.loads(Path(path).read_text())
        if art.get("split") != "test":
            print(f"!! {name}: 需要 test 划分，实际 {art.get('split')}")
            return 2
        sc, lb = arrays(art)
        models.append({"name": name, "path": path, "tf": float(tf), "ts": float(ts),
                       "scores": sc, "labels": lb})
        print(f"[{name}] 图 {len(sc)}  正样本图 {(lb.sum(1) > 0).sum()}  背景图 {(lb.sum(1) == 0).sum()}")

    # 基线工作点（k=1 即各自源域 1% FPR）
    L = ["# 目标域能力对照（FASDD test）\n",
         "- 工作点由「源域阈值 × k」给出；k=1 即该模型在源域 val 上 1% FPR 的点",
         "- `背景告警率` = 无 GT 图像中任一类的分数超阈；`全类召回` = 正样本图中 GT 被命中比例\n"]
    L.append("## 1. 各模型的工作点曲线\n")
    L.append("| 模型 | k | fire阈值 | smoke阈值 | fire召回 | smoke召回 | 全类召回 | 背景告警率 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in models:
        for k in KS:
            rf, rs, neg, ra = curve(m["tf"] * k, m["ts"] * k, m["scores"], m["labels"])
            L.append(f"| {m['name']} | {k} | {m['tf']*k:.4f} | {m['ts']*k:.4f} | "
                     f"{rf:.4f} | {rs:.4f} | {ra:.4f} | **{neg:.4f}** |")
        L.append("")

    # 在同一背景告警率下比较召回
    L.append("## 2. 同背景告警率下的召回（能力对照的核心）\n")
    L.append("| 目标背景告警率 | 模型 | 实际告警率 | fire召回 | smoke召回 | 全类召回 |")
    L.append("|---:|---|---:|---:|---:|---:|")
    summary = {}
    for target in MATCH_ALARMS:
        for m in models:
            best = None
            for k in np.linspace(0.05, 12, 240):
                rf, rs, neg, ra = curve(m["tf"] * k, m["ts"] * k, m["scores"], m["labels"])
                if best is None or abs(neg - target) < abs(best[2] - target):
                    best = (rf, rs, neg, ra, k)
            rf, rs, neg, ra, k = best
            summary[f"{m['name']}@{target}"] = {"fire_recall": rf, "smoke_recall": rs,
                                                "alarm": neg, "any_recall": ra, "k": k}
            L.append(f"| {target*100:.0f}% | {m['name']} | {neg*100:.2f}% | {rf:.4f} | {rs:.4f} | {ra:.4f} |")
        L.append("")

    L.append("## 3. 判读\n")
    L.append("- **在同一背景告警率下，谁的全类召回更高，谁才真的更好。**")
    L.append("  若 FS 在同告警率下召回也更高 → 「负样本类型」确实是主因。")
    L.append("  若 FS 只是整条曲线更保守（同告警率下召回更低）→ 它的低误报只是「能力弱」的副产品，")
    L.append("  **不能**据此说 FS 的负样本更好。")
    L.append("- 只比源域 1% FPR 那一个点是不够的：那相当于在两条曲线的不同位置取值。")
    L.append("")
    L.append("> 不含任何新训练；数据来自已有 checkpoint 的重新评估。")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "det_compare.md").write_text("\n".join(L), encoding="utf-8")
    (out / "det_compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
