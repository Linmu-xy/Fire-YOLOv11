# -*- coding: utf-8 -*-
"""固定阈值 vs 每种子重选阈值：告警指标的门槛差多少。

动机
----
实测发现「固定 1% FPR 的图像级 recall」种子 σ 极大（fire 1.28pp / smoke 3.94pp），
比 mAP50-95 还差。原因不是指标本身，而是**每个种子在自己的 val 上重新选阈值**,
而阈值本身 σ 就有 3.4~4.3pp —— 把这份方差传染给了 recall。

本脚本用**已有产物**（零训练、零推理）验证：把阈值冻结成**一个共享值**后，
recall 的种子方差是否显著下降。直接复用仓库自己的 select_threshold / alarm_metrics，
保证语义与官方口径完全一致。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/fixed_threshold_probe.py --out analysis/out_reliability
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from firesmoke.reliability import alarm_metrics, arrays, select_threshold  # noqa: E402

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def stats(xs):
    mean = st.fmean(xs)
    sd = st.stdev(xs)
    n = len(xs)
    return mean, sd, T_CRIT.get(n - 1, 1.96 + 2.4 / max(n - 1, 1)) * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis/out_reliability")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-fpr", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.dir)
    out = Path(a.out) if a.out else root
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    val, test = {}, {}
    for s in seeds:
        val[s] = arrays(json.loads((root / f"val_s{s}" / "predictions.json").read_text()))
        test[s] = arrays(json.loads((root / f"test_s{s}" / "predictions.json").read_text()))
    print(f"载入 val/test predictions: {len(seeds)} 个种子")

    # ---------- A) 每种子在自己的 val 上重选阈值（现有协议） ----------
    per_seed_thr, per_seed_alarm = [], []
    for s in seeds:
        sv, lv = val[s]
        ts = [select_threshold(sv[:, c], lv[:, c], a.max_fpr) for c in (0, 1)]
        per_seed_thr.append([t["threshold"] for t in ts])
        stt, ltt = test[s]
        per_seed_alarm.append(alarm_metrics(stt, ltt, [t["threshold"] for t in ts]))

    # ---------- B) 阈值冻结成共享值（用 seed0 的 val 拟合，应用到所有种子） ----------
    sv0, lv0 = val[seeds[0]]
    fixed = [select_threshold(sv0[:, c], lv0[:, c], a.max_fpr)["threshold"] for c in (0, 1)]
    fixed_alarm = []
    for s in seeds:
        stt, ltt = test[s]
        fixed_alarm.append(alarm_metrics(stt, ltt, fixed))

    # ---------- C) 阈值取三种子 val 拟合值的均值（另一种共享方式） ----------
    pooled = [float(np.mean([per_seed_thr[i][c] for i in range(len(seeds))])) for c in (0, 1)]
    pooled_alarm = []
    for s in seeds:
        stt, ltt = test[s]
        pooled_alarm.append(alarm_metrics(stt, ltt, pooled))

    def collect(records, key):
        return [r[key] for r in records]

    rows = []
    for label, recs, thr in (("A. 每种子重选阈值（现有协议）", per_seed_alarm, per_seed_thr),
                             ("B. 冻结为 seed0 阈值", fixed_alarm, [fixed] * len(seeds)),
                             ("C. 冻结为三种子均值阈值", pooled_alarm, [pooled] * len(seeds))):
        for metric in ("fire.recall", "smoke.recall", "background_alarm_fraction",
                       "fire.negative_image_fpr", "smoke.negative_image_fpr"):
            cls, _, field = metric.partition(".")
            if field == "background_alarm_fraction" or cls == "background_alarm_fraction":
                xs = [r["background_alarm_fraction"] for r in recs]
            else:
                xs = [r[cls][field] for r in recs]
            mean, sd, mdd = stats(xs)
            rows.append({"scheme": label, "metric": metric, "mean": mean,
                         "sd_pp": sd * 100, "mdd_pp": mdd * 100,
                         "per_seed": [round(x, 5) for x in xs]})
        if thr and len(thr[0]) == 2:
            for c, nm in ((0, "fire"), (1, "smoke")):
                xs = [t[c] for t in thr]
                mean, sd, mdd = stats(xs)
                rows.append({"scheme": label, "metric": f"threshold.{nm}", "mean": mean,
                             "sd_pp": sd * 100, "mdd_pp": mdd * 100,
                             "per_seed": [round(x, 4) for x in xs]})

    ref = 0.9176  # test 集上三种子实测的 mAP50-95 MDD
    L = ["# 固定阈值 vs 每种子重选阈值 —— 告警指标的门槛对比\n",
         f"- 数据: 已有产物 `{root}/`，零训练零推理",
         f"- 每类 FPR 预算: {a.max_fpr:.1%}；参照 mAP50-95 的 MDD = {ref:.4f}pp",
         "- 阈值一律作用在**原始分数**上（不经温度缩放），三种方案口径一致\n",
         "| 方案 | 指标 | 均值 | 种子σ | **MDD** | vs mAP50-95 |",
         "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        ratio = ref / r["mdd_pp"] if r["mdd_pp"] > 0 else float("inf")
        tag = f"低 {ratio:.1f} 倍" if ratio > 1.05 else (f"高 {1/ratio:.1f} 倍" if ratio < 0.95 else "相当")
        L.append(f"| {r['scheme']} | `{r['metric']}` | {r['mean']:.5f} | {r['sd_pp']:.4f}pp | "
                 f"**{r['mdd_pp']:.4f}pp** | {tag} |")
    L.append("")

    def pick(scheme, metric):
        for r in rows:
            if r["scheme"] == scheme and r["metric"] == metric:
                return r
        return None

    L.append("## 读法\n")
    for m in ("fire.recall", "smoke.recall", "background_alarm_fraction"):
        A, B = pick("A. 每种子重选阈值（现有协议）", m), pick("B. 冻结为 seed0 阈值", m)
        if A and B:
            L.append(f"- `{m}`：每种子重选 MDD = {A['mdd_pp']:.2f}pp → 冻结阈值 "
                     f"{B['mdd_pp']:.2f}pp（**改善 {A['mdd_pp'] / max(B['mdd_pp'], 1e-9):.1f} 倍**）")
    L.append("")
    L.append("> 若冻结阈值后 MDD 大幅下降，说明「每种子重选阈值」是方差主源，")
    L.append("> 论文报固定 FPR 下的 recall 时**必须共用一个阈值**，否则该指标不可用于方法比较。")
    L.append(">")
    L.append("> 注意：冻结阈值会让 FPR 不再精确等于 1%（各模型分布不同），")
    L.append("> 这是刻意的取舍 —— 用 FPR 的轻微偏移换取指标的可比性。报告时需并列说明。")
    (out / "fixed_threshold_probe.md").write_text("\n".join(L), encoding="utf-8")
    (out / "fixed_threshold_probe.json").write_text(
        json.dumps({"max_fpr": a.max_fpr, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
