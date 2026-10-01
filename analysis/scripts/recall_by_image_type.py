# -*- coding: utf-8 -*-
"""按"图像类别"拆召回：区分"模型本质依赖火"与"只在跨域时才依赖火"。

为什么关键
----------
跨域 FASDD 上：纯烟图 smoke 召回仅 29.9–39.0%，而火烟同框图有 48.1–53.2%。
但这**不能**直接得出"模型靠火推断烟"，因为那可能只是域差。
必须看**域内**（D-Fire val）同一拆分的表现：
  - 若域内纯烟图召回也很低 → 依赖火是**模型内在**特性，换数据集救不了，得改数据构成/损失
  - 若域内纯烟图召回高、只有跨域才崩 → 那是**烟的域差**，解法是加烟的多样性

类别从 GT 框直接推出（FASDD 的文件名前缀与 GT 完全一致，D-Fire 无前缀但可由 GT 判定）：
    fire_only / smoke_only / both / none(背景)

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/recall_by_image_type.py --out analysis/out_recall_by_type
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CATS = ("fire_only", "smoke_only", "both", "none")


def categorize(truth) -> str:
    cls = {int(t[0]) for t in truth}
    if not cls:
        return "none"
    if cls == {0}:
        return "fire_only"
    if cls == {1}:
        return "smoke_only"
    return "both"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default="analysis/out_zeroshot/zeroshot_eval.json")
    ap.add_argument("--out", default="analysis/out_recall_by_type")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(a.eval_json).read_text())
    t_fire, t_smoke = float(cfg["thresholds"][0]), float(cfg["thresholds"][1])
    print(f"统一使用源域冻结阈值: fire={t_fire:.4f} smoke={t_smoke:.4f}\n")

    report = {}
    for name, info in sorted(cfg["results"].items()):
        for side, key in (("src", "src_val"), ("tgt", "tgt_test")):
            pp = Path(info[key])
            if not pp.is_file():
                continue
            art = json.loads(pp.read_text())
            acc = defaultdict(lambda: {"n": 0, "fire_tot": 0, "fire_hit": 0,
                                       "smoke_tot": 0, "smoke_hit": 0})
            for im in art["images"]:
                cat = categorize(im["truth"])
                fs = [p[5] for p in im["predictions"] if int(p[0]) == 0]
                ss = [p[5] for p in im["predictions"] if int(p[0]) == 1]
                mf = max(fs) if fs else 0.0
                ms = max(ss) if ss else 0.0
                rec = acc[cat]
                rec["n"] += 1
                tcls = {int(t[0]) for t in im["truth"]}
                if 0 in tcls:
                    rec["fire_tot"] += 1
                    if mf >= t_fire:
                        rec["fire_hit"] += 1
                if 1 in tcls:
                    rec["smoke_tot"] += 1
                    if ms >= t_smoke:
                        rec["smoke_hit"] += 1
            report[f"{name}::{side}"] = {k: dict(v) for k, v in acc.items()}

    L = ["# 按图像类别拆召回（域内 vs 跨域，同一套源域冻结阈值）\n",
         f"- 阈值 fire={t_fire:.4f} / smoke={t_smoke:.4f}（来自源域 D-Fire val 的 1% FPR）",
         "- 图像类别由 GT 框推出：fire_only / smoke_only / both / none\n",
         "| 预测配置 | 域 | 图类别 | 图数 | **fire 召回** | **smoke 召回** |",
         "|---|---|---|---:|---:|---:|"]
    for cfgname, keys in [("p2_s0", ("p2_s0::src", "p2_s0::tgt")),
                          ("baseline640_s0", ("baseline640_s0::src", "baseline640_s0::tgt"))]:
        for k in keys:
            r = report.get(k)
            if not r:
                continue
            dom = "域内 D-Fire val" if k.endswith("::src") else "跨域 FASDD test"
            for cat in CATS:
                v = r.get(cat)
                if not v or v["n"] == 0:
                    continue
                fr = f"{v['fire_hit']/v['fire_tot']*100:.1f}%" if v["fire_tot"] else "—"
                sr = f"{v['smoke_hit']/v['smoke_tot']*100:.1f}%" if v["smoke_tot"] else "—"
                if cat == "none" and not v["fire_tot"] and not v["smoke_tot"]:
                    continue
                L.append(f"| {k.split('::')[0]} | {dom} | {cat} | {v['n']} | {fr} | {sr} |")
    L.append("")

    # 关键对照
    L.append("## 关键对照：smoke_only 图的召回\n")
    L.append("| 配置 | 域内 | 跨域 | 落差 |")
    L.append("|---|---:|---:|---:|")
    for cfgname in ("p2_s0", "baseline640_s0"):
        s = report.get(f"{cfgname}::src", {}).get("smoke_only")
        t = report.get(f"{cfgname}::tgt", {}).get("smoke_only")
        if not s or not t or not s["smoke_tot"] or not t["smoke_tot"]:
            continue
        a1 = s["smoke_hit"] / s["smoke_tot"] * 100
        a2 = t["smoke_hit"] / t["smoke_tot"] * 100
        L.append(f"| {cfgname} | **{a1:.1f}%** | **{a2:.1f}%** | {a2-a1:+.1f}pp |")
    L.append("")
    L.append("## 判读\n")
    L.append("- **域内 smoke_only 召回也低** → 依赖火是模型内在特性；解法是数据构成/损失，换数据集无用。")
    L.append("- **域内高、跨域崩** → 是烟的域差；解法是增加烟的多样性与来源。")
    L.append("- 同时看 `both` 一列：若 both 图的 smoke 召回明显高于 smoke_only，")
    L.append("  则「有火时烟好检」这个模式在两个域内都成立，是稳健的结构性发现。")
    L.append("")
    L.append("> 不含任何新训练或新推理；全部来自已有 predictions。")
    (out / "recall_by_image_type.md").write_text("\n".join(L), encoding="utf-8")
    (out / "recall_by_image_type.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
