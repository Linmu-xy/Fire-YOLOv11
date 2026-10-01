# -*- coding: utf-8 -*-
"""误报归因：FASDD 目标域的告警是哪些图像、哪一类、什么分辨率触发的。

背景
----
零样本结果：源域冻结的 1% FPR 阈值搬到 FASDD 后，背景告警比例 9.5–12.3%（源域 0/790）。
FASDD_CV 的文件名前缀本身就编码了图像级类别：
    neitherFireNorSmoke / fire / smoke / bothFireAndSmoke
而 `neitherFireNorSmoke` 的 6,533 张全部无 GT —— 就是背景集。
所以可以**完全离线**地把误报拆开。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/false_alarm_attribution.py --out analysis/out_false_alarm
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

POSITIVE_CATS = ("fire", "smoke", "bothFireAndSmoke")
NEG_CAT = "neitherFireNorSmoke"


def cat_of(img_id: str) -> str:
    m = re.match(r"^([A-Za-z]+)_", str(img_id).split("/")[-1])
    return m.group(1) if m else "(other)"


def resolve_image(img_root: str, img_id: str) -> Path:
    root = Path(img_root)
    parts = str(img_id).split("/")
    if len(parts) >= 3:
        c = root / parts[0] / "images" / parts[1] / "/".join(parts[2:])
        if c.exists():
            return c
    return root / img_id


def bucket(long_side: int) -> str:
    for lim, lab in ((480, "<=480"), (720, "481-720"), (1080, "721-1080"),
                     (1440, "1081-1440"), (1920, "1441-1920"), (10**9, ">1920")):
        if long_side <= lim:
            return lab
    return ">1920"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--img-root", default="artifacts/data")
    ap.add_argument("--eval-json", default="analysis/out_zeroshot/zeroshot_eval.json")
    ap.add_argument("--out", default="analysis/out_false_alarm")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(a.eval_json).read_text())
    thr = cfg["thresholds"]
    t_fire, t_smoke = float(thr[0]), float(thr[1])
    print(f"冻结阈值: fire={t_fire:.4f}  smoke={t_smoke:.4f}")

    from PIL import Image
    size_cache: dict[str, tuple[int, int]] = {}

    def size_of(p: Path):
        k = str(p)
        if k not in size_cache:
            try:
                with Image.open(p) as im:
                    size_cache[k] = im.size
            except Exception:
                size_cache[k] = (0, 0)
        return size_cache[k]

    preds = cfg["results"]
    report = {}
    for name, info in preds.items():
        pp = Path(info["tgt_test"])
        if not pp.is_file():
            continue
        art = json.loads(pp.read_text())
        neg = Counter()
        neg_res = defaultdict(lambda: [0, 0])
        pos = defaultdict(lambda: defaultdict(int))
        neg_max_fire, neg_max_smoke = [], []
        hardest = []
        for im in art["images"]:
            cat = cat_of(im["id"])
            fs = [p[5] for p in im["predictions"] if int(p[0]) == 0]
            ss = [p[5] for p in im["predictions"] if int(p[0]) == 1]
            mf = max(fs) if fs else 0.0
            ms = max(ss) if ss else 0.0
            has_fire = mf >= t_fire
            has_smoke = ms >= t_smoke
            truth = im["truth"]
            tc = {int(t[0]) for t in truth}
            if cat == NEG_CAT or not truth:
                if has_fire and has_smoke:
                    neg["both"] += 1
                elif has_fire:
                    neg["fire_only"] += 1
                elif has_smoke:
                    neg["smoke_only"] += 1
                else:
                    neg["none"] += 1
                neg["total"] += 1
                neg_max_fire.append(mf)
                neg_max_smoke.append(ms)
                W, H = size_of(resolve_image(a.img_root, im["id"]))
                b = bucket(max(W, H)) if W > 0 else "(unknown)"
                neg_res[b][0] += 1
                if has_fire or has_smoke:
                    neg_res[b][1] += 1
                if has_fire or has_smoke:
                    hardest.append((max(mf, ms), im["id"], round(mf, 3), round(ms, 3)))
            else:
                pos[cat]["total"] += 1
                if 0 in tc:
                    pos[cat]["has_fire"] += 1
                    if has_fire:
                        pos[cat]["fire_hit"] += 1
                if 1 in tc:
                    pos[cat]["has_smoke"] += 1
                    if has_smoke:
                        pos[cat]["smoke_hit"] += 1
        hardest.sort(reverse=True)
        report[name] = {
            "neg": dict(neg),
            "neg_res": {k: v for k, v in neg_res.items()},
            "pos": {k: dict(v) for k, v in pos.items()},
            "neg_score_fire": {"mean": st.fmean(neg_max_fire) if neg_max_fire else None,
                               "p90": sorted(neg_max_fire)[int(len(neg_max_fire) * 0.9)] if neg_max_fire else None,
                               "p99": sorted(neg_max_fire)[int(len(neg_max_fire) * 0.99)] if neg_max_fire else None,
                               "max": max(neg_max_fire) if neg_max_fire else None},
            "neg_score_smoke": {"mean": st.fmean(neg_max_smoke) if neg_max_smoke else None,
                                "p90": sorted(neg_max_smoke)[int(len(neg_max_smoke) * 0.9)] if neg_max_smoke else None,
                                "p99": sorted(neg_max_smoke)[int(len(neg_max_smoke) * 0.99)] if neg_max_smoke else None,
                                "max": max(neg_max_smoke) if neg_max_smoke else None},
            "hardest": hardest[:12],
        }

    L = ["# FASDD 误报归因（目标域，源域冻结阈值）\n",
         f"- 冻结阈值 fire={t_fire:.4f}, smoke={t_smoke:.4f}（来自源域 D-Fire val 的 1% FPR）",
         f"- 背景集 = 文件名前缀 `{NEG_CAT}` 的图像（全部无 GT）\n"]

    L.append("## 1. 背景集告警分解\n")
    L.append("| 配置 | 背景图数 | 仅 fire 触发 | 仅 smoke 触发 | 两类都触发 | **任意触发** | 未触发 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, r in report.items():
        n = r["neg"]
        tot = n.get("total", 1)
        L.append(f"| {name} | {tot} | {n.get('fire_only',0)} ({n.get('fire_only',0)/tot*100:.1f}%) | "
                 f"{n.get('smoke_only',0)} ({n.get('smoke_only',0)/tot*100:.1f}%) | "
                 f"{n.get('both',0)} ({n.get('both',0)/tot*100:.1f}%) | "
                 f"**{(n.get('fire_only',0)+n.get('smoke_only',0)+n.get('both',0))} "
                 f"({(n.get('fire_only',0)+n.get('smoke_only',0)+n.get('both',0))/tot*100:.2f}%)** | "
                 f"{n.get('none',0)} ({n.get('none',0)/tot*100:.1f}%) |")
    L.append("")

    L.append("## 2. 背景集触发率按分辨率\n")
    L.append("| 配置 | 分辨率档 | 图数 | 触发数 | 触发率 |")
    L.append("|---|---|---:|---:|---:|")
    for name, r in report.items():
        for b in ("<=480", "481-720", "721-1080", "1081-1440", "1441-1920", ">1920"):
            v = r["neg_res"].get(b)
            if not v or v[0] == 0:
                continue
            L.append(f"| {name} | {b} | {v[0]} | {v[1]} | **{v[1]/v[0]*100:.1f}%** |")
    L.append("")

    L.append("## 3. 正样本召回按图像类别\n")
    L.append("| 配置 | 图类别 | 图数 | fire 召回 | smoke 召回 |")
    L.append("|---|---|---:|---:|---:|")
    for name, r in report.items():
        for cat in POSITIVE_CATS:
            v = r["pos"].get(cat)
            if not v:
                continue
            fr = f"{v['fire_hit']/v['has_fire']*100:.1f}%" if v.get("has_fire") else "—"
            sr = f"{v['smoke_hit']/v['has_smoke']*100:.1f}%" if v.get("has_smoke") else "—"
            L.append(f"| {name} | {cat} | {v.get('total',0)} | {fr} | {sr} |")
    L.append("")

    L.append("## 4. 背景集最高分的图像（最难负样本）\n")
    L.append("| 配置 | max(fire,smoke) | 文件名 | fire 最高分 | smoke 最高分 |")
    L.append("|---|---:|---|---:|---:|")
    for name, r in report.items():
        for score, iid, mf, ms in r["hardest"][:8]:
            L.append(f"| {name} | {score:.4f} | `{iid.split('/')[-1]}` | {mf:.3f} | {ms:.3f} |")
    L.append("")

    L.append("## 5. 背景集分数分布\n")
    L.append("| 配置 | 类 | 均值 | p90 | p99 | 最大 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for name, r in report.items():
        for c, key in (("fire", "neg_score_fire"), ("smoke", "neg_score_smoke")):
            s = r[key]
            if s["mean"] is None:
                continue
            L.append(f"| {name} | {c} | {s['mean']:.4f} | {s['p90']:.4f} | {s['p99']:.4f} | {s['max']:.4f} |")
    L.append("")

    L.append("## 6. 读法\n")
    L.append("- 看第 1 节的**触发类别构成**：若 smoke 侧占主导，说明误报主要来自「把非烟当烟」；")
    L.append("  若 fire 侧主导，则是「把暖色/高亮当火」。这决定 hard-negative 该采哪一类。")
    L.append("- 看第 2 节：若触发率随分辨率单调变化，说明是尺度/细节问题；若各档相近，")
    L.append("  说明是**语义混淆**（与分辨率无关），那才是真正需要负样本的地方。")
    L.append("- 看第 3 节：`smoke` 单独那一类的召回，能区分「烟本身难」与「火烟同框时才难」。")
    L.append("- 第 4 节给出具体文件名，可以直接打开看是什么东西被误报了。\n")
    L.append("> 本报告不含任何新训练或新推理；全部来自已有 predictions。")

    (out / "false_alarm_attribution.md").write_text("\n".join(L), encoding="utf-8")
    (out / "false_alarm_attribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
