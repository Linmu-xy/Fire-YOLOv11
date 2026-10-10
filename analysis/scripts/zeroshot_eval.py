# -*- coding: utf-8 -*-
"""零样本跨域评估：D-Fire 训练的模型在 FASDD_CV test 上表现如何。

测两件事
--------
1. **检测泛化**：目标域检测指标（mAP50 / mAP50-95 / 分类型 AP）。
2. **告警泛化（H2）**：把 **D-Fire val 上冻结的 1% FPR 阈值**直接用到 FASDD test 的
   图像级分数上，看误报率与召回能保住多少 —— 这正是"在未见过的数据来源上降误报"的检验。
   注意协议规定阈值只能在**源域** val 上选（`reliability.fit` 会拒绝 target 校准），
   所以这里严格用 D-Fire val 的阈值，不做任何 target 侧调整。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/zeroshot_eval.py --out analysis/out_zeroshot
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from firesmoke.reliability import alarm_metrics, arrays, select_threshold  # noqa: E402

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

# (配置名, 权重路径, seed)
# 2026-10-02 更新：旧的 outputs/ 与 experiments/ 已并入 results/（见 PROJECT_LAYOUT 第 3 节），
# 原路径全部失效、脚本会静默 skip 到空结果。这里改为 results/ 下的真实路径，并补上
# baseline640 的 seed1/2（此前只有 seed0，导致对照组没有跨域种子 σ）与 Arm B（dfire_fsneg3）。
# 已发布的历史报告 analysis/reports/out_zeroshot__zeroshot_eval.md 由旧路径版本产出，口径未变。
CONFIGS = [
    ("baseline640", "results/dfire/paper-v1/baseline/seed0/weights/best.pt", 0),
    ("baseline640", "results/dfire/paper-v1/baseline/seed1/weights/best.pt", 1),
    ("baseline640", "results/dfire/paper-v1/baseline/seed2/weights/best.pt", 2),
    ("p2", "results/dfire/paper-v1/p2/seed0/weights/best.pt", 0),
    ("p2", "results/dfire/paper-v1/p2/seed1/weights/best.pt", 1),
    ("p2", "results/dfire/paper-v1/p2/seed2/weights/best.pt", 2),
    ("p2_srdg", "results/dfire/module-srdg-v1/p2_srdg/seed0/weights/best.pt", 0),
    ("p2_dcbr", "results/dfire/module-dcbr-v1/p2_dcbr/seed0/weights/best.pt", 0),
    ("p2_dcbr", "results/dfire/module-dcbr-v1/p2_dcbr/seed2/weights/best.pt", 2),
    ("p2_acr", "results/dfire/module-acr-v1/p2_acr/seed0/weights/best.pt", 0),
    # Arm B（下一轮）：训练脚本产出后自动被读到；没训练时会被 skip，不影响历史配置
    ("armb", "results/dfire_fsneg3/armb-v1/baseline/seed0/weights/best.pt", 0),
    ("armb", "results/dfire_fsneg3/armb-v1/baseline/seed1/weights/best.pt", 1),
    ("armb", "results/dfire_fsneg3/armb-v1/baseline/seed2/weights/best.pt", 2),
]


def run(cmd, log):
    log.append("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.append(p.stdout[-1500:] + "\n" + p.stderr[-1500:])
        raise SystemExit("失败: " + " ".join(cmd))
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default=str(Path.home() / "miniconda3/envs/yolo11/bin/python"))
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--src", default="dfire")
    ap.add_argument("--tgt", default="fasdd")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-fpr", type=float, default=0.01)
    ap.add_argument("--out", default="analysis/out_zeroshot")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--threshold-from", default="p2_s0",
                    help="冻结阈值的来源，格式 <配置名>_s<seed>；默认沿用 out_zeroshot 报告口径")
    ap.add_argument("--variance-groups", default="p2",
                    help="计算目标域种子方差的配置组，逗号分隔（例如 p2,armb,baseline640）")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    base = [a.py, "-m", "firesmoke", "--config", a.config]

    def ev(weights, ds, split, dest):
        if a.reuse and (dest / "predictions.json").is_file() and (dest / "metrics.json").is_file():
            print(f"  [reuse] {dest.name}", flush=True)
            return
        run(base + ["evaluate", "--weights", weights, "--dataset", ds, "--split", split,
                    "--output", str(dest), "--device", a.device,
                    "--imgsz", str(a.imgsz), "--batch", str(a.batch)], log)

    results = {}
    for cfg, w, s in CONFIGS:
        if not Path(w).is_file():
            print(f"[skip] 缺权重 {cfg}/s{s}: {w}")
            continue
        print(f"[{cfg}/s{s}] 源域 val ...", flush=True)
        svdir = out / f"src_{cfg}_s{s}"
        ev(w, a.src, "val", svdir)
        print(f"[{cfg}/s{s}] 目标域 {a.tgt} test ...", flush=True)
        tdir = out / f"tgt_{cfg}_s{s}"
        ev(w, a.tgt, "test", tdir)
        tm = json.loads((tdir / "metrics.json").read_text())
        results[(cfg, s)] = {
            "metrics": tm.get("metrics", {}),
            "per_class": tm.get("per_class", {}),
            "images": tm.get("image_count"),
            "src_val": str(svdir / "predictions.json"),
            "tgt_test": str(tdir / "predictions.json"),
        }

    # 冻结阈值：所有配置共用同一个工作点，且只能在**源域 val** 上选（协议要求，不做目标域调整）。
    # 默认沿用 out_zeroshot 报告的口径（p2/seed0 的源域 val，fire=0.3143/smoke=0.6057），
    # 这样新结果能与已发布的基线数字直接比较；换别的来源必须整表重算。
    group0, seed0 = a.threshold_from.rsplit("_s", 1)
    key0 = (group0, int(seed0))
    if key0 not in results:
        raise SystemExit(f"阈值来源 {a.threshold_from} 不在本次结果里；可用："
                         f"{sorted(f'{c}_s{s}' for c, s in results)}")
    sv, lv = arrays(json.loads(Path(results[key0]["src_val"]).read_text()))
    shared = [select_threshold(sv[:, c], lv[:, c], a.max_fpr)["threshold"] for c in (0, 1)]
    print(f"\n冻结阈值(来自 {key0} 的 {a.src} val): fire={shared[0]:.4f} smoke={shared[1]:.4f}")

    for k, r in results.items():
        stt, ltt = arrays(json.loads(Path(r["tgt_test"]).read_text()))
        r["alarm"] = alarm_metrics(stt, ltt, shared)
        # 源域 val 上的同一组告警指标（用于对照：域内 vs 跨域）
        r["alarm_src"] = alarm_metrics(sv, lv, shared)

    L = ["# 零样本跨域：D-Fire → FASDD_CV test\n",
         f"- 源域 {a.src} val 上冻结 1% FPR 阈值（fire={shared[0]:.4f}, smoke={shared[1]:.4f}），"
         f"不做任何目标域调整",
         f"- 目标域测试图像数：{next(iter(results.values()))['images']}",
         f"- imgsz {a.imgsz}，与训练一致；结论只反映域差，不反映分辨率差异\n",
         "## 1. 检测指标（目标域 test）\n",
         "| 配置 | seed | mAP50 | mAP50-95 | fire AP50-95 | smoke AP50-95 |",
         "|---|---:|---:|---:|---:|---:|"]
    for (cfg, s), r in sorted(results.items()):
        m = r["metrics"]
        pc = r["per_class"]
        f = pc.get("fire", {}).get("AP50_95")
        sm = pc.get("smoke", {}).get("AP50_95")
        f_txt = f"{f:.4f}" if f is not None else "—"
        sm_txt = f"{sm:.4f}" if sm is not None else "—"
        L.append(f"| {cfg} | {s} | {m.get('metrics/mAP50(B)', float('nan')):.4f} | "
                 f"**{m.get('metrics/mAP50-95(B)', float('nan')):.4f}** | {f_txt} | {sm_txt} |")
    L.append("")

    L.append("## 2. 告警指标（冻结的源域阈值，直接用于目标域）\n")
    L.append("| 配置 | seed | fire 召回 (域内→跨域) | smoke 召回 | 背景告警比例 | fire 负图 FPR |")
    L.append("|---|---:|---|---:|---:|---:|")
    for (cfg, s), r in sorted(results.items()):
        al, asrc = r["alarm"], r["alarm_src"]
        fr, sr = al["fire"]["recall"], al["smoke"]["recall"]
        bg, fpr = al["background_alarm_fraction"], al["fire"]["negative_image_fpr"]
        L.append(f"| {cfg} | {s} | {asrc['fire']['recall']:.4f} → **{fr:.4f}** | "
                 f"{asrc['smoke']['recall']:.4f} → **{sr:.4f}** | "
                 f"{asrc['background_alarm_fraction']:.4f} → **{bg:.4f}** | "
                 f"{asrc['fire']['negative_image_fpr']:.4f} → {fpr:.4f} |")
    L.append("")
    L.append("> 箭头左边是**源域 val**（阈值就在这里选的，属 in-sample），右边是**目标域 test**（跨域）。")
    L.append("")

    # 各配置组在目标域上的种子方差（这就是判断"改善是否可分辨"的门槛）
    for group in [g for g in a.variance_groups.split(",") if g]:
        keys = sorted(k for k in results if k[0] == group)
        if len(keys) < 2:
            continue
        L.append(f"## 3. {group} 在目标域上的种子方差\n")
        L.append("| 指标 | 均值 | 种子σ | **MDD (α=.05)** |")
        L.append("|---|---:|---:|---:|")
        for name, get in (("mAP50-95", lambda r: r["metrics"].get("metrics/mAP50-95(B)")),
                          ("mAP50", lambda r: r["metrics"].get("metrics/mAP50(B)")),
                          ("fire 召回", lambda r: r["alarm"]["fire"]["recall"]),
                          ("背景告警比例", lambda r: r["alarm"]["background_alarm_fraction"])):
            xs = [get(results[k]) for k in keys]
            xs = [x for x in xs if x is not None]
            if len(xs) < 2:
                continue
            mean, sd = st.fmean(xs), st.stdev(xs)
            mdd = T_CRIT.get(len(xs) - 1, 2.0) * sd / math.sqrt(len(xs))
            L.append(f"| {name} | {mean:.4f} | {sd*100:.4f}pp | **{mdd*100:.4f}pp** |")
        L.append("")
        L.append("> 这是**目标域上的**种子门槛。与域内比较时要注意两者不是同一个量。\n")

    L.append("## 4. 读法\n")
    L.append("- 域内→跨域的落差本身是预期内的（D-Fire 与 FASDD_CV 的来源、分辨率、标注标准都不同）。")
    L.append("- 关键不是绝对值，而是**各配置之间的排序是否稳定**：若某改动只在域内赢、跨域输，")
    L.append("  说明它拟合的是源域特性。")
    L.append("- 告警表里要看的是：**冻结阈值搬到目标域后，背景告警比例涨了多少** ——")
    L.append("  这就是「在未见来源上降误报」的直接度量。")
    L.append("")
    L.append("> 本报告不含任何新训练；全部来自已有 checkpoint 的重新评估。")
    (out / "zeroshot_eval.md").write_text("\n".join(L), encoding="utf-8")
    (out / "zeroshot_eval.json").write_text(
        json.dumps({"thresholds": shared,
                    "results": {f"{k[0]}_s{k[1]}": v for k, v in results.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "zeroshot_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
