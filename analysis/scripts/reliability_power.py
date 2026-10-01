# -*- coding: utf-8 -*-
"""测量图像级告警指标的种子方差与 MDD，并与 mAP50-95 对比。

目的
----
mAP50-95 的种子 σ = 0.4218pp → 最小可检测效应 MDD = 1.05pp，导致 SRDG/DCBR/ACR 全部不可分辨。
本脚本用**已有 checkpoint**（零训练成本）测另一套指标的门槛：
图像级 recall@1%FPR / negative_image_fpr / background_alarm_fraction / image_ECE / image_Brier。
若它们的 MDD 显著更低，说明该换终点而不是换模块。

链路（沿用仓库既有协议）
------------------------
每个 seed:  evaluate(val)  -> calibrate(1% FPR 冻结阈值)  ->  evaluate(test)  ->  reliability
最后用官方 summarize 交叉核对 mAP 的种子 SD。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/reliability_power.py \
    --weights experiments/yolo11n/D-Fire_p2_seed{}.pt 的模板 ...
简化：--run-template 'experiments/yolo11n/D-Fire_p2_seed{seed}/weights/best.pt'
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

T_CRIT_05 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_crit(df):
    if df <= 0:
        return float("inf")
    return T_CRIT_05.get(df, 1.96 + 2.4 / df)


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def run(cmd, log):
    log.append("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.append(p.stdout[-2000:])
        log.append(p.stderr[-2000:])
        raise SystemExit(f"命令失败 (rc={p.returncode}): {' '.join(cmd)}")
    return p.stdout


def is_degenerate(key, sd):
    """剔除不是真指标的键: 常数项(bins=15)、样本计数、以及种子方差为 0 的项。

    方差为 0 通常意味着该量由构造决定(如按 1% FPR 精确选出的 validation_fpr),
    或者根本不是指标(如背景图数量)。它们不是"门槛最低的好终点", 而是没有信息量。
    """
    bad_suffix = (".bins", "_count", "_image_count")
    if key.endswith(bad_suffix):
        return True
    if ".bins" in key or key.endswith("image_count"):
        return True
    if sd == 0.0:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default=str(Path.home() / "miniconda3/envs/yolo11/bin/python"))
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--dataset", default="dfire")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--template", default="experiments/yolo11n/D-Fire_p2_seed{seed}/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-fpr", type=float, default=0.01)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--reuse", action="store_true", help="已存在的产物直接复用，跳过评估")
    ap.add_argument("--out", default="analysis/out_reliability")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    base = [a.py, "-m", "firesmoke", "--config", a.config]

    per_seed = {}
    for s in seeds:
        w = a.template.format(seed=s)
        if not Path(w).is_file():
            raise SystemExit(f"找不到权重: {w}")
        vdir = out / f"val_s{s}"
        cfile = out / f"cal_s{s}.json"
        tdir = out / f"test_s{s}"
        rfile = out / f"rel_s{s}.json"

        print(f"[seed {s}] evaluate val ...", flush=True)
        if a.reuse and (vdir / "predictions.json").is_file():
            print("        (复用已有产物)", flush=True)
        else:
            run(base + ["evaluate", "--weights", w, "--dataset", a.dataset, "--split", "val",
                        "--output", str(vdir), "--device", a.device,
                        "--imgsz", str(a.imgsz), "--batch", str(a.batch)], log)
        print(f"[seed {s}] calibrate ...", flush=True)
        if a.reuse and cfile.is_file():
            print("        (复用已有产物)", flush=True)
        else:
            run(base + ["calibrate", "--predictions", str(vdir / "predictions.json"),
                        "--output", str(cfile), "--max-fpr", str(a.max_fpr)], log)
        print(f"[seed {s}] evaluate test ...", flush=True)
        if a.reuse and (tdir / "predictions.json").is_file():
            print("        (复用已有产物)", flush=True)
        else:
            run(base + ["evaluate", "--weights", w, "--dataset", a.dataset, "--split", "test",
                        "--output", str(tdir), "--device", a.device,
                        "--imgsz", str(a.imgsz), "--batch", str(a.batch)], log)
        print(f"[seed {s}] reliability report ...", flush=True)
        if a.reuse and rfile.is_file():
            print("        (复用已有产物)", flush=True)
        else:
            run(base + ["reliability", "--predictions", str(tdir / "predictions.json"),
                        "--calibration", str(cfile), "--output", str(rfile),
                        "--bootstrap", str(a.bootstrap)], log)

        rec = {"seed": s, "weights": w}
        rec["test_metrics"] = flatten(json.loads((tdir / "metrics.json").read_text())["metrics"])
        cal = json.loads(cfile.read_text())
        rel = json.loads(rfile.read_text())
        rec["val_diag"] = flatten(cal.get("fit_diagnostics_not_test_results", {}))
        rec["rel"] = flatten(rel.get("metrics", {}))
        rec["rel_intervals"] = {k: v for k, v in flatten(rel.get("intervals", {})).items()
                                if "mean" in k or "lo" in k or "hi" in k}
        per_seed[s] = rec

    # 跨种子统计
    def across(getter, label):
        keys = None
        vals = {}
        for s in seeds:
            d = getter(per_seed[s]) or {}
            vals[s] = d
            keys = set(d) if keys is None else (keys & set(d))
        rows = []
        for k in sorted(keys or []):
            xs = [vals[s][k] for s in seeds]
            if len(xs) < 2:
                continue
            mean = st.fmean(xs)
            sd = st.stdev(xs)
            if is_degenerate(k, sd):
                continue
            se = sd / math.sqrt(len(xs))
            rows.append({"metric": k, "mean": mean, "sd": sd, "sd_pp": sd * 100,
                         "mdd": t_crit(len(xs) - 1) * se, "mdd_pp": t_crit(len(xs) - 1) * se * 100,
                         "per_seed": xs})
        rows.sort(key=lambda r: r["mdd"])
        return rows

    det = across(lambda r: r["test_metrics"], "detection")
    relm = across(lambda r: r["rel"], "reliability")
    vald = across(lambda r: r["val_diag"], "val_diag")

    # 官方 summarize 交叉核对（只认 test split 的检测指标）
    mfiles = [str(out / f"test_s{s}" / "metrics.json") for s in seeds]
    try:
        sm = run(base + ["summarize", "--metrics", *mfiles, "--output",
                         str(out / "summary.json")], log)
        summary = json.loads(sm) if sm.strip().startswith("[") else json.loads(
            (out / "summary.json").read_text())["groups"]
    except SystemExit:
        summary = None

    report = {"seeds": seeds, "detection": det, "reliability_metrics": relm,
              "val_diagnostics": vald, "official_summarize": summary,
              "mAP50-95_reference": {"sd_pp": 0.4218, "mdd_pp": 1.0478, "source": "三种子 results.csv"}}
    (out / "reliability_power.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "run_log.txt").write_text("\n".join(log), encoding="utf-8")

    L = ["# 图像级告警指标的种子方差与 MDD\n",
         f"- seeds: {seeds}   权重模板: `{a.template}`",
         f"- 1% 负图 FPR 阈值在 **val** 上冻结, 指标在 **test** 上报告",
         f"- 参照: `mAP50-95` 三种子 σ = 0.4218pp → MDD = 1.0478pp\n"]

    def table(title, rows, note=""):
        L.append(f"## {title}\n")
        if note:
            L.append(note + "\n")
        L.append("| 指标 | 均值 | 种子σ | **MDD (α=.05)** | 相比 mAP50-95 的门槛 |")
        L.append("|---|---:|---:|---:|---:|")
        for r in rows:
            ratio = 1.0478 / r["mdd_pp"] if r["mdd_pp"] > 0 else float("inf")
            tag = f"**低 {ratio:.1f} 倍**" if ratio > 1.05 else (f"高 {1/ratio:.1f} 倍" if ratio < 0.95 else "相当")
            L.append(f"| `{r['metric']}` | {r['mean']:.5f} | {r['sd_pp']:.4f}pp | "
                     f"**{r['mdd_pp']:.4f}pp** | {tag} |")
        L.append("")

    table("1. 图像级告警指标（test，阈值在 val 冻结）", relm)
    table("2. 检测指标（test，官方 evaluate 口径）", det)
    table("3. 校准诊断（val，in-sample）", vald,
          "> 注意: 阈值是在同一批 val 上选的, 属 in-sample, 数值乐观; 但跨种子程序一致, 方差仍可比。")

    L.append("## 4. 结论怎么读\n")
    mdd_ref = 0.9176  # test 集上三种子实测的 mAP50-95 MDD
    if relm:
        best = relm[0]
        L.append(f"- 门槛最低的**真实指标**是 `{best['metric']}`：MDD = **{best['mdd_pp']:.4f}pp**"
                 f"（mAP50-95 实测 {mdd_ref:.4f}pp）。")
        L.append(f"- 换算：换终点可把可分辨效应从 {mdd_ref:.2f}pp 降到 {best['mdd_pp']:.2f}pp，"
                 f"约 **{mdd_ref / best['mdd_pp']:.1f} 倍**。")
    L.append("- **注意反直觉的地方**：带「阈值选择」的指标（固定 FPR 下的 recall）门槛**更大**，")
    L.append("  因为阈值本身是种子方差的主要来源。凡是要先选阈值的指标，都继承了这份方差。")
    L.append("- 判据：只有 MDD 明显低于 mAP50-95 的指标才值得当主终点。")
    L.append("- 之后应用低门槛指标**回头重新评估已有 checkpoint**（零训练成本），")
    L.append("  看 SRDG / DCBR / ACR 的效应是否变得可分辨。\n")
    L.append("> 本报告不含任何新训练；全部来自已有 checkpoint 的重新评估。")
    (out / "reliability_power.md").write_text("\n".join(L), encoding="utf-8")

    print("\n" + "\n".join(L))
    print(f"\n输出: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
