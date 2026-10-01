# -*- coding: utf-8 -*-
"""用低门槛终点重新评估 SRDG / DCBR / ACR 相对 P2 的效应。

背景
----
mAP50-95 的种子门槛是 0.92pp，而三个模块相对父级 P2 的效应只有 −0.29 ~ +0.47pp，
全部落在噪声里。上一轮测出两根救命稻草：
  1. 把 FPR 阈值**冻结成共享值**后，fire.recall 的门槛从 3.18pp 降到 0.34~0.79pp
  2. 图像级校准指标（fire 的 ECE / Brier）门槛只有 ~0.11pp
本脚本用这两类终点重算三个模块的配对效应，看它们是否变得可分辨。

链路：evaluate(val) -> calibrate(1% FPR) -> evaluate(test) -> reliability
阈值一律作用在**原始分数**上，并统一取自 **P2 seed0 的 val**，保证跨配置可比。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/module_effects.py --out analysis/out_modules \
    --reuse-p2 analysis/out_reliability
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

T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}

# 配置名 -> (权重模板, seeds, 依赖的父级)
CONFIGS = [
    ("p2", "experiments/yolo11n/D-Fire_p2_seed{seed}/weights/best.pt", [0, 1, 2], None),
    ("p2_srdg", "experiments/yolo11n/D-Fire_p2_srdg_seed{seed}/weights/best.pt", [0], "p2"),
    ("p2_dcbr", "experiments/yolo11n/D-Fire_p2_dcbr_seed{seed}_module-dcbr-v1/weights/best.pt", [0, 2], "p2"),
    ("p2_acr", "experiments/yolo11n/D-Fire_p2_acr_seed{seed}_module-acr-v1/weights/best.pt", [0], "p2"),
]


def run(cmd, log):
    log.append("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.append(p.stdout[-1500:] + "\n" + p.stderr[-1500:])
        raise SystemExit("失败: " + " ".join(cmd))
    return p.stdout


def art(p: Path):
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default=str(Path.home() / "miniconda3/envs/yolo11/bin/python"))
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--dataset", default="dfire")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-fpr", type=float, default=0.01)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="analysis/out_modules")
    ap.add_argument("--reuse-p2", default=None, help="P2 已跑完的产物目录，可复用")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = {"p2": Path(a.reuse_p2)} if a.reuse_p2 else {}
    log = []
    base = [a.py, "-m", "firesmoke", "--config", a.config]

    def paths(cfg, s, root):
        """root 非空 = 复用已有产物目录。

        复用目录（reliability_power.py 的产出）用 `val_s0` / `cal_s0.json` 这种**无配置前缀**的命名；
        新跑的产物则带配置前缀 (`val_{cfg}_s0`)，避免不同配置互相覆盖。
        """
        r = Path(root) if root else out
        pre = "" if root else f"{cfg}_"
        return (r / f"val_{pre}s{s}", r / f"cal_{pre}s{s}.json",
                r / f"test_{pre}s{s}", r / f"rel_{pre}s{s}.json")

    for cfg, tmpl, seeds, _par in CONFIGS:
        root = cache.get(cfg)
        for s in seeds:
            w = tmpl.format(seed=s)
            if not Path(w).is_file():
                print(f"[skip] 缺权重 {cfg}/seed{s}: {w}")
                continue
            vdir, cfile, tdir, rfile = paths(cfg, s, root)
            need = []
            if not (vdir / "predictions.json").is_file():
                need.append(("val", vdir))
            if not cfile.is_file():
                need.append(("cal", cfile))
            if not (tdir / "predictions.json").is_file():
                need.append(("test", tdir))
            if not rfile.is_file():
                need.append(("rel", rfile))
            if need and a.reuse and root is not None:
                raise SystemExit(f"复用目录缺少 {cfg}/seed{s} 的产物")
            if need:
                print(f"[{cfg}/seed{s}] 需运行: {[n[0] for n in need]}", flush=True)
                if not (vdir / "predictions.json").is_file():
                    run(base + ["evaluate", "--weights", w, "--dataset", a.dataset, "--split", "val",
                                "--output", str(vdir), "--device", a.device,
                                "--imgsz", str(a.imgsz), "--batch", str(a.batch)], log)
                if not cfile.is_file():
                    run(base + ["calibrate", "--predictions", str(vdir / "predictions.json"),
                                "--output", str(cfile), "--max-fpr", str(a.max_fpr)], log)
                if not (tdir / "predictions.json").is_file():
                    run(base + ["evaluate", "--weights", w, "--dataset", a.dataset, "--split", "test",
                                "--output", str(tdir), "--device", a.device,
                                "--imgsz", str(a.imgsz), "--batch", str(a.batch)], log)
                if not rfile.is_file():
                    run(base + ["reliability", "--predictions", str(tdir / "predictions.json"),
                                "--calibration", str(cfile), "--output", str(rfile),
                                "--bootstrap", str(a.bootstrap)], log)
            else:
                print(f"[{cfg}/seed{s}] 复用已有产物", flush=True)

    # ---------------- 汇总 ----------------
    # 共享阈值：取自 p2 seed0 的 val（原始分数，1% FPR）
    vdir0, _, _, _ = paths("p2", 0, cache.get("p2"))
    sv0, lv0 = arrays(art(vdir0 / "predictions.json"))
    shared = [select_threshold(sv0[:, c], lv0[:, c], a.max_fpr)["threshold"] for c in (0, 1)]
    print(f"\n共享阈值(来自 p2/seed0 的 val): fire={shared[0]:.4f}  smoke={shared[1]:.4f}")

    per = {}
    for cfg, tmpl, seeds, _par in CONFIGS:
        root = cache.get(cfg)
        for s in seeds:
            vdir, cfile, tdir, rfile = paths(cfg, s, root)
            if not (tdir / "predictions.json").is_file():
                continue
            stt, ltt = arrays(art(tdir / "predictions.json"))
            al = alarm_metrics(stt, ltt, shared)
            rel = art(rfile)
            met = art(tdir / "metrics.json")["metrics"]
            rec = {"p2_map50_95": met.get("metrics/mAP50-95(B)"),
                   "p2_map50": met.get("metrics/mAP50(B)"),
                   "fixed_fire_recall": al["fire"]["recall"],
                   "fixed_smoke_recall": al["smoke"]["recall"],
                   "fixed_bg_alarm": al["background_alarm_fraction"],
                   "fixed_fire_fpr": al["fire"]["negative_image_fpr"]}
            for cls in ("fire", "smoke"):
                for phase in ("before", "after"):
                    cm = rel.get("metrics", {}).get("calibration", {}).get(cls, {}).get(phase, {})
                    rec[f"ece_{cls}_{phase}"] = cm.get("image_ECE")
                    rec[f"brier_{cls}_{phase}"] = cm.get("image_Brier")
            per.setdefault(cfg, {})[s] = rec

    ENDPOINTS = ["p2_map50_95", "fixed_fire_recall", "fixed_smoke_recall", "fixed_bg_alarm",
                 "fixed_fire_fpr", "ece_fire_after", "brier_fire_after",
                 "ece_smoke_after", "brier_smoke_after"]
    REF = {"p2_map50_95": 0.9176, "fixed_fire_recall": 0.3428, "fixed_bg_alarm": 0.5678,
           "ece_fire_after": 0.1089, "brier_fire_after": 0.1146}

    rows = []
    for cfg, _, _, par in CONFIGS:
        if par is None or cfg not in per or par not in per:
            continue
        common = sorted(set(per[cfg]) & set(per[par]))
        if not common:
            continue
        for ep in ENDPOINTS:
            d = [per[cfg][s][ep] - per[par][s][ep] for s in common
                 if per[cfg][s].get(ep) is not None and per[par][s].get(ep) is not None]
            if not d:
                continue
            mean = st.fmean(d)
            n = len(d)
            sd = st.stdev(d) if n > 1 else None
            mdd = (T_CRIT.get(n - 1, 2.0) * sd / math.sqrt(n)) if sd else None
            rows.append({"cfg": cfg, "vs": par, "endpoint": ep, "seeds": common,
                         "effect_pp": mean * 100,
                         "sd_d_pp": sd * 100 if sd else None,
                         "mdd_pp": mdd * 100 if mdd else None,
                         "ref_mdd_pp": REF.get(ep),
                         "resolvable": (abs(mean) > mdd) if mdd else None})

    L = ["# 三模块在低门槛终点下的配对效应\n",
         f"- 数据源：`{a.out}`" + (f" + P2 复用自 `{a.reuse_p2}`" if a.reuse_p2 else ""),
         f"- 共享 FPR 阈值取自 p2/seed0 的 val：fire={shared[0]:.4f}, smoke={shared[1]:.4f}",
         "- 效应 = 该模块 − 父级 P2（同 seed 配对）；负值表示比 P2 差\n",
         "| 模块 | 终点 | seeds | 效应 | 配对σ | MDD | 参照门槛 | 判定 |",
         "|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if r["resolvable"] is None:
            verdict = "—"
        elif r["resolvable"]:
            verdict = "**可分辨**"
        else:
            verdict = "不可分辨"
        sd_txt = f"{r['sd_d_pp']:.3f}pp" if r["sd_d_pp"] else "—"
        mdd_txt = f"{r['mdd_pp']:.3f}pp" if r["mdd_pp"] else "—"
        ref_txt = f"{r['ref_mdd_pp']:.3f}pp" if r["ref_mdd_pp"] else "—"
        L.append(f"| {r['cfg']} | `{r['endpoint']}` | {len(r['seeds'])} | {r['effect_pp']:+.3f}pp | "
                 f"{sd_txt} | {mdd_txt} | {ref_txt} | {verdict} |")
    L.append("")
    L.append("## 读法\n")
    L.append("- **配对 n=1**（srdg / acr）只有点估计，没有 σ，无法判定 —— 这是样本量问题，不是方法问题。")
    L.append("- **配对 n=2**（dcbr）可以给出 σ_d 和 MDD，但 df=1 的 t 临界值高达 12.706，门槛很宽。")
    L.append("- 参照门槛是该终点在**三种子 P2 上**实测的 MDD，用来判断“换终点值不值”。")
    L.append("- 真正要看的：同一个模块的效应，在不同终点下的**符号与量级是否一致**。")
    L.append("  若在低门槛终点上符号翻转或量级乱跳，说明它本来就在噪声里。")
    (out / "module_effects.md").write_text("\n".join(L), encoding="utf-8")
    (out / "module_effects.json").write_text(
        json.dumps({"shared_thresholds": shared, "rows": rows, "per_config": per},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "module_effects_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
