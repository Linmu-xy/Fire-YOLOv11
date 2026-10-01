#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metric_power_analysis.py
========================
从已有的 results.csv 里，回答一个问题:

    "我这套实验装置，到底能不能分辨出我想测的效应?"

不需要重训、不需要重新推理、不需要 ultralytics —— 只读你已经在写的 results.csv。

它算三件事:

1. **方差分解**: 把每个 run 的指标波动拆成两部分
   - 组内 (run 内 epoch 之间的波动) —— 反映"选哪个 epoch 当 best"的不确定性
   - 组间 (同一配置不同 seed 之间的波动) —— 反映初始化/优化轨迹的真实随机性
   如果组间远大于组内，说明**加种子是唯一能降方差的手段**，把 val 评估做得更精细没用。

2. **最小可检测效应 MDD**: 在 n 个种子、α=0.05 双尾的配对 t 检验下，
   要多大效应才能被判为显著。MDD = t_crit(df=n-1) * σ/√n。
   这就是"你这套装置的分辨率下限"。

3. **达标所需种子数**: 给定一个目标效应，反算需要多少 seed 才测得出。

用法
----
python metric_power_analysis.py --root experiments --out power_out
python metric_power_analysis.py --root experiments --reference D-Fire_protocol_baseline_800
python metric_power_analysis.py --root experiments --targets 0.1,0.3,0.5,1.0

只看标准库。目录名解析规则: 去掉 _seedN 与 _module-* 后作为配置名。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

# α=0.05 双尾 t 临界值表 (df=1..30)
T_CRIT_05 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

SEED_RE = re.compile(r"_seed(\d+)$")
SEED_DIR_RE = re.compile(r"^seed(\d+)$")
MODULE_RE = re.compile(r"_module-[^/]*$")
IMGSZ_RE = re.compile(r"^\s*imgsz:\s*(\d+)", re.M)


def t_crit(df: int, alpha: float = 0.05) -> float:
    """双尾 t 临界值。df<=30 用精确表；更大用 Cornish-Fisher 展开。"""
    if df <= 0:
        return float("inf")
    if alpha == 0.05 and df in T_CRIT_05:
        return T_CRIT_05[df]
    z = 1.959964  # 标准正态 0.975 分位
    if df <= 30:
        # 表内插值
        lo = max(1, df - 3)
        hi = min(30, df + 3)
        if hi > lo:
            f = (df - lo) / (hi - lo)
            return T_CRIT_05[lo] * (1 - f) + T_CRIT_05[hi] * f
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def parse_run_dir(results_path: Path, root: Path, strip_prefix: str) -> tuple[str, int | None, int | None]:
    """返回 (配置名, seed, imgsz)。

    同时支持两种实验目录布局:
      A) experiments/yolo11n/D-Fire_p2_seed0/results.csv   (seed 在目录名里)
      B) outputs/paper-v1/dfire/p2/seed0/results.csv       (seed 是独立目录)
    """
    run_dir = results_path.parent
    try:
        rel = run_dir.relative_to(root)
    except ValueError:
        rel = Path(run_dir.name)
    parts = [p for p in rel.parts]

    seed = None
    if parts:
        last = parts[-1]
        m = SEED_DIR_RE.fullmatch(last)
        if m:
            seed = int(m.group(1))
            parts = parts[:-1]
        else:
            m = SEED_RE.search(last)
            if m:
                seed = int(m.group(1))
                parts[-1] = SEED_RE.sub("", last)
    if not parts:
        parts = [run_dir.name]
    cfg = "/".join(parts)
    cfg = MODULE_RE.sub("", cfg)
    if strip_prefix:
        cfg = cfg.replace(strip_prefix, "")
    cfg = cfg.replace("/_", "/").replace("_/", "/")
    cfg = re.sub(r"_{2,}", "_", cfg)
    cfg = cfg.strip("/_") or run_dir.name

    imgsz = None
    args_yaml = run_dir / "args.yaml"
    if args_yaml.is_file():
        m = IMGSZ_RE.search(args_yaml.read_text(encoding="utf-8", errors="ignore"))
        if m:
            imgsz = int(m.group(1))
    return cfg, seed, imgsz


def required_n(sigma: float, target: float, alpha: float = 0.05, nmax: int = 5000) -> int | None:
    """要检测 target 大小的效应，配对 t 检验需要多少个独立种子。"""
    if sigma <= 0 or target <= 0:
        return None
    for n in range(2, nmax + 1):
        mdd = t_crit(n - 1, alpha) * sigma / math.sqrt(n)
        if mdd <= target:
            return n
    return None


def parse_dir_name(name: str, strip_prefix: str) -> tuple[str, int | None]:
    seed = None
    m = SEED_RE.search(name)
    if m:
        seed = int(m.group(1))
    cfg = MODULE_RE.sub("", name)
    cfg = SEED_RE.sub("", cfg)
    if strip_prefix and cfg.startswith(strip_prefix):
        cfg = cfg[len(strip_prefix):]
    cfg = cfg.strip("_") or name
    return cfg, seed


def read_results(path: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        cols = [c for c in (rd.fieldnames or []) if c.startswith("metrics/")]
        for r in rd:
            try:
                rec = {"epoch": int(float(r.get("epoch", 0)))}
            except Exception:
                continue
            for c in cols:
                try:
                    rec[c] = float(r[c])
                except Exception:
                    pass
            if len(rec) > 1:
                rows.append(rec)
    return rows, cols


def plateau_stats(rows: list[dict], col: str, face_metric: str, frac: float) -> dict:
    """给定 run 的曲线，返回 best / last / 尾部平台段的统计。"""
    series = [(r["epoch"], r[col]) for r in rows if col in r]
    if not series:
        return {}
    series.sort()
    best_ep, best_v = max(series, key=lambda t: t[1])
    n = len(series)
    k = max(3, int(round(n * frac)))
    tail = [v for _e, v in series[-k:]]
    tail_mean = st.fmean(tail)
    tail_sigma = st.stdev(tail) if len(tail) > 1 else 0.0
    # 尾部趋势: 用最小二乘斜率 * 尾部长度, 表示"还没收敛"带来的漂移量
    xs = list(range(len(tail)))
    xm = st.fmean(xs)
    ym = tail_mean
    denom = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, tail)) / denom if denom else 0.0
    return {
        "epochs": n, "best_epoch": best_ep, "best": best_v, "last": series[-1][1],
        "plateau_mean": tail_mean, "plateau_sigma": tail_sigma,
        "max_selection_gap": best_v - tail_mean,
        "tail_slope_per_epoch": slope,
        "tail_drift": abs(slope) * k,
    }


def fmt(v, nd=5, plus=False):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="从 results.csv 计算方差分解与最小可检测效应")
    ap.add_argument("--root", nargs="+", default=["experiments"],
                    help="一个或多个实验根目录, 空格分隔 (如: experiments outputs)")
    ap.add_argument("--glob", default="**/results.csv", help="results.csv 的匹配模式")
    ap.add_argument("--out", default="power_out")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--plateau-frac", type=float, default=0.3, help="尾部平台段占全长的比例")
    ap.add_argument("--strip-prefix", default="D-Fire", help="配置名要剥掉的数据集前缀")
    ap.add_argument("--metric", default="metrics/mAP50-95(B)", help="主指标列名")
    ap.add_argument("--reference", default=None, help="参照配置名; 有则计算各配置相对它的效应")
    ap.add_argument("--targets", default="0.1,0.3,0.5,1.0", help="要反算所需种子数的目标效应(pp)")
    args = ap.parse_args(argv)

    runs = []
    for r0 in args.root:
        root = Path(r0)
        files = sorted(root.glob(args.glob)) if root.is_dir() else []
        if not files:
            print(f"[warn] 在 {root} 下没找到 {args.glob}")
            continue
        for p in files:
            rows, cols = read_results(p)
            if not rows:
                continue
            cfg, seed, imgsz = parse_run_dir(p, root, args.strip_prefix)
            runs.append({"dir": str(p.parent), "cfg": cfg, "seed": seed, "imgsz": imgsz,
                         "rows": rows, "cols": cols, "path": str(p)})
    if not runs:
        print("[fatal] 没有读到任何 results.csv", file=sys.stderr)
        return 2

    all_cols = sorted({c for r in runs for c in r["cols"]})
    if args.metric not in all_cols and all_cols:
        args.metric = all_cols[0]
    print(f"[init] 读到 {len(runs)} 个 run, {len(all_cols)} 个指标列")
    print(f"[init] 主指标: {args.metric}")

    # ---- 每个 run 的曲线统计 ----------------------------------------- #
    for r in runs:
        r["stats"] = {c: plateau_stats(r["rows"], c, args.metric, args.plateau_frac) for c in r["cols"]}

    groups = defaultdict(list)
    for r in runs:
        groups[(r["cfg"], r["imgsz"])].append(r)
    gname = {k: (f"{k[0]}  @imgsz{k[1]}" if k[1] else k[0]) for k in groups}

    # ---- 组内 vs 组间 方差分解 --------------------------------------- #
    decomp = []
    for key, rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        with_seed = [r for r in rs if r["seed"] is not None]
        for col in all_cols:
            vals = [r["stats"][col]["best"] for r in with_seed if r["stats"].get(col)]
            within = [r["stats"][col]["plateau_sigma"] for r in rs if r["stats"].get(col)]
            if len(vals) >= 2:
                between = st.stdev(vals)
                df = len(vals) - 1
                se = between / math.sqrt(len(vals))
                mdd = t_crit(df, args.alpha) * se
            else:
                between = None
                df = None
                se = None
                mdd = None
            decomp.append({
                "cfg": gname[key], "imgsz": key[1], "metric": col, "n_seeds": len(vals),
                "mean_best": st.fmean(vals) if vals else None,
                "sigma_between": between,
                "sigma_within_mean": st.fmean(within) if within else None,
                "ratio": (between / st.fmean(within)) if (between and within and st.fmean(within) > 0) else None,
                "se": se, "df": df, "mdd": mdd,
                "mdd_pp": mdd * 100 if mdd is not None else None,
                "sigma_between_pp": between * 100 if between else None,
                "sigma_within_pp": (st.fmean(within) * 100) if within else None,
            })

    main_rows = [d for d in decomp if d["metric"] == args.metric and d["n_seeds"] >= 2]
    main_rows.sort(key=lambda d: (d["mdd_pp"] is None, d["mdd_pp"] or 9e9))

    # 每个指标挑种子数最多的配置, 用于横向比较"哪个终点分辨率最高"
    per_metric = []
    for col in all_cols:
        cand = [d for d in decomp if d["metric"] == col and d["n_seeds"] >= 2 and d["mdd_pp"]]
        if not cand:
            continue
        best = sorted(cand, key=lambda d: (-d["n_seeds"], d["mdd_pp"]))[0]
        per_metric.append(best)
    per_metric.sort(key=lambda d: d["mdd_pp"])
    sigma_for_n = None
    if per_metric:
        sigma_for_n = per_metric[0]["sigma_between"]

    # ---- 效应对照 ----------------------------------------------------- #
    effects = []
    ref = args.reference

    def resolve_ref(name):
        """按配置名或显示名找到分组键, 允许唯一前缀匹配。"""
        if not name:
            return None
        exact = [k for k in groups if gname[k] == name or k[0] == name]
        if len(exact) == 1:
            return exact[0]
        pref = [k for k in groups if gname[k].startswith(name) or k[0].startswith(name)]
        return pref[0] if len(pref) == 1 else None

    if ref:
        rkey = resolve_ref(ref)
        ref_stats = {r["seed"]: r["stats"].get(args.metric)
                     for r in groups.get(rkey, []) if r["seed"] is not None} if rkey else {}
        if not ref_stats:
            print(f"[warn] 找不到唯一匹配的参照配置: {ref}; 可用配置如下:")
            for k in sorted(groups, key=lambda k: str(k[0])):
                if any(r["seed"] is not None for r in groups[k]):
                    print(f"        {gname[k]}")
        else:
            for key, rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
                if key == rkey:
                    continue
                common = [r for r in rs if r["seed"] in ref_stats and ref_stats[r["seed"]]]
                if common:
                    d = [r["stats"][args.metric]["best"] - ref_stats[r["seed"]]["best"] for r in common]
                    nd = len(d)
                    sd = st.stdev(d) if nd > 1 else None
                    mdd_p = (t_crit(nd - 1, args.alpha) * sd / math.sqrt(nd)) if sd else None
                    effects.append({"cfg": gname[key], "vs": gname[rkey],
                                    "mode": f"配对 (n={nd}, seeds={sorted(r['seed'] for r in common)})",
                                    "effect": st.fmean(d), "sigma_d": sd, "mdd": mdd_p,
                                    "effect_pp": st.fmean(d) * 100,
                                    "mdd_pp": mdd_p * 100 if mdd_p else None,
                                    "significant": (abs(st.fmean(d)) > mdd_p) if mdd_p else None})
                else:
                    a = [r["stats"][args.metric]["best"] for r in rs if r["stats"].get(args.metric)]
                    b = [v["best"] for v in ref_stats.values() if v]
                    if len(a) >= 1 and len(b) >= 1:
                        effects.append({"cfg": gname[key], "vs": gname[rkey], "mode": "非配对 (种子不可比)",
                                        "effect": st.fmean(a) - st.fmean(b), "sigma_d": None, "mdd": None,
                                        "effect_pp": (st.fmean(a) - st.fmean(b)) * 100,
                                        "mdd_pp": None, "significant": None})

    # ---- 所需种子数 --------------------------------------------------- #
    targets = [float(x) / 100.0 for x in str(args.targets).replace(" ", "").split(",") if x]
    n_req = []
    if main_rows:
        sigma = main_rows[0]["sigma_between"]
        if sigma:
            for t in targets:
                n_req.append({"target_pp": t * 100, "required_n": required_n(sigma, t, args.alpha)})

    # ---- 报告 --------------------------------------------------------- #
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "power.json").write_text(json.dumps({
        "metric": args.metric, "runs": [
            {"dir": r["dir"], "cfg": r["cfg"], "seed": r["seed"],
             "stats": r["stats"].get(args.metric)} for r in runs],
        "variance_decomposition": decomp, "effects": effects, "required_n": n_req,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 统计功效分析：这套装置能分辨多大的效应\n",
         f"- 主指标: `{args.metric}`  (α={args.alpha}, 双尾)",
         f"- run 数: {len(runs)}；配置数: {len(groups)}",
         f"- 尾部平台段: 各 run 最后 {args.plateau_frac*100:.0f}% 的 epoch；"
         f"best 取全曲线最大值（对齐 ultralytics fitness 口径）\n"]

    L.append("## 1. 各 run 的曲线统计\n")
    L.append("| 配置 | imgsz | seed | 轮数 | best 轮 | best | 末轮 | 平台均值 | 平台σ | best−平台 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(runs, key=lambda x: (x["cfg"], x["imgsz"] or 0, x["seed"] if x["seed"] is not None else -1)):
        s = r["stats"].get(args.metric) or {}
        if not s:
            continue
        L.append(f"| {r['cfg']} | {r['imgsz'] if r['imgsz'] else '-'} | "
                 f"{r['seed'] if r['seed'] is not None else '-'} | {s['epochs']} | "
                 f"{s['best_epoch']} | {fmt(s['best'])} | {fmt(s['last'])} | {fmt(s['plateau_mean'])} | "
                 f"{s['plateau_sigma']*100:.4f}pp | {s['max_selection_gap']*100:+.4f}pp |")
    L.append("")

    L.append("## 2. 方差分解：加种子有没有用\n")
    L.append("| 配置 | 种子数 | 均值 | **组间σ (seed 间)** | **组内σ (epoch 间)** | 组间/组内 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for d in main_rows:
        ratio = f"{d['ratio']:.1f}×" if d["ratio"] else "-"
        L.append(f"| {d['cfg']} | {d['n_seeds']} | {fmt(d['mean_best'])} | "
                 f"{d['sigma_between_pp']:.4f}pp | {d['sigma_within_pp']:.4f}pp | {ratio} |")
    L.append("")
    L.append("> 组间 ≫ 组内 时，波动来自初始化/优化轨迹本身，不是评估噪声。")
    L.append("> 这种情况下把验证做得更精细不会降方差，**只能加种子**。\n")

    L.append("## 3. 最小可检测效应 MDD\n")
    L.append("| 配置 | 种子数 (df) | 组间σ | SE | t_crit | **MDD (α=0.05)** |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for d in main_rows:
        L.append(f"| {d['cfg']} | {d['n_seeds']} ({d['df']}) | {d['sigma_between_pp']:.4f}pp | "
                 f"{d['se']*100:.4f}pp | {t_crit(d['df'], args.alpha):.3f} | "
                 f"**{d['mdd_pp']:.4f}pp** |")
    L.append("")
    if main_rows:
        best = main_rows[0]
        L.append(f"> 主指标 `{args.metric}` 当前的分辨率下限约 **{best['mdd_pp']:.2f}pp**"
                 f"（配置 `{best['cfg']}`，n={best['n_seeds']}）。"
                 f"小于这个数的效应，在**当前种子数**下判为不显著；要测出来需要第 4 节给出的种子数。\n")

    if per_metric:
        L.append("## 4. 各指标的分辨率对比 —— 该把哪个当主终点\n")
        L.append("| 指标 | 达到最优的配置 | 种子数 | 均值 | 组间σ | 组内σ | **MDD** |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for d in per_metric:
            L.append(f"| `{d['metric']}` | {d['cfg']} | {d['n_seeds']} | {fmt(d['mean_best'])} | "
                     f"{d['sigma_between_pp']:.4f}pp | {d['sigma_within_pp']:.4f}pp | "
                     f"**{d['mdd_pp']:.4f}pp** |")
        L.append("")
        if len(per_metric) >= 2:
            b, w = per_metric[0], per_metric[-1]
            L.append(f"> MDD 最小的是 `{b['metric']}`（{b['mdd_pp']:.4f}pp），"
                     f"最大的是 `{w['metric']}`（{w['mdd_pp']:.4f}pp），"
                     f"相差 {w['mdd_pp']/b['mdd_pp']:.1f} 倍。"
                     f"**在同一个实验装置里换终点，就能换来这个倍数的分辨力。**")
            L.append(">")
            L.append("> 注意: `metrics/precision(B)` 与 `metrics/recall(B)` 是各 run 在自己的最佳 F1 置信度阈值下报的，")
            L.append("> 阈值每个 run 重调一次，因此跨 run 比较不是同一工作点。它们的 σ 若异常小，先确认不是这个原因造成的假象。\n")
        else:
            L.append("")
        dropped = [c for c in all_cols if c not in {d["metric"] for d in per_metric}]
        if dropped:
            L.append(f"> 未纳入对比的列: {', '.join('`%s`' % c for c in dropped)}"
                     "（种子数不足 2，或该列方差为 0 —— 方差为 0 通常说明这列在全部 run 里是常数，先查原因）。\n")

    if n_req:
        L.append("## 5. 要检测给定效应需要多少种子\n")
        if sigma_for_n is not None:
            L.append(f"（按第 4 节 MDD 最小的配置的组间 σ = {sigma_for_n*100:.4f}pp 反算）\n")
        L.append("| 目标效应 | 所需独立种子数 |")
        L.append("|---:|---:|")
        for r in n_req:
            L.append(f"| {r['target_pp']:.2f}pp | {r['required_n'] if r['required_n'] else '>5000 (不可行)'} |")
        L.append("")

    if effects:
        L.append(f"## 6. 与 `{ref}` 的效应对照\n")
        L.append("| 配置 | 比较方式 | 效应 | 配对σ | MDD | 判定 |")
        L.append("|---|---|---:|---:|---:|---|")
        for e in effects:
            if e["significant"] is None:
                verdict = "-"
            elif e["significant"]:
                verdict = "**显著**"
            else:
                verdict = "不显著（落在噪声内）"
            sd_txt = f"{e['sigma_d']*100:.4f}pp" if e["sigma_d"] else "-"
            mdd_txt = f"{e['mdd_pp']:.2f}pp" if e["mdd_pp"] else "-"
            L.append(f"| {e['cfg']} | {e['mode']} | {e['effect_pp']:+.2f}pp | "
                     f"{sd_txt} | {mdd_txt} | {verdict} |")
        L.append("")

    L.append("## 7. 结论怎么用\n")
    L.append("1. 先看第 2 节：组间/组内 的比值决定策略。比值大 → 波动来自初始化/优化轨迹，")
    L.append("   **把验证做精细没用，只有加种子或改配对设计**；比值小 → 还有评估层面的空间。")
    L.append("2. 第 3 节是主指标当下的分辨率下限，把它连同种子数一起写进论文的方法学部分。")
    L.append("3. **第 4 节是最该看的一节**：如果换一个终点指标能把 MDD 降几倍，那比加几倍种子便宜得多。")
    L.append("   换之前先在这里量出新指标的 MDD，确认它真的更小，而不是想当然。")
    L.append("4. 第 5 节回答「要坚持用当前指标要补多少种子」。若数字不可接受，走第 3 条的路线。")
    L.append("5. 配对比较通常比非配对强得多，但前提是两边跑**同一批 seed**。若参照配置缺种子，")
    L.append("   配对设计就落不了地 —— 补参照配置的种子，优先级高于补实验组的种子。\n")
    L.append("> 本分析只读已有 results.csv，不构成任何新的训练或评估结果。\n")

    (out_dir / "power.md").write_text("\n".join(L), encoding="utf-8")

    print("\n=== 摘要 ===")
    for d in main_rows:
        print(f"[{d['cfg']}] n={d['n_seeds']} 组间σ={d['sigma_between_pp']:.4f}pp "
              f"组内σ={d['sigma_within_pp']:.4f}pp 比值={d['ratio']:.1f}x "
              f"-> MDD={d['mdd_pp']:.4f}pp" if d["ratio"] else
              f"[{d['cfg']}] n={d['n_seeds']} MDD={d['mdd_pp']:.4f}pp")
    for r in n_req:
        print(f"[需要种子] 要检测 {r['target_pp']:.2f}pp -> n = {r['required_n']}")
    print(f"\n输出: {out_dir.resolve()}  (power.json / power.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
