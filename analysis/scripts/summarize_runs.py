#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 experiments/ 下所有训练的 results.csv，输出「精度-计算量」对照表与 Pareto 图。

放到 Fire-YOLOv11 仓库根目录下的 analysis/ 里运行：

    python analysis/summarize_runs.py
    python analysis/summarize_runs.py --root experiments --out analysis/out
    python analysis/summarize_runs.py --complexity analysis/complexity.csv

设计要点
--------
1. 只用标准库读 CSV，不依赖 pandas；matplotlib 缺失时自动跳过画图。
2. 每个 run 同时给「best 行」和「末轮行」两个口径，避免把不同 epoch 的 P/R/AP 拼在一起。
   模型选择口径固定为 val mAP50-95 最大（与上游 fitness 一致），不另挑最高 P 或最高 R。
3. 计算量两条来源，互不混用：
   a. --complexity 提供的结构实测值（推荐，记录 imgsz 与 profiler 口径）；
   b. 未提供时按 (imgsz/基准)^2 从 --complexity 缩放，并在表里标 est。
4. 训练总时长只作为参考列。跨机器的 wall-clock 不可用于速度结论，报告里必须换成
   同设备 batch1 的 latency p50/p95。

--complexity 格式（CSV，第三列 gflops 按 --base-imgsz 记录）：

    run_match,params,gflops
    protocol_baseline,2590230,6.4417
    p2,2638312,10.1378

run_match 是目录名子串（不区分大小写），最长匹配优先。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

BEST_KEY = "metrics/mAP50-95(B)"
MAP50_KEY = "metrics/mAP50(B)"
P_KEY = "metrics/precision(B)"
R_KEY = "metrics/recall(B)"
TIME_KEY = "time"


def read_args_yaml(path: Path) -> dict:
    """轻量解析 ultralytics 的 args.yaml。有 PyYAML 就用，没有就按 key: value 逐行读。"""
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
    except Exception:
        pass

    out: dict = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=None):
    f = as_float(value)
    return default if f is None else int(round(f))


def scan_results(csv_path: Path) -> dict | None:
    """读取一次训练的结果，返回 best 行与末轮行。"""
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows:
        return None

    def clean(row: dict) -> dict | None:
        value = as_float(row.get(BEST_KEY))
        if value is None:
            return None
        return {
            "epoch": as_int(row.get("epoch")),
            "mAP50_95": value,
            "mAP50": as_float(row.get(MAP50_KEY)),
            "precision": as_float(row.get(P_KEY)),
            "recall": as_float(row.get(R_KEY)),
            "cum_time_s": as_float(row.get(TIME_KEY)),
        }

    scored = [(clean(r), r) for r in rows]
    scored = [(c, r) for c, r in scored if c is not None]
    if not scored:
        return None

    best_row, _ = max(scored, key=lambda item: item[0]["mAP50_95"])
    last_row, _ = scored[-1]
    epochs = len(scored)
    # 单轮平均耗时：末轮累计时间 / 已完成 epoch 数，只用于粗看相对成本
    per_epoch = None
    if last_row["cum_time_s"] and epochs:
        per_epoch = last_row["cum_time_s"] / epochs
    return {
        "epochs_done": epochs,
        "best": best_row,
        "last": last_row,
        "sec_per_epoch": per_epoch,
    }


def load_complexity(path: Path | None) -> list[dict]:
    if path is None:
        return []
    if not path.is_file():
        print(f"[warn] complexity 文件不存在，忽略：{path}", file=sys.stderr)
        return []
    items = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            match = (row.get("run_match") or "").strip()
            gflops = as_float(row.get("gflops"))
            if not match or gflops is None:
                continue
            items.append(
                {
                    "match": match.lower(),
                    "params": as_int(row.get("params")),
                    "gflops": gflops,
                }
            )
    items.sort(key=lambda d: len(d["match"]), reverse=True)
    return items


def resolve_complexity(name: str, items: list[dict]) -> dict | None:
    lowered = name.lower()
    for item in items:
        if item["match"] in lowered:
            return item
    return None


def collect(root: Path, complexity: list[dict], base_imgsz: int) -> list[dict]:
    runs = []
    for csv_path in sorted(root.rglob("results.csv")):
        run_dir = csv_path.parent
        rel = run_dir.relative_to(root)
        parsed = scan_results(csv_path)
        if parsed is None:
            print(f"[skip] 无有效指标：{rel}", file=sys.stderr)
            continue

        args = read_args_yaml(run_dir / "args.yaml")
        run_json = run_dir / "run.json"
        meta = {}
        if run_json.is_file():
            try:
                meta = json.loads(run_json.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                meta = {}

        imgsz = as_int(args.get("imgsz")) or base_imgsz
        entry = {
            "run": str(rel).replace("\\", "/"),
            "imgsz": imgsz,
            "model": str(args.get("model", "")),
            "batch": as_int(args.get("batch")),
            "optimizer": str(args.get("optimizer", "")),
            "amp": str(args.get("amp", "")),
            "seed": as_int(args.get("seed")),
            "protocol": str(meta.get("protocol", "") or meta.get("protocol_version", "")),
            "best_epoch": parsed["best"]["epoch"],
            "mAP50_95": parsed["best"]["mAP50_95"],
            "mAP50": parsed["best"]["mAP50"],
            "precision": parsed["best"]["precision"],
            "recall": parsed["best"]["recall"],
            "last_mAP50_95": parsed["last"]["mAP50_95"],
            "epochs_done": parsed["epochs_done"],
            "sec_per_epoch": parsed["sec_per_epoch"],
            "params": None,
            "gflops": None,
            "gflops_source": "missing",
        }

        hit = resolve_complexity(entry["run"], complexity)
        if hit:
            entry["params"] = hit["params"]
            ratio = (imgsz / float(base_imgsz)) ** 2
            entry["gflops"] = hit["gflops"] * ratio
            entry["gflops_source"] = "meas" if abs(ratio - 1.0) < 1e-9 else f"est(imgsz={imgsz})"
        runs.append(entry)
    return runs


def fmt(value, digits=4, dash="-"):
    return dash if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, runs: list[dict]) -> None:
    cols = [
        "run", "imgsz", "best_epoch", "epochs_done", "mAP50_95", "mAP50",
        "precision", "recall", "last_mAP50_95", "params", "gflops",
        "gflops_source", "sec_per_epoch", "optimizer", "amp", "seed", "model",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in runs:
            writer.writerow(row)


def write_markdown(path: Path, runs: list[dict]) -> None:
    header = (
        "| run | imgsz | best ep | mAP50-95 | mAP50 | P | R | 末轮 mAP50-95 | "
        "params | GFLOPs | 来源 | s/epoch |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n"
    )
    lines = []
    for row in runs:
        lines.append(
            "| {run} | {imgsz} | {ep} | {b} | {m} | {p} | {r} | {l} | {pa} | {g} | {gs} | {t} |".format(
                run=row["run"],
                imgsz=row["imgsz"],
                ep=row["best_epoch"],
                b=fmt(row["mAP50_95"]),
                m=fmt(row["mAP50"]),
                p=fmt(row["precision"]),
                r=fmt(row["recall"]),
                l=fmt(row["last_mAP50_95"]),
                pa=row["params"] if row["params"] is not None else "-",
                g=fmt(row["gflops"], 3),
                gs=row["gflops_source"],
                t=fmt(row["sec_per_epoch"], 1),
            )
        )
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def write_pareto_png(path: Path, runs: list[dict]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] 跳过 Pareto 图（matplotlib 不可用：{exc}）", file=sys.stderr)
        return False

    pts = [r for r in runs if r["gflops"]]
    if not pts:
        print("[warn] 没有计算量数据，跳过 Pareto 图；请提供 --complexity", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=160)
    for row in pts:
        marker = "o" if row["gflops_source"] == "meas" else "^"
        ax.scatter(row["gflops"], row["mAP50_95"], s=52, marker=marker,
                   color="#2f6fb2", edgecolor="#173f66", linewidth=0.6, zorder=3)
        ax.annotate(
            f"{row['run'].split('/')[-1]}\n@{row['imgsz']}",
            (row["gflops"], row["mAP50_95"]),
            textcoords="offset points", xytext=(7, 5), fontsize=7, color="#333333",
        )

    # 左上方越远越好：标出 Pareto 前沿
    ordered = sorted(pts, key=lambda r: r["gflops"])
    frontier, best = [], -1.0
    for row in ordered:
        if row["mAP50_95"] > best:
            frontier.append(row)
            best = row["mAP50_95"]
    if len(frontier) > 1:
        ax.plot([r["gflops"] for r in frontier], [r["mAP50_95"] for r in frontier],
                color="#c0392b", linewidth=1.2, linestyle="--", zorder=2,
                label="Pareto frontier")
        ax.legend(fontsize=8, frameon=False)

    ax.set_xlabel("GFLOPs @ imgsz (est. by area scaling unless marked meas)", fontsize=9)
    ax.set_ylabel("best val mAP50-95", fontsize=9)
    ax.set_title("Accuracy vs compute", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总所有训练 run 的精度与计算量")
    default_root = Path("experiments") if Path("experiments").is_dir() else Path(".")
    parser.add_argument("--root", type=Path, default=default_root,
                       help="扫描根目录，默认 experiments/")
    parser.add_argument("--out", type=Path, default=Path("analysis/out"),
                       help="输出目录，默认 analysis/out")
    parser.add_argument("--complexity", type=Path, default=None,
                       help="结构参数量/GFLOPs 表（CSV），用于精度-计算量对照")
    parser.add_argument("--base-imgsz", type=int, default=640,
                       help="complexity 表中 gflops 对应的 imgsz，默认 640")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"[error] 目录不存在：{args.root}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    complexity = load_complexity(args.complexity)
    runs = collect(args.root, complexity, args.base_imgsz)
    if not runs:
        print(f"[error] 在 {args.root} 下没找到可用的 results.csv", file=sys.stderr)
        return 1

    runs.sort(key=lambda r: r["mAP50_95"], reverse=True)
    write_csv(args.out / "runs.csv", runs)
    write_markdown(args.out / "runs.md", runs)
    wrote_png = write_pareto_png(args.out / "pareto.png", runs)

    print(f"扫描到 {len(runs)} 个 run")
    print(f"  {args.out / 'runs.csv'}")
    print(f"  {args.out / 'runs.md'}")
    if wrote_png:
        print(f"  {args.out / 'pareto.png'}")

    # 直接打印一张紧凑表，方便粘进实验记录
    print("\nrun".ljust(46), "imgsz", "best", "mAP50-95", "GFLOPs")
    for row in runs:
        print(
            row["run"][:45].ljust(46),
            str(row["imgsz"]).rjust(5),
            str(row["best_epoch"]).rjust(4),
            fmt(row["mAP50_95"]).rjust(8),
            fmt(row["gflops"], 3).rjust(8),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
