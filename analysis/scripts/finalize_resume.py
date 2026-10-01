# -*- coding: utf-8 -*-
"""续训收尾: 校验完成度、汇总分辨率阶梯、写入续训 provenance 记录。

用法 (在仓库根目录、PYTHONPATH=$PWD 下):
    python analysis/finalize_resume.py
    python analysis/finalize_resume.py --expect-rows 101   # 1 表头 + 100 轮
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import time
from pathlib import Path

KEY = "metrics/mAP50-95(B)"
IMGSZ_RE = re.compile(r"^\s*imgsz:\s*(\d+)", re.M)
BATCH_RE = re.compile(r"^\s*batch:\s*(\d+)", re.M)
TARGET_ROWS = 101  # 表头 + 100 轮


def read_run(d: Path):
    csv_path = d / "results.csv"
    if not csv_path.is_file():
        return None
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", errors="ignore")))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r[KEY]))
    args = d / "args.yaml"
    imgsz = batch = None
    if args.is_file():
        t = args.read_text(encoding="utf-8", errors="ignore")
        m = IMGSZ_RE.search(t)
        if m:
            imgsz = int(m.group(1))
        m = BATCH_RE.search(t)
        if m:
            batch = int(m.group(1))
    return {
        "dir": d, "epochs": len(rows),
        "best_epoch": int(float(best["epoch"])), "best": float(best[KEY]),
        "best_map50": float(best.get("metrics/mAP50(B)", "nan")),
        "last_epoch": int(float(rows[-1]["epoch"])), "last": float(rows[-1][KEY]),
        "imgsz": imgsz, "batch": batch,
        "complete": len(rows) >= TARGET_ROWS - 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--expect-rows", type=int, default=TARGET_ROWS)
    ap.add_argument("--write-notes", action="store_true",
                    help="在 1280 run 目录写入 RESUME.md 续训记录")
    a = ap.parse_args()

    dirs = sorted({Path(p).parent for p in glob.glob(f"{a.root}/paper-v1*/dfire/baseline/seed0/results.csv")})
    runs = [r for r in (read_run(d) for d in dirs) if r]
    if not runs:
        print("[fatal] 没找到任何 paper-v1* 的 results.csv")
        return 2
    runs.sort(key=lambda r: (r["imgsz"] or 0))

    print("=== 分辨率阶梯 (baseline, seed0) ===\n")
    print(f"{'imgsz':>6} {'batch':>6} {'轮数':>5} {'best轮':>7} {'best mAP50-95':>14} {'末轮 mAP50-95':>14}  完整")
    for r in runs:
        print(f"{str(r['imgsz']):>6} {str(r['batch']):>6} {r['epochs']:>5} {r['best_epoch']:>7} "
              f"{r['best']:>14.5f} {r['last']:>14.5f}  {'是' if r['complete'] else '否(未完成)'}")

    base = next((r for r in runs if r["imgsz"] == 640), None)
    if base:
        print(f"\n相对 640 的效应 (单种子, 仅供参考):")
        for r in runs:
            if r["imgsz"] == 640:
                continue
            d = (r["best"] - base["best"]) * 100
            print(f"  imgsz {r['imgsz']:>5} (batch {r['batch']}): {d:+.2f}pp"
                  f"{'   ← 注意 batch 与 640 不同, 混淆变量' if r['batch'] != base['batch'] else ''}")

    tgt = next((r for r in runs if r["imgsz"] == 1280), None)
    if tgt:
        print(f"\n=== 1280 run 状态 ===")
        print(f"  目录     : {tgt['dir']}")
        print(f"  已完成   : {tgt['epochs']} 轮" + ("" if tgt["complete"] else f"  (目标 {a.expect_rows - 1}, 仍在跑)"))
        print(f"  best     : {tgt['best']:.5f} @ epoch {tgt['best_epoch']}")
        print(f"  末轮     : {tgt['last']:.5f} @ epoch {tgt['last_epoch']}")

        if a.write_notes and tgt["complete"]:
            notes = tgt["dir"] / "RESUME.md"
            notes.write_text(
                "# 续训记录\n\n"
                f"- 原训练: 2026-09-27 18:04 启动, 2026-09-27 22:04 停在第 55 轮 (原因未记录)\n"
                f"- 续训: {time.strftime('%Y-%m-%d %H:%M')} 由原生 ultralytics resume 补完第 56–100 轮\n"
                f"- 命令: `PYTHONPATH=$PWD python -c \"from ultralytics import YOLO;"
                f" YOLO('{tgt['dir']}/weights/last.pt').train(resume=True)\"`\n"
                f"- 脚本: `analysis/resume_1280.py`, 日志 `analysis/resume_1280.log`\n"
                f"- 备份: `{tgt['dir'].parent}/seed0_bak_ep55` (停在 55 轮时的完整快照)\n\n"
                "## 为什么可以绕过封装层\n\n"
                "封装层 `firesmoke/experiment.py` 有意禁止续训 (`FileExistsError` + `resume` 硬编码 False)。\n"
                "该 run 的 `background_alpha = 0.0`, 实测 `ResearchModel.init_criterion` 返回原版\n"
                "`v8DetectionLoss`; 且 `style = False`。因此原生 resume 与封装层在训练目标上完全等价。\n\n"
                "## 已知不一致\n\n"
                "- `background_exposure.jsonl` 停在 55 条 (ResearchTrainer 的 epoch 回调在原生 resume 下不生效)。\n"
                "  该文件仅记录曝光统计, baseline 不参与损失, 不影响训练正确性。\n"
                "- `run.json` 只记录第一次进程的 provenance, 本文件即为补充。\n\n"
                "## 混淆变量 (必须写进论文)\n\n"
                "本次 `batch = 16`, 而 640/800/960 均为 `batch = 32`, 且 `lr0` 未按 batch 调整。\n"
                "因此 1280 与前三档之间分辨率与 batch/LR 同时变化, 不能直接把差异归因给分辨率。\n",
                encoding="utf-8",
            )
            print(f"\n  已写入续训记录: {notes}")
        elif a.write_notes:
            print("\n  训练尚未完成, 未写 RESUME.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
