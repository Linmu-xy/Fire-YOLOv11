# -*- coding: utf-8 -*-
"""列出 outputs/ 下所有 run 的训练状态与最佳结果。

用法:
    cd <repo>; python analysis/scripts/run_status.py                 # 全部
    cd <repo>; python analysis/scripts/run_status.py --tag paper-v1   # 只看某个 tag
    cd <repo>; python analysis/scripts/run_status.py --incomplete     # 只看未跑完的
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

KEY = "metrics/mAP50-95(B)"
TARGET_ROWS = 100  # 协议规定 epochs=100


def read_run(d: Path):
    csv_path = d / "results.csv"
    if not csv_path.is_file():
        return None
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", errors="ignore")))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r[KEY]))
    info = {
        "dir": d,
        "epochs": len(rows),
        "best": float(best[KEY]),
        "best_epoch": int(float(best["epoch"])),
        "last": float(rows[-1][KEY]),
        "last_epoch": int(float(rows[-1]["epoch"])),
        "complete": len(rows) >= TARGET_ROWS,
        "finished_flag": None,
        "dataset": None,
        "method": None,
        "seed": None,
        "tag": d.parent.parent.parent.name if len(d.parts) >= 4 else None,
    }
    rj = d / "run.json"
    if rj.is_file():
        try:
            j = json.loads(rj.read_text())
            info["finished_flag"] = bool(j.get("training_finished"))
            info["dataset"] = j.get("dataset")
            info["method"] = j.get("method")
            info["seed"] = j.get("seed")
        except Exception:
            pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--incomplete", action="store_true")
    a = ap.parse_args()

    found = []
    for d in sorted(Path(a.root).glob("*/*/*/seed*")):
        r = read_run(d)
        if not r:
            continue
        r["tag"] = str(d).split("/")[1]
        if a.tag and r["tag"] != a.tag:
            continue
        if a.incomplete and r["complete"]:
            continue
        found.append(r)

    if not found:
        print("(没有匹配的 run)")
        return 0

    print(f"{'tag':<14}{'dataset':<22}{'method':<10}{'seed':>4} {'轮数':>5} {'best':>9} {'@ep':>4} {'末轮':>9}  {'完成':<5} run.json")
    print("-" * 104)
    n_inc = 0
    for r in found:
        if not r["complete"]:
            n_inc += 1
        print(f"{str(r['tag']):<14}{str(r['dataset']):<22}{str(r['method']):<10}{str(r['seed']):>4} "
              f"{r['epochs']:>5} {r['best']:>9.5f} {r['best_epoch']:>4} {r['last']:>9.5f}  "
              f"{'是' if r['complete'] else '否':<5} {r['finished_flag']}")
    print("-" * 104)
    print(f"共 {len(found)} 个 run，其中未完成 {n_inc} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
