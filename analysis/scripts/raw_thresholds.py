# -*- coding: utf-8 -*-
"""从源域 val 的 predictions.json 求**原始分数**上的 1% FPR 阈值。

为什么要单独做
--------------
`firesmoke.reliability.fit` 返回的 threshold 是在**温度缩放后**的分数上选的，
不能直接用于原始分数（而 `probe_negatives.py` 作用在原始分数上）。
本脚本复用 `select_threshold`，但**不缩放**，得到可直接用于探测的原始阈值。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/scripts/raw_thresholds.py \
    --predictions analysis/work/calib/val_dfire/predictions.json \
    --label dfire --max-fpr 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from firesmoke.reliability import arrays, select_threshold  # noqa: E402

NAMES = ("fire", "smoke")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-fpr", type=float, default=0.01)
    ap.add_argument("--json-out", default=None, help="把阈值写到该 json（便于脚本串联）")
    a = ap.parse_args()

    art = json.loads(Path(a.predictions).read_text())
    if art.get("split") != "val":
        print(f"!! 只接受 val 划分，实际是 {art.get('split')}")
        return 2
    scores, labels = arrays(art)
    print(f"{a.label or a.predictions}")
    print(f"  图像数 {len(scores)} | 训练域 {art.get('training_dataset')} | 评估域 {art.get('dataset')}")
    out = {}
    for c, name in enumerate(NAMES):
        sel = select_threshold(scores[:, c], labels[:, c], a.max_fpr)
        npos = int((labels[:, c] == 1).sum())
        nneg = int((labels[:, c] == 0).sum())
        out[name] = sel["threshold"]
        print(f"  {name:<6} 原始阈值={sel['threshold']:.4f}  "
              f"val召回={sel['validation_recall']:.4f}  val_FPR={sel['validation_fpr']:.4f}  "
              f"(正样本图 {npos} / 负样本图 {nneg})")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写入 {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
