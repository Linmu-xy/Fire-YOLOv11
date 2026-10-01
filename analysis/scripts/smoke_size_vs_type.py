# -*- coding: utf-8 -*-
"""判明"纯烟图召回低"是置信度效应还是目标尺寸效应。

问题
----
实测（P2 seed0，同一套源域冻结阈值）：
    域内 D-Fire val   smoke_only 62.8%  vs  both 79.6%   (+16.8pp)
    跨域 FASDD test   smoke_only 33.9%  vs  both 51.8%   (+17.9pp)
两个域的落差幅度几乎一样 → 看似是稳健的"依赖火"结构。

但还有另一个解释：**both 图里的烟本来就更大更显著**（火通常伴随浓烟）。
若如此，"依赖火"只是尺寸效应的伪装。

判据
----
按 **box 级**统计（不是图级）：
  1. 列出烟框在两个域的**尺寸分布**（smoke_only 图 vs both 图）
  2. 在**同一尺寸档内**比较 smoke_only vs both 的 box 召回
若同档内差距消失 → 是尺寸效应 (b)；若同档内差距仍在 → 是置信度效应 (a)。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/smoke_size_vs_type.py --out analysis/out_smoke_size
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EDGES = [0.0, 8.0, 16.0, 32.0, 64.0, 128.0, float("inf")]
LABELS = ["<8", "8-16", "16-32", "32-64", "64-128", ">=128"]
IOU_THR = 0.5


def xywhn_to_xyxy(b, W, H):
    cx, cy, w, h = b
    x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
    return [x1, y1, x1 + w * W, y1 + h * H]


def iou(a, bs):
    if not bs:
        return []
    a = np.asarray(a, dtype=float)
    b = np.asarray(bs, dtype=float)
    ix1 = np.maximum(a[0], b[:, 0]); iy1 = np.maximum(a[1], b[:, 1])
    ix2 = np.minimum(a[2], b[:, 2]); iy2 = np.minimum(a[3], b[:, 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(aa + ab - inter, 1e-9)).tolist()


def band(v):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= v < EDGES[i + 1]:
            return i
    return len(EDGES) - 2


def resolve(img_root: str, img_id: str) -> Path:
    root = Path(img_root)
    parts = str(img_id).split("/")
    if len(parts) >= 3:
        c = root / parts[0] / "images" / parts[1] / "/".join(parts[2:])
        if c.exists():
            return c
    return root / img_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-pred", default=None, help="域内(源域 val) predictions.json")
    ap.add_argument("--tgt-pred", default=None, help="跨域(目标域 test) predictions.json")
    ap.add_argument("--thr-fire", type=float, required=True)
    ap.add_argument("--thr-smoke", type=float, required=True)
    ap.add_argument("--img-root", default="artifacts/data")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="analysis/out_smoke_size")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t_fire, t_smoke = a.thr_fire, a.thr_smoke
    DOMS = []
    if a.src_pred:
        DOMS.append(("域内 D-Fire val", Path(a.src_pred)))
    if a.tgt_pred:
        DOMS.append(("跨域 FASDD test", Path(a.tgt_pred)))
    DOMS = [(n, p) for n, p in DOMS if p.is_file()]
    if not DOMS:
        raise SystemExit("没有任何可用的 predictions 路径")

    from PIL import Image
    cache: dict[str, tuple[int, int]] = {}

    def size_of(p: Path):
        k = str(p)
        if k not in cache:
            try:
                with Image.open(p) as im:
                    cache[k] = im.size
            except Exception:
                cache[k] = (0, 0)
        return cache[k]

    # recall[dom][cat][band] = [hit, total];  sizes[dom][cat] = [long_lb,...]
    recall = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    sizes = defaultdict(lambda: defaultdict(list))

    for dom, pp in DOMS:
        art = json.loads(pp.read_text())
        for im in art["images"]:
            W, H = size_of(resolve(a.img_root, im["id"]))
            if W <= 0:
                continue
            s = a.imgsz / max(W, H)
            tcls = {int(t[0]) for t in im["truth"]}
            cat = "none" if not tcls else ("fire_only" if tcls == {0} else
                                           ("smoke_only" if tcls == {1} else "both"))
            if cat not in ("smoke_only", "both"):
                continue
            # 烟 GT
            gts = [xywhn_to_xyxy(t[1:5], W, H) for t in im["truth"] if int(t[0]) == 1]
            glong = [max((t[3]) * W, (t[4]) * H) * s for t in im["truth"] if int(t[0]) == 1]
            preds = [(xywhn_to_xyxy(p[1:5], W, H), float(p[5]))
                     for p in im["predictions"] if int(p[0]) == 1 and float(p[5]) >= t_smoke]
            preds.sort(key=lambda x: -x[1])
            used = [False] * len(gts)
            for pb, _pc in preds:
                ious = iou(pb, gts)
                best, bj = 0.0, -1
                for j, v in enumerate(ious):
                    if not used[j] and v > best:
                        best, bj = v, j
                if bj >= 0 and best >= IOU_THR:
                    used[bj] = True
            for j, gl in enumerate(glong):
                b = band(gl)
                recall[dom][cat][b][1] += 1
                if used[j]:
                    recall[dom][cat][b][0] += 1
                sizes[dom][cat].append(gl)

    L = [f"# 烟的召回：尺寸效应 vs 类别效应\n",
         f"- box 级统计，IoU≥{IOU_THR}，预测需 ≥ 源域冻结阈值（fire={t_fire:.3f}, smoke={t_smoke:.3f}）",
         "- `smoke_only` = 图里只有烟；`both` = 火烟同框\n",
         "## 1. 烟框的尺寸分布（share = 占该类图烟框的百分比）\n",
         "| 域 | 图类别 | 烟框数 | 中位长边 | " + " | ".join(LABELS) + " |",
         "|---|---|---:|---:|" + "---:|" * len(LABELS)]
    for dom, _pp in DOMS:
        for cat in ("smoke_only", "both"):
            vals = sorted(sizes[dom][cat])
            if not vals:
                continue
            n = len(vals)
            med = vals[n // 2]
            cnt = [0] * len(LABELS)
            for v in vals:
                cnt[band(v)] += 1
            L.append(f"| {dom} | {cat} | {n} | {med:.1f}px | " +
                     " | ".join(f"{c/n*100:.1f}%" for c in cnt) + " |")
    L.append("")

    L.append("## 2. 分档的 box 召回\n")
    L.append("| 域 | 图类别 | " + " | ".join(LABELS) + " |")
    L.append("|---|---|" + "---:|" * len(LABELS))
    for dom, _pp in DOMS:
        for cat in ("smoke_only", "both"):
            cells = []
            for b in range(len(LABELS)):
                h, t = recall[dom][cat][b]
                cells.append(f"{h/t*100:.1f}% (n={t})" if t else "—")
            L.append(f"| {dom} | {cat} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 3. 同档内的类别落差（both − smoke_only）\n")
    L.append("| 域 | " + " | ".join(LABELS) + " |")
    L.append("|---|" + "---:|" * len(LABELS))
    for dom, _pp in DOMS:
        cells = []
        for b in range(len(LABELS)):
            h1, t1 = recall[dom]["both"][b]
            h2, t2 = recall[dom]["smoke_only"][b]
            if t1 >= 5 and t2 >= 5:
                cells.append(f"{(h1/t1 - h2/t2)*100:+.1f}pp")
            else:
                cells.append("—")
        L.append(f"| {dom} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 4. 判读\n")
    L.append("- 先看第 1 节：若 `both` 的烟框中位长边明显大于 `smoke_only`，说明两者**目标尺度分布不同**。")
    L.append("- 再看第 3 节：**同档内**如果落差大幅缩小甚至反转 → 落差主要来自尺寸 (b)；")
    L.append("  若同档内落差依旧 → 是置信度/类别效应 (a)，那才是真正需要干预的结构问题。")
    L.append("- 第 2 节里 n<20 的格子不要解读。\n")
    L.append("> 不含任何新训练或新推理。")
    (out / "smoke_size_vs_type.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
