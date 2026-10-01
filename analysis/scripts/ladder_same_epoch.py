# -*- coding: utf-8 -*-
"""分辨率阶梯的同轮次 / 同优化步数对比。

为什么两种对齐都要看
--------------------
1280 那次 batch=16, 640/800/960 都是 batch=32。因此:
  - 同 epoch     : 数据都过了一遍, 但 1280 走的 optimizer step 是别人的 2 倍
  - 同 step      : 优化更新次数相同, 但 1280 只看了一半的数据量
两者都不完美, 所以都列出来; 只有当两种对齐下结论一致时, 结论才可信。

用法 (仓库根目录):
    python analysis/ladder_same_epoch.py
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

KEY = "metrics/mAP50-95(B)"
LOSS_COLS = ["train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"]
IMGSZ_RE = re.compile(r"^\s*imgsz:\s*(\d+)", re.M)
BATCH_RE = re.compile(r"^\s*batch:\s*(\d+)", re.M)
TRAIN_IMAGES = 15485  # artifacts/data/dfire/train.txt 行数


def load(d: Path):
    p = d / "results.csv"
    if not p.is_file():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8", errors="ignore")))
    if not rows:
        return None
    args = d / "args.yaml"
    if not args.is_file():
        return None
    t = args.read_text(encoding="utf-8", errors="ignore")
    mi, mb = IMGSZ_RE.search(t), BATCH_RE.search(t)
    if not mi or not mb:
        print(f"[skip] {d}: args.yaml 里没有 imgsz/batch")
        return None
    imgsz, batch = int(mi.group(1)), int(mb.group(1))
    iters = math.ceil(TRAIN_IMAGES / batch)
    seq = []
    for r in rows:
        e = int(float(r["epoch"]))
        rec = {"epoch": e, "map": float(r[KEY]),
               "map50": float(r.get("metrics/mAP50(B)", "nan")),
               "lr": float(r.get("lr/pg0", "nan")), "steps": e * iters}
        for c in LOSS_COLS:
            v = r.get(c)
            rec[c] = float(v) if v not in (None, "") else None
        seq.append(rec)
    return {"dir": d, "imgsz": imgsz, "batch": batch, "iters": iters, "seq": seq}


def at_epoch(run, epoch):
    for s in run["seq"]:
        if s["epoch"] == epoch:
            return s
    return None


def at_steps(run, steps):
    """找累计优化步数最接近的轮次 (要求同向, 不插值)。"""
    best = None
    for s in run["seq"]:
        if s["steps"] is None:
            continue
        d = abs(s["steps"] - steps)
        if best is None or d < best[0]:
            best = (d, s)
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--epochs", default="10,20,30,40,50,55,60,70,80,90,100")
    a = ap.parse_args()

    runs = {}
    for p in sorted(Path(a.root).glob("paper-v1*/dfire/baseline/seed0")):
        r = load(p)
        if r and r["imgsz"]:
            runs[r["imgsz"]] = r
    if not runs:
        print("[fatal] 没找到任何 paper-v1* run")
        return 2

    order = sorted(runs)
    print("=== 各 run 的 batch / 每轮步数 / 已完成轮数 ===")
    for k in order:
        r = runs[k]
        print(f"  imgsz {k:>5}  batch {r['batch']:>3}  每轮 {r['iters']:>4} step  "
              f"已完成 {len(r['seq']):>3} 轮")
    print()

    epochs = [int(x) for x in a.epochs.split(",") if x.strip()]
    print("=== 表 A: 同 epoch 对比 (mAP50-95) ===")
    hdr = "  轮次 |" + "".join(f"{k:>12}" for k in order) + " |  1280-960  1280-640"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for e in epochs:
        vals = {}
        for k in order:
            s = at_epoch(runs[k], e)
            vals[k] = s["map"] if s else None
        row = f"  {e:>4} |" + "".join(
            (f"{vals[k]:>12.5f}" if vals[k] is not None else f"{'--':>12}") for k in order)
        d1 = (vals[1280] - vals[960]) * 100 if vals.get(1280) and vals.get(960) else None
        d2 = (vals[1280] - vals[640]) * 100 if vals.get(1280) and vals.get(640) else None
        row += " | " + (f"{d1:+9.2f}pp" if d1 is not None else f"{'--':>11}")
        row += " " + (f"{d2:+9.2f}pp" if d2 is not None else f"{'--':>11}")
        print(row)

    # 同优化步数
    ref = 640
    print(f"\n=== 表 B: 同累计优化步数对比 (以 {ref} 的步数为基准) ===")
    print(f"  {'累计step':>9} {'≈'+ref.__str__()+'轮':>7} |" +
          "".join(f"{k:>10}(步)" for k in order) + " |  1280-960")
    for s in runs[ref]["seq"]:
        if s["epoch"] not in epochs:
            continue
        tgt = s["steps"]
        cells = []
        vals = {}
        for k in order:
            m = at_steps(runs[k], tgt)
            if m:
                vals[k] = m["map"]
                cells.append(f"{m['map']:>6.4f}({m['epoch']:>3})")
            else:
                cells.append(f"{'--':>10}")
        d1 = (vals[1280] - vals[960]) * 100 if vals.get(1280) and vals.get(960) else None
        print(f"  {tgt:>9} {s['epoch']:>7} |" + "".join(f"{c:>10}" for c in cells) +
              " | " + (f"{d1:+9.2f}pp" if d1 is not None else f"{'--':>11}"))

    # 损失对比: 区分"优化没跟上"还是"指标口径/泛化问题"
    print("\n=== 同 epoch 的损失对比 ===")
    print("  读法: loss 更低 = 更贴合训练目标。若 1280 的 loss 更低但 mAP 更差,")
    print("        说明它拟合得不差, 差距来自评估/泛化而非优化; 反之则是优化问题。\n")
    for c in LOSS_COLS:
        print(f"  --- {c}")
        print("    轮次 |" + "".join(f"{k:>13}" for k in order))
        for e in epochs:
            cells = []
            for k in order:
                s = at_epoch(runs[k], e)
                v = s.get(c) if s else None
                cells.append(f"{v:>13.4f}" if v is not None else f"{'--':>13}")
            print(f"    {e:>4} |" + "".join(cells))
        print()

    if 1280 in runs and 960 in runs:
        print("  --- 差值 (1280 - 960)")
        print("    轮次 |" + "".join(f"{c:>16}" for c in LOSS_COLS))
        for e in epochs:
            a, b = at_epoch(runs[1280], e), at_epoch(runs[960], e)
            if not a or not b:
                continue
            cells = []
            for c in LOSS_COLS:
                cells.append(f"{a[c] - b[c]:>+16.4f}"
                             if a.get(c) is not None and b.get(c) is not None else f"{'--':>16}")
            print(f"    {e:>4} |" + "".join(cells))

    # 学习率对齐检查
    print("\n=== 学习率调度是否对齐 (lr/pg0) ===")
    for e in (10, 30, 50, 55):
        lrs = {k: (at_epoch(runs[k], e) or {}).get("lr") for k in order}
        same = len({round(v, 8) for v in lrs.values() if v is not None}) <= 1
        print(f"  epoch {e:>3}: " + "  ".join(
            f"{k}={lrs[k]:.6f}" if lrs[k] is not None else f"{k}=--" for k in order) +
            ("   [一致]" if same else "   [不一致!]"))

    # 尾部斜率
    print("\n=== 末段斜率 (判断是否还在爬) ===")
    for k in order:
        sq = runs[k]["seq"]
        if len(sq) < 5:
            continue
        tail = sq[-10:] if len(sq) >= 10 else sq
        n = len(tail)
        xm = sum(x["epoch"] for x in tail) / n
        ym = sum(x["map"] for x in tail) / n
        den = sum((x["epoch"] - xm) ** 2 for x in tail)
        slope = sum((x["epoch"] - xm) * (x["map"] - ym) for x in tail) / den if den else 0
        print(f"  imgsz {k:>5}: 末 {n} 轮 {sq[-n]['epoch']}–{sq[-1]['epoch']}, "
              f"斜率 {slope*100:+.4f}pp/轮  → 再跑 20 轮约 {slope*20*100:+.2f}pp")
    # 过拟合/欠拟合诊断: 决定"加数据"有没有用
    print("\n=== 数据是否已吃满 (末段斜率, 决定加数据有没有用) ===")
    print("  判读: train loss 仍显著下降 且 val loss 走平/上升 → 过拟合, 加数据有用;")
    print("        train 与 val loss 都走平 → 已收敛, 加数据的收益有限;")
    print("        val loss 仍下降但 mAP 走平 → 指标饱和, 加数据改变不了这个终点。\n")

    def slope(seq, col, n=10):
        tail = [s for s in seq[-n:] if s.get(col) is not None]
        if len(tail) < 3:
            return None
        xm = sum(s["epoch"] for s in tail) / len(tail)
        ym = sum(s[col] for s in tail) / len(tail)
        den = sum((s["epoch"] - xm) ** 2 for s in tail)
        return sum((s["epoch"] - xm) * (s[col] - ym) for s in tail) / den if den else None

    print(f"  {'imgsz':>6} {'末段轮次':>12} | {'train/box':>12} {'train/cls':>12} "
          f"{'val/box':>12} {'val/cls':>12} | {'mAP 斜率':>12}")
    for k in order:
        sq = runs[k]["seq"]
        if len(sq) < 12:
            continue
        n = min(10, len(sq))
        rng = f"{sq[-n]['epoch']}-{sq[-1]['epoch']}"
        cells = []
        for c in LOSS_COLS:
            s = slope(sq, c, n)
            cells.append(f"{s:>+12.4f}" if s is not None else f"{'--':>12}")
        sm = slope(sq, "map", n)
        print(f"  {k:>6} {rng:>12} |" + "".join(cells) +
              f" | {(sm*100 if sm is not None else 0):>+11.4f}pp")

    print("\n  泛化间隙 (val/box + val/cls) - (train/box + train/cls), 末段趋势:")
    for k in order:
        sq = runs[k]["seq"]
        if len(sq) < 12:
            continue
        def gap(s):
            ks = [c for c in LOSS_COLS]
            if any(s.get(c) is None for c in ks):
                return None
            return (s["val/box_loss"] + s["val/cls_loss"]) - (s["train/box_loss"] + s["train/cls_loss"])
        pts = [(s["epoch"], gap(s)) for s in sq if gap(s) is not None]
        if len(pts) >= 12:
            first = pts[max(0, len(pts) - 10)]
            last = pts[-1]
            print(f"    imgsz {k:>5}: {first[0]} 轮 {first[1]:+.3f} → {last[0]} 轮 {last[1]:+.3f} "
                  f"(变化 {last[1]-first[1]:+.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
