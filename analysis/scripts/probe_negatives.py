# -*- coding: utf-8 -*-
"""探测一批图像对某个 checkpoint 的"告警难度"。

为什么先做这一步
----------------
在把 FireSmoke-YOLO 的负样本接进训练集之前，必须先确认它们**对我们的模型确实是难点**。
D-Fire train 已有的 7,043 张背景图，模型在 D-Fire val 上的告警率是 **0/790** —— 即模型对
D-Fire 风格的背景本来就已经全对，再加权重没有意义。
所以判据是：这批新负样本的告警率是否**显著高于 0**、且接近目标域（FASDD 的 9.5–12.3%）。
若告警率接近 0，说明它们不是我们缺的那一类，接进去也没用。

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/scripts/probe_negatives.py \
    --weights outputs/paper-v1/dfire/baseline/seed0/weights/best.pt \
    --images datasets/FireSmoke-YOLO/images/val \
    --pattern 'NoFileSmoke*.jpg' \
    --thr-fire 0.3148 --thr-smoke 0.6060 \
    --out analysis/work/probe_fsneg
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="图像目录")
    ap.add_argument("--pattern", default="*.jpg", help="文件名 glob（默认全部）")
    ap.add_argument("--thr-fire", type=float, default=0.3148)
    ap.add_argument("--thr-smoke", type=float, default=0.6060)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 张（0=全部）")
    ap.add_argument("--out", default="analysis/work/probe")
    a = ap.parse_args()

    d = Path(a.images)
    files = sorted(d.glob(a.pattern))
    if a.limit:
        files = files[: a.limit]
    if not files:
        print("没有匹配的图像:", d / a.pattern)
        return 2

    from ultralytics import YOLO
    model = YOLO(a.weights)
    print(f"权重: {a.weights}")
    print(f"图像: {len(files)} 张来自 {d}  阈值 fire={a.thr_fire} smoke={a.thr_smoke}")

    # 必须分块: 一次性把上万个路径塞进 source=[...] 会静默退出（实测 6,533 张时发生）。
    CHUNK = 512
    max_fire, max_smoke = [], []
    for start in range(0, len(files), CHUNK):
        chunk = files[start:start + CHUNK]
        for r in model.predict(source=[str(p) for p in chunk], imgsz=a.imgsz, batch=a.batch,
                               device=a.device, conf=0.001, iou=0.7, max_det=300,
                               augment=False, half=False, rect=False, verbose=False, stream=True):
            cls = r.boxes.cls.cpu().tolist() if r.boxes is not None else []
            conf = r.boxes.conf.cpu().tolist() if r.boxes is not None else []
            f = [c for k, c in zip(cls, conf) if int(k) == 0]
            s = [c for k, c in zip(cls, conf) if int(k) == 1]
            max_fire.append(max(f) if f else 0.0)
            max_smoke.append(max(s) if s else 0.0)
        print(f"  进度 {min(start + CHUNK, len(files))}/{len(files)}", flush=True)
    if len(max_fire) != len(files):
        print(f"  !! 只处理了 {len(max_fire)}/{len(files)} 张", flush=True)

    n = len(max_fire)
    hit_f = [v >= a.thr_fire for v in max_fire]
    hit_s = [v >= a.thr_smoke for v in max_smoke]
    any_hit = [f or s for f, s in zip(hit_f, hit_s)]
    res = {
        "weights": a.weights, "images": str(d), "pattern": a.pattern, "n": n,
        "thresholds": {"fire": a.thr_fire, "smoke": a.thr_smoke},
        "alarm_fire_only": sum(1 for f, s in zip(hit_f, hit_s) if f and not s),
        "alarm_smoke_only": sum(1 for f, s in zip(hit_f, hit_s) if s and not f),
        "alarm_both": sum(1 for f, s in zip(hit_f, hit_s) if f and s),
        "alarm_any": sum(any_hit),
        "alarm_rate": sum(any_hit) / n,
        "score_fire": {"mean": st.fmean(max_fire), "p90": sorted(max_fire)[int(n * .9)],
                       "p99": sorted(max_fire)[int(n * .99)], "max": max(max_fire)},
        "score_smoke": {"mean": st.fmean(max_smoke), "p90": sorted(max_smoke)[int(n * .9)],
                        "p99": sorted(max_smoke)[int(n * .99)], "max": max(max_smoke)},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    L = [
        f"# 负样本难度探测\n",
        f"- 权重: `{a.weights}`",
        f"- 图像: {n} 张，`{d}` / `{a.pattern}`",
        f"- 冻结阈值: fire={a.thr_fire}, smoke={a.thr_smoke}（来自源域 D-Fire val 的 1% FPR）\n",
        f"| 指标 | 值 |",
        f"|---|---:|",
        f"| **任意类告警率** | **{res['alarm_rate']*100:.2f}%** ({res['alarm_any']}/{n}) |",
        f"| 仅 fire 触发 | {res['alarm_fire_only']} |",
        f"| 仅 smoke 触发 | {res['alarm_smoke_only']} |",
        f"| 两类都触发 | {res['alarm_both']} |",
        f"| fire 分数 均值 / p90 / p99 / max | {res['score_fire']['mean']:.4f} / {res['score_fire']['p90']:.4f} / {res['score_fire']['p99']:.4f} / {res['score_fire']['max']:.4f} |",
        f"| smoke 分数 均值 / p90 / p99 / max | {res['score_smoke']['mean']:.4f} / {res['score_smoke']['p90']:.4f} / {res['score_smoke']['p99']:.4f} / {res['score_smoke']['max']:.4f} |",
        "",
        "## 判据\n",
        "- **告警率接近 0%** → 这批图对模型不难，接进训练集没有意义（就像 D-Fire 自己的背景图：0/790）。",
        "- **告警率显著 >0、接近目标域的 9.5–12.3%** → 正是需要的那一类负样本，值得接入。",
        "- 用 330 张估计时，95% 置信区间约 ±5pp，所以 <3% 与 >8% 的差别是可靠的。",
    ]
    (out / "probe.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
