# -*- coding: utf-8 -*-
"""图像原生分辨率普查 —— 回答"提 imgsz 到底是在插值还是在回收细节"。

为什么这个比 GT 框尺寸分布更关键
--------------------------------
letterbox 的缩放系数 = imgsz / max(W, H)。所以:
  - 若原生长边 < imgsz  → 上采样, 输入比原生多出的像素全是插值, 新信息为 0
  - 若原生长边 > imgsz  → 降采样, 输入丢掉了原生的一部分像素, 那部分是"被扔掉的信息"
因此"提 imgsz 有没有用"等价于"当前 imgsz 相对原生分辨率处在哪一侧"。

用法
----
python analysis/image_resolution_census.py \
    --ds dfire:artifacts/data/dfire/data.yaml \
    --ds fasdd:artifacts/data/fasdd/data.yaml \
    --imgsz 640,960,1280 --out analysis/out_res_census
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BUCKETS = [(0, 416), (417, 640), (641, 960), (961, 1280), (1281, 1920), (1921, 10**9)]
BUCKET_LABELS = ["<=416", "417-640", "641-960", "961-1280", "1281-1920", ">1920"]


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def read_yaml_splits(yaml_path: Path):
    """取出 path / train / val / test。注意 path 也要解析，否则根目录会被拼两次。"""
    out, cur = {}, None
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if line.startswith(("path:", "train:", "val:", "test:")):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def images_of(root: Path, entry: str):
    p = Path(entry)
    if not p.is_absolute():
        p = root / p
    if p.is_file() and p.suffix.lower() == ".txt":
        res = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            q = Path(line)
            q = q if q.is_absolute() else (p.parent / q)
            if q.suffix.lower() in IMAGE_EXT and q.exists():
                res.append(q)
        return res
    if p.is_file() and p.suffix.lower() in IMAGE_EXT:
        return [p]
    if p.is_dir():
        return [q for q in sorted(p.rglob("*")) if q.suffix.lower() in IMAGE_EXT]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", action="append", required=True, metavar="NAME:YAML")
    ap.add_argument("--imgsz", default="640,960,1280")
    ap.add_argument("--out", default="out_res_census")
    a = ap.parse_args()

    imgszs = [int(x) for x in a.imgsz.split(",") if x.strip()]
    results = {}

    for spec in a.ds:
        name, _, ypath = spec.partition(":")
        yaml_path = Path(ypath)
        if not yaml_path.is_file():
            print(f"[warn] 找不到 {yaml_path}")
            continue
        cfg = read_yaml_splits(yaml_path)
        root = Path(cfg.get("path") or str(yaml_path.parent))
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
        print(f"[{name}] root = {root}")
        per_split = {}
        for split in ("train", "val", "test"):
            entry = cfg.get(split)
            if not entry:
                continue
            paths = images_of(root, entry)
            if not paths:
                continue
            from PIL import Image
            dims = []
            bad = 0
            for q in paths:
                try:
                    with Image.open(q) as im:
                        w, h = im.size
                    if w > 0 and h > 0:
                        dims.append((w, h))
                except Exception:
                    bad += 1
            if not dims:
                continue
            per_split[split] = dims
            print(f"  {split}: {len(dims)} 张可读" + (f", {bad} 张读取失败" if bad else ""))
        results[name] = per_split

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = ["# 图像原生分辨率普查\n"]

    # 1. 分位数
    L.append("## 1. 原生尺寸分位数\n")
    L.append("| 数据集 | split | 图像数 | 长边 p5 | p25 | **p50** | p75 | p95 | 最大 | 宽高比 p50 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, sp in results.items():
        for split, dims in sp.items():
            longs = sorted(max(w, h) for w, h in dims)
            ars = sorted(max(w, h) / min(w, h) for w, h in dims)
            L.append(f"| {name} | {split} | {len(dims)} | {pct(longs,0.05):.0f} | {pct(longs,0.25):.0f} | "
                     f"**{pct(longs,0.50):.0f}** | {pct(longs,0.75):.0f} | {pct(longs,0.95):.0f} | "
                     f"{longs[-1]:.0f} | {pct(ars,0.5):.2f} |")
    L.append("")

    # 2. 分桶
    L.append("## 2. 长边分布（桶）\n")
    hdr = "| 数据集 | split | 图像数 | " + " | ".join(BUCKET_LABELS) + " |"
    L.append(hdr)
    L.append("|---|---|---:|" + "---:|" * len(BUCKET_LABELS))
    for name, sp in results.items():
        for split, dims in sp.items():
            cnt = [0] * len(BUCKETS)
            for w, h in dims:
                lo = max(w, h)
                for i, (a_, b_) in enumerate(BUCKETS):
                    if a_ <= lo <= b_:
                        cnt[i] += 1
                        break
            n = len(dims)
            L.append(f"| {name} | {split} | {n} | " +
                     " | ".join(f"{c/n*100:.1f}%" for c in cnt) + " |")
    L.append("")

    # 3. letterbox 行为
    L.append("## 3. 各 imgsz 下 letterbox 是上采样还是降采样\n")
    L.append("| 数据集 | split | imgsz | 上采样占比 | 约等于1 | **降采样占比** | scale p50 | "
             "输入像素/原生像素 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    verdict_rows = []
    for name, sp in results.items():
        for split, dims in sp.items():
            if split != "train":
                continue
            for iz in imgszs:
                scales = sorted(iz / max(w, h) for w, h in dims)
                n = len(scales)
                up = sum(1 for s in scales if s > 1.02) / n
                eq = sum(1 for s in scales if 0.98 <= s <= 1.02) / n
                dn = sum(1 for s in scales if s < 0.98) / n
                # 输入像素/原生像素: letterbox 后总输入像素 / 原生总像素
                nat = sum(w * h for w, h in dims)
                inp = sum((math.floor(max(w, h) * (iz / max(w, h)))) ** 2 for w, h in dims)
                ratio = inp / nat
                L.append(f"| {name} | {split} | {iz} | {up*100:.1f}% | {eq*100:.1f}% | "
                         f"**{dn*100:.1f}%** | {pct(scales,0.5):.3f} | {ratio:.2f}× |")
                verdict_rows.append((name, split, iz, up, dn, pct(scales, 0.5), ratio))
    L.append("")

    L.append("## 4. 判读\n")
    L.append("以 `输入像素/原生像素` 为主判据（它综合了整条分布，不像占比那样是二分法）：")
    L.append("")
    L.append("- `< 0.9×`：**净丢信息**。提高 imgsz 是在回收被丢弃的细节，预期收益大。")
    L.append("- `0.9–1.3×`：**接近原生**。处于收益递减区，再往上主要是插值。")
    L.append("- `> 1.3×`：**净插值**。输入像素已远超原生，新信息为 0，收益上限低。")
    L.append("")
    for name, split, iz, up, dn, s50, ratio in verdict_rows:
        if ratio < 0.9:
            tag, v = "净丢信息", "提高 imgsz 还有真实细节可回收"
        elif ratio <= 1.3:
            tag, v = "接近原生", "已在收益递减区"
        else:
            tag, v = "净插值", "纯插值, 收益上限低"
        L.append(f"- `{name}` / {split} @ imgsz {iz}：**{tag}**（ratio {ratio:.2f}×，"
                 f"scale p50 {s50:.2f}，降采样 {dn*100:.0f}% / 上采样 {up*100:.0f}%）—— {v}。")
    L.append("")
    L.append("> `输入像素/原生像素` > 1 表示总输入像素多于原生（插值放大）；< 1 表示净丢信息。")
    L.append(">")
    L.append("> 注意这是**整条分布的聚合比值**。分辨率分布若很宽，同一个 imgsz 下会同时存在")
    L.append("> 大幅上采样与大幅降采样，此时「停止丢信息」的 imgsz ≈ 原生长边中位数。")
    L.append("")

    (out / "res_census.md").write_text("\n".join(L), encoding="utf-8")
    (out / "res_census.json").write_text(json.dumps(
        {n: {s: len(d) for s, d in sp.items()} for n, sp in results.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"\n输出: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
