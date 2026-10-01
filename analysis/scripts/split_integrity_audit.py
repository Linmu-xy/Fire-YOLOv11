#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_integrity_audit.py
========================
目标检测数据集 split 完整性审查 —— 回答三个问题:

  Q1  图像原生分辨率分布是什么? 训练 imgsz 属于上采样还是降采样?
      (直接决定 P2 高分辨率分支有没有"真细节"可吃)
  Q2  train/val/test 之间有多少近重复泄漏? 有多少是同源视频的相邻帧?
      (直接决定 val 指标是否被"背过的场景"抬高并饱和)
  Q3  经过 letterbox 之后, 有多少 GT 落在 P2(stride 4) 唯一能覆盖、
      P3(stride 8) 覆盖不到的尺寸段? 也就是 P2 的"结构性余量"。

本脚本对照 docs/EXPERIMENT_PROTOCOL.md 的既定口径实现, 并且明确遵守其边界:
  - 哈希距离只产生候选, 不证明同场景 / 同内容;
  - 不自动删除任何图像; 只输出候选清单与 clean/dup 子集清单, 供人工确认后按组重划分;
  - 不根据文件名猜测 scene ID; 文件名前缀聚类仅作诊断信号输出。

依赖: numpy, Pillow  (ultralytics 环境自带)

示例:
  python split_integrity_audit.py --data datasets/D-Fire --out audit_out --imgsz 640,800,960
  python split_integrity_audit.py --data artifacts/data/dfire/data.yaml --out audit_out --emit-manifests
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {"val": ("val", "valid", "validation"), "train": ("train",), "test": ("test",)}
# 常见 stride 下限假设: 目标长边至少要有 min_cells 个格子才被认为"可定位"
DEFAULT_MIN_CELLS = 2.0

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_DCT_CACHE: dict[int, np.ndarray] = {}


# --------------------------------------------------------------------------- #
# 哈希原语
# --------------------------------------------------------------------------- #
def dct_matrix(n: int) -> np.ndarray:
    if n not in _DCT_CACHE:
        k = np.arange(n, dtype=np.float64)[:, None]
        x = np.arange(n, dtype=np.float64)[None, :]
        m = np.cos(np.pi * (x + 0.5) * k / n) * np.sqrt(2.0 / n)
        m[0] = np.sqrt(1.0 / n)
        _DCT_CACHE[n] = m
    return _DCT_CACHE[n]


def dhash64(gray9x8: np.ndarray) -> int:
    """9x8 灰度 -> 64bit (水平相邻像素比较)。"""
    bits = (gray9x8[:, 1:] > gray9x8[:, :-1]).reshape(-1)
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def phash64(gray32: np.ndarray) -> int:
    """32x32 灰度 -> DCT -> 左上 8x8(去 DC) 与中位数比较 -> 64bit。"""
    d = dct_matrix(32)
    coef = d @ gray32.astype(np.float64) @ d.T
    low = coef[:8, :8].reshape(-1)
    med = np.median(low[1:])
    bits = (low > med).reshape(-1)
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def u64_to_bytes(v: int) -> bytes:
    return int(v).to_bytes(8, "big", signed=False)


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def ssim_gray(a32: np.ndarray, b32: np.ndarray, win: int = 8, step: int = 4) -> float:
    """32x32 灰度图上的窗口化 SSIM, 用于对哈希候选做二次确认。"""
    a = a32.astype(np.float64)
    b = b32.astype(np.float64)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    if a.shape[0] < win or a.shape[1] < win:
        return float("nan")
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        wa = sliding_window_view(a, (win, win))[::step, ::step]
        wb = sliding_window_view(b, (win, win))[::step, ::step]
    except Exception:
        return float("nan")
    n = win * win
    ma, mb = wa.mean((-1, -2)), wb.mean((-1, -2))
    va = wa.var((-1, -2)) * n / max(n - 1, 1)
    vb = wb.var((-1, -2)) * n / max(n - 1, 1)
    cov = ((wa - ma[..., None, None]) * (wb - mb[..., None, None])).mean((-1, -2)) * n / max(n - 1, 1)
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))
    return float(np.clip(s, -1.0, 1.0).mean())


# --------------------------------------------------------------------------- #
# 数据集发现
# --------------------------------------------------------------------------- #
def load_dataset_names(data_root: Path, yaml_path: Path | None) -> tuple[list[str], str]:
    """返回 (class_names, 描述用的路径字符串)。"""
    if yaml_path and yaml_path.exists():
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            names = cfg.get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda z: int(z))]
            if isinstance(names, list) and names:
                return [str(x) for x in names], str(yaml_path)
        except Exception:
            pass
    return ["fire", "smoke"], str(data_root)


def discover_splits(data_arg: str, yaml_path: Path | None) -> dict[str, dict[str, Path]]:
    """兼容两种常见布局, 自动探测每个 split 的 images / labels 目录。

    A) ultralytics 风格:  <root>/images/<split>  与  <root>/labels/<split>
    B) D-Fire 官方布局:   <root>/<split>/images  与  <root>/<split>/labels
    """
    roots: list[Path] = []
    cand = Path(data_arg)
    if cand.is_file() and cand.suffix.lower() in {".yaml", ".yml"}:
        yaml_path = cand
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(cand.read_text(encoding="utf-8"))
            p = cfg.get("path")
            if p:
                pp = Path(str(p))
                roots.append(pp if pp.is_absolute() else (cand.parent / pp).resolve())
        except Exception:
            pass
        roots.append(cand.parent)
        roots.append(cand.parent.parent)
    else:
        roots.append(cand)

    found: dict[str, dict[str, Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for split, aliases in SPLIT_ALIASES.items():
            if split in found:
                continue
            for alias in aliases:
                # 布局 A
                img_a, lbl_a = root / "images" / alias, root / "labels" / alias
                # 布局 B
                img_b, lbl_b = root / alias / "images", root / alias / "labels"
                if img_a.is_dir():
                    found[split] = {"images": img_a, "labels": lbl_a if lbl_a.is_dir() else Path("")}
                    break
                if img_b.is_dir():
                    found[split] = {"images": img_b, "labels": lbl_b if lbl_b.is_dir() else Path("")}
                    break
        if len(found) == len(SPLIT_ALIASES):
            break
    return found


def list_images(d: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            out.append(p)
    return out


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path or not path.exists():
        return boxes
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
            boxes.append((c, x, y, w, h))
    except Exception:
        pass
    return boxes


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def scan_images(splits, out_dir, limit: int, log) -> tuple[list[dict], list[dict], list[str]]:
    """遍历全部图像, 返回 (meta, size_rows, class_names_hint)。"""
    meta: list[dict] = []
    size_counter: Counter = Counter()
    size_rows: list[dict] = []
    split_order = [s for s in ("train", "val", "test") if s in splits]

    for split in split_order:
        img_dir = splits[split]["images"]
        paths = list_images(img_dir)
        if limit:
            paths = paths[:limit]
        log(f"[scan] {split}: {len(paths)} 张图 ...")
        t0 = time.time()
        for i, p in enumerate(paths, 1):
            try:
                with Image.open(p) as im:
                    W, H = im.size
                    g = im.convert("L")
                    g9x8 = np.asarray(g.resize((9, 8), Image.BILINEAR))
                    g32 = np.asarray(g.resize((32, 32), Image.BILINEAR))
            except Exception as e:
                log(f"[scan] 跳过无法解码的文件: {p} ({e})")
                continue
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            meta.append(
                {
                    "id": len(meta),
                    "split": split,
                    "path": str(p),
                    "stem": p.stem,
                    "w": W,
                    "h": H,
                    "sha256": sha,
                    "dhash": dhash64(g9x8),
                    "phash": phash64(g32),
                    "thumb": g32.astype(np.uint8),
                }
            )
            size_counter[(split, W, H)] += 1
            if i % 500 == 0:
                log(f"[scan]   {split} {i}/{len(paths)}  {time.time()-t0:.1f}s")

    for (split, W, H), n in sorted(size_counter.items(), key=lambda kv: -kv[1]):
        size_rows.append({"split": split, "width": W, "height": H, "count": n})
    return meta, size_rows, []


def banded_candidate_pairs(hashes: np.ndarray, bands: int, threshold: int, bucket_cb) -> None:
    """把 64bit 哈希切成 bands 段做桶索引。

    鸽笼原理保证: 汉明距离 <= bands-1 的两条哈希, 至少在 1 段上完全相同。
    因此 threshold <= bands-1 时本搜索对该阈值是完备的(不丢召回)。
    """
    w = 64 // bands
    n = len(hashes)
    if n == 0:
        return
    mask = np.uint64((1 << w) - 1)
    for b in range(bands):
        if b * w >= 64:
            break
        vals = (hashes >> np.uint64(b * w)) & mask
        order = np.argsort(vals, kind="stable")
        sv = vals[order]
        cuts = np.flatnonzero(sv[1:] != sv[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [n]))
        for s, e in zip(starts, ends):
            if e - s >= 2:
                bucket_cb(order[s:e])


def hamming_pairs_in_bucket(ids: np.ndarray, hbytes: np.ndarray, threshold: int, chunk: int = 512):
    """对桶内成员做分块汉明距离, 返回 (i<j 且距离<=threshold) 的全局 id 对。"""
    m = len(ids)
    B = hbytes[ids]
    for a0 in range(0, m, chunk):
        a1 = min(a0 + chunk, m)
        x = B[a0:a1, None, :] ^ B[None, :, :]
        d = _POPCOUNT[x].sum(-1)
        ii, jj = np.nonzero(d <= threshold)
        for ra, cj in zip(ii.tolist(), jj.tolist()):
            gi, gj = a0 + ra, cj
            if gi < gj:
                yield int(ids[gi]), int(ids[gj])


def find_near_duplicates(meta, args, log):
    hashes_d = np.array([m["dhash"] for m in meta], dtype=np.uint64)
    hashes_p = np.array([m["phash"] for m in meta], dtype=np.uint64)
    hb_d = hashes_d.view(np.uint8).reshape(-1, 8)
    hb_p = hashes_p.view(np.uint8).reshape(-1, 8)

    candidates: set[tuple[int, int]] = set()
    for tag, hashes, hbytes, thr in (
        ("dhash", hashes_d, hb_d, args.dhash_band_thr),
        ("phash", hashes_p, hb_p, args.phash_band_thr),
    ):
        log(f"[ndup] 桶搜索 {tag} (threshold={thr}, bands={args.bands}) ...")
        seen: set[tuple[int, int]] = set()

        def cb(ids, hbytes=hbytes, thr=thr, seen=seen):
            for pair in hamming_pairs_in_bucket(ids, hbytes, thr):
                if pair not in seen:
                    seen.add(pair)
                    candidates.add(pair)

        banded_candidate_pairs(hashes, args.bands, thr, cb)
        log(f"[ndup]   {tag} 候选对: {len(seen)}")

    log(f"[ndup] 候选对合计: {len(candidates)}, 开始二次确认 ...")
    near: list[tuple[int, int, int, int, float]] = []
    for gi, gj in sorted(candidates):
        dd = hamming(meta[gi]["dhash"], meta[gj]["dhash"])
        dp = hamming(meta[gi]["phash"], meta[gj]["phash"])
        if dd > args.dhash_max or dp > args.phash_max:
            continue
        s = ssim_gray(meta[gi]["thumb"], meta[gj]["thumb"])
        ok = (not np.isnan(s) and s >= args.ssim_min) or (dd <= args.dhash_tight and dp <= args.phash_tight)
        if ok:
            near.append((gi, gj, dd, dp, float(s) if not np.isnan(s) else -1.0))
    log(f"[ndup] 确认近重复对: {len(near)}")
    return near


def analyze_filename_groups(meta, near):
    """按文件名「字母前缀 + 数字 id」聚类, 报告跨 split 的同源候选与相邻帧。

    仅作诊断信号: 本函数不推断 scene ID, 也不作为划分依据。
    """
    def key(stem: str):
        letters = "".join(ch for ch in stem if ch.isalpha())
        digits = "".join(ch for ch in stem if ch.isdigit())
        return letters, int(digits) if digits else -1

    buckets = defaultdict(list)
    for m in meta:
        pre, num = key(m["stem"])
        buckets[pre].append((num, m))

    cross = []
    adjacent = 0
    for pre, items in buckets.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t[0])
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                na, ma = items[a]
                nb, mb = items[b]
                if ma["split"] == mb["split"]:
                    continue
                if na >= 0 and nb >= 0 and abs(na - nb) <= 3:
                    cross.append(
                        {"prefix": pre, "id_a": na, "id_b": nb, "split_a": ma["split"],
                         "split_b": mb["split"], "gap": abs(na - nb)}
                    )
    adjacent = sum(1 for c in cross if c["gap"] <= 3)
    return buckets, cross, adjacent


def analyze_sizes(meta, splits, imgszs, min_cells, log):
    """letterbox 后的 GT 长边分布, 按 split x class x 尺寸段统计。"""
    bands = [0, 8, 16, 32, 64, 128, 10**9]
    label_names = ["<8px", "8-16px", "16-32px", "32-64px", "64-128px", ">=128px"]
    rows = []
    for split in ("train", "val", "test"):
        if split not in splits:
            continue
        lbl_dir = splits[split]["labels"]
        if not lbl_dir:
            continue
        per_class = defaultdict(lambda: defaultdict(int))
        n_img_with_labels = 0
        for m in meta:
            if m["split"] != split:
                continue
            boxes = read_label(lbl_dir / f"{m['stem']}.txt")
            if boxes:
                n_img_with_labels += 1
            W, H = m["w"], m["h"]
            for cls, _x, _y, bw, bh in boxes:
                native_long = max(bw * W, bh * H)
                for imgsz in imgszs:
                    scale = imgsz / max(W, H)
                    v = native_long * scale
                    k = int(np.searchsorted(bands, v, side="right") - 1)
                    k = min(max(k, 0), len(label_names) - 1)
                    per_class[imgsz][(cls, label_names[k])] += 1
        for imgsz in imgszs:
            stride8_floor = 8 * min_cells
            stride4_floor = 4 * min_cells
            cls_ids = sorted({c for (c, _b) in per_class[imgsz]})
            for cls in cls_ids:
                total = sum(n for (c, _b), n in per_class[imgsz].items() if c == cls)
                if total == 0:
                    continue
                dist = {b: per_class[imgsz].get((cls, b), 0) for b in label_names}
                lt16 = sum(n for b, n in dist.items() if b in ("<8px", "8-16px"))
                lt8 = dist["<8px"]
                rows.append(
                    {
                        "split": split, "imgsz": imgsz, "class_id": cls, "n_boxes": total,
                        "frac_lt_stride8_floor": round(lt16 / total, 4),
                        "frac_lt_stride4_floor": round(lt8 / total, 4),
                        "stride8_floor_px": stride8_floor, "stride4_floor_px": stride4_floor,
                        **{f"frac_{b}": round(dist[b] / total, 4) for b in label_names},
                    }
                )
        log(f"[size] {split}: 有标注图像 {n_img_with_labels}")
    return rows, label_names


def build_verdicts(meta, global_size_rows, size_rows, ndup_stats, size_stats, args):
    V = []
    # Q1 原生分辨率 (跨 split 聚合)
    if global_size_rows:
        top = global_size_rows[0]
        total = sum(r["count"] for r in global_size_rows)
        uniq = len(global_size_rows)
        share = top["count"] / max(total, 1)
        max_long = max(max(r["width"], r["height"]) for r in global_size_rows)
        med_long = float(np.median([max(m["w"], m["h"]) for m in meta]))
        V.append(
            {
                "id": "Q1_native_resolution",
                "level": "info",
                "text": (
                    f"原生分辨率: {uniq} 种, 最常见 {top['width']}x{top['height']} "
                    f"({share*100:.1f}% 的图像); 长边中位数 {med_long:.0f}px, 最大 {max_long}px; "
                    f"共 {total} 张。"
                ),
            }
        )
        for imgsz in args.imgsz:
            scale = imgsz / max(top["width"], top["height"])
            if scale > 1.02:
                V.append(
                    {
                        "id": f"Q1_upsample_{imgsz}",
                        "level": "warn" if scale > 1.5 else "info",
                        "text": (
                            f"imgsz={imgsz} 相对最常见原生尺寸是 {scale:.2f}x 上采样。"
                            "上采样不产生新的光学细节, 它买到的是目标相对 stride 的格子数; "
                            "因此「加 P2」与「提 imgsz」在机制上是替代关系, 收益不可加。"
                        ),
                    }
                )
    # Q2 泄漏 (按查询侧 split 归一, 与已发表口径一致)
    for pair, st in ndup_stats["by_split_pair"].items():
        if st["a_total"] == 0:
            continue
        rate = st["rate"]
        lvl = "error" if rate >= 0.10 else ("warn" if rate >= 0.03 else "info")
        V.append(
            {
                "id": f"Q2_leak_{pair}",
                "level": lvl,
                "text": (
                    f"{pair}: {st['a_total']} 张中有 {st['a_matched']} 张 "
                    f"({rate*100:.1f}%) 在对方 split 里存在近重复。"
                    "(Barz & Denzler 认为 3.3% 已足以让泛化比较出现偏差)"
                ),
            }
        )
    if ndup_stats["within_train_exact"] > 0:
        V.append(
            {
                "id": "Q2_train_redundancy",
                "level": "warn",
                "text": (
                    f"train 内部有 {ndup_stats['within_train_exact']} 张图与同 split 内"
                    "另一张逐字节相同 (SHA256 冲突), 有效训练集小于名义大小。"
                ),
            }
        )
    if ndup_stats["adjacent_frame_pairs"] > 0:
        V.append(
            {
                "id": "Q2_video_frames",
                "level": "warn",
                "text": (
                    f"发现 {ndup_stats['adjacent_frame_pairs']} 组跨 split 的"
                    "同文件名前缀且编号相差 <=3 的图像; 高度提示是同一段视频的相邻帧被随机划分切开了。"
                ),
            }
        )
    # Q3 P2 结构性余量
    for row in size_stats:
        if row["split"] == "val":
            frac = row["frac_lt_stride8_floor"]
            lvl = "info" if frac >= 0.15 else "warn"
            V.append(
                {
                    "id": f"Q3_p2_headroom_{row['imgsz']}_c{row['class_id']}",
                    "level": lvl,
                    "text": (
                        f"val / imgsz={row['imgsz']} / class_id={row['class_id']}: "
                        f"{frac*100:.1f}% 的框 letterbox 后长边 < {row['stride8_floor_px']:.0f}px "
                        f"(P3 下限), 其中 {row['frac_lt_stride4_floor']*100:.1f}% 连 "
                        f"{row['stride4_floor_px']:.0f}px (P2 下限) 都不到。"
                    ),
                }
            )
    return V


def write_manifests(meta, ndup_stats, splits, out_dir, names, args, log):
    """输出 clean / dup 子集清单 (txt + 可直接给 ultralytics 的 yaml)。"""
    man = out_dir / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    dup_of = ndup_stats["dup_in_train"]  # 全局 id -> 是否在 train 里有近重复
    written = []

    for split in ("val", "test"):
        ids = [m["id"] for m in meta if m["split"] == split]
        if not ids:
            continue
        clean = [i for i in ids if not dup_of.get(i, False)]
        dup = [i for i in ids if dup_of.get(i, False)]
        for tag, sel in (("clean", clean), ("dup", dup), ("full", ids)):
            if not sel and tag != "full":
                continue
            txt = man / f"{split}_{tag}.txt"
            # 写绝对路径, 避免 ultralytics 依赖 cwd 解析清单
            txt.write_text(
                "\n".join(str(Path(meta[i]["path"]).resolve()) for i in sel) + "\n",
                encoding="utf-8",
            )
            yml = man / f"{split}_{tag}.yaml"
            data_root = splits[split]["images"].parent.parent.resolve()
            yml.write_text(
                "# 由 split_integrity_audit.py 生成; val 指向图像清单(绝对路径)\n"
                f"path: {data_root.as_posix()}\n"
                f"train: {splits[split]['images'].resolve().as_posix()}\n"
                f"val: {txt.resolve().as_posix()}\n"
                f"nc: {len(names)}\n"
                "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
                encoding="utf-8",
            )
            written.append({"subset": f"{split}_{tag}", "n_images": len(sel),
                            "txt": str(txt), "yaml": str(yml)})
            log(f"[manifest] {split}_{tag}: {len(sel)} 张 -> {yml.name}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="目标检测数据集 split 完整性审查 (原生分辨率 / 近重复泄漏 / P2 结构性余量)"
    )
    ap.add_argument("--data", required=True, help="数据集根目录, 或 data.yaml 路径")
    ap.add_argument("--out", default="audit_out", help="输出目录")
    ap.add_argument("--imgsz", default="640,800,960", help="要评估的输入分辨率, 逗号分隔")
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只处理前 N 张 (调试用)")
    ap.add_argument("--bands", type=int, default=8, help="哈希分桶段数, 保证召回的上限是 bands-1")
    ap.add_argument("--dhash-band-thr", type=int, default=5, help="dHash 桶搜索阈值")
    ap.add_argument("--phash-band-thr", type=int, default=5, help="pHash 桶搜索阈值")
    ap.add_argument("--dhash-max", type=int, default=18, help="确认阶段的 dHash 门限")
    ap.add_argument("--phash-max", type=int, default=12, help="确认阶段的 pHash 门限")
    ap.add_argument("--dhash-tight", type=int, default=4, help="免 SSIM 直接判定的 dHash 紧阈值")
    ap.add_argument("--phash-tight", type=int, default=6, help="免 SSIM 直接判定的 pHash 紧阈值")
    ap.add_argument("--ssim-min", type=float, default=0.88, help="SSIM 确认门限")
    ap.add_argument("--min-cells", type=float, default=DEFAULT_MIN_CELLS, help="长边至少几个 stride 格子才算可定位")
    ap.add_argument("--emit-manifests", action="store_true", help="输出 clean/dup 子集清单")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    args.imgsz = [int(v) for v in str(args.imgsz).replace(" ", "").split(",") if v]
    # 分桶搜索的完备性依赖鸽笼上界: 阈值必须 <= bands-1, 否则会静默漏检
    for nm, thr in (("dhash_band_thr", args.dhash_band_thr), ("phash_band_thr", args.phash_band_thr)):
        if thr > args.bands - 1:
            log(
                f"[fatal] {nm}={thr} 超过完备性上界 bands-1={args.bands - 1}; "
                f"请提高 --bands 或降低该阈值, 否则桶搜索会漏检。"
            )
            return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_arg = args.data
    if args.limit and args.emit_manifests:
        log("[warn] 同时使用了 --limit 与 --emit-manifests: 生成的子集清单只覆盖前 N 张, 不能用于正式评估。")
    yaml_path = Path(data_arg) if Path(data_arg).suffix.lower() in {".yaml", ".yml"} else None
    splits = discover_splits(data_arg, yaml_path)
    if not splits:
        log(f"[fatal] 在 {data_arg} 下没找到任何 images/labels 目录结构")
        return 2
    log(f"[init] 发现 split: {sorted(splits)}")
    names, desc = load_dataset_names(Path(data_arg), yaml_path)
    log(f"[init] 类别名: {names}")

    meta, size_rows, _ = scan_images(splits, out_dir, args.limit, log)
    if not meta:
        log("[fatal] 没有成功读取任何图像")
        return 2
    log(f"[scan] 共 {len(meta)} 张图")

    gcnt = Counter((m["w"], m["h"]) for m in meta)
    global_size_rows = [
        {"width": w, "height": h, "count": n}
        for (w, h), n in sorted(gcnt.items(), key=lambda kv: -kv[1])
    ]

    # ---- 精确重复 (SHA256) -------------------------------------------- #
    by_sha = defaultdict(list)
    for m in meta:
        by_sha[m["sha256"]].append(m["id"])
    exact_groups = {k: v for k, v in by_sha.items() if len(v) > 1}
    exact_pair_count = sum(len(v) * (len(v) - 1) // 2 for v in exact_groups.values())
    within_train_exact = 0
    for v in exact_groups.values():
        sp = {meta[i]["split"] for i in v}
        if sp == {"train"}:
            within_train_exact += len(v) - 1
    log(f"[exact] 完全重复组 {len(exact_groups)} 个, {exact_pair_count} 对; train 内冗余 {within_train_exact} 张")

    # ---- 近重复 ------------------------------------------------------- #
    near = find_near_duplicates(meta, args, log)
    near_pairs_by_id = defaultdict(set)
    for gi, gj, dd, dp, s in near:
        near_pairs_by_id[gi].add(gj)
        near_pairs_by_id[gj].add(gi)

    pair_stats: dict[str, dict] = {}
    split_ids = {sp: [m["id"] for m in meta if m["split"] == sp] for sp in ("train", "val", "test")}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a == b or not split_ids[a] or not split_ids[b]:
                continue
            b_set = set(split_ids[b])
            matched = [i for i in split_ids[a] if near_pairs_by_id.get(i, set()) & b_set]
            pair_stats[f"{a}->{b}"] = {
                "a_total": len(split_ids[a]),
                "a_matched": len(matched),
                "rate": round(len(matched) / len(split_ids[a]), 4),
                "b_total": len(split_ids[b]),
                "matched_ids": matched,
            }

    within = {}
    for sp in ("train", "val", "test"):
        ids = [m["id"] for m in meta if m["split"] == sp]
        idset = set(ids)
        hit = [i for i in ids if near_pairs_by_id.get(i, set()) & idset]
        within[sp] = {"n": len(ids), "n_with_near_dup": len(hit), "rate": round(len(hit) / max(len(ids), 1), 4)}

    _buckets, cross_groups, adjacent = analyze_filename_groups(meta, near)

    dup_in_train = {}
    train_ids = {m["id"] for m in meta if m["split"] == "train"}
    for i, js in near_pairs_by_id.items():
        if meta[i]["split"] in ("val", "test") and (js & train_ids):
            dup_in_train[i] = True
    # 精确重复也要算进去
    for v in exact_groups.values():
        sp = {meta[i]["split"] for i in v}
        if "train" in sp:
            for i in v:
                if meta[i]["split"] in ("val", "test"):
                    dup_in_train[i] = True

    ndup_stats = {
        "near_pair_count": len(near),
        "exact_group_count": len(exact_groups),
        "exact_pair_count": exact_pair_count,
        "within_train_exact": within_train_exact,
        "by_split_pair": pair_stats,
        "within_split": within,
        "cross_split_filename_groups": cross_groups[:200],
        "cross_split_filename_group_count": len(cross_groups),
        "adjacent_frame_pairs": adjacent,
        "dup_in_train": dup_in_train,
        "thresholds": {
            "bands": args.bands, "dhash_band_thr": args.dhash_band_thr,
            "phash_band_thr": args.phash_band_thr, "dhash_max": args.dhash_max,
            "phash_max": args.phash_max, "dhash_tight": args.dhash_tight,
            "phash_tight": args.phash_tight, "ssim_min": args.ssim_min,
        },
    }

    # ---- 尺寸分布 ----------------------------------------------------- #
    size_stats, band_names = analyze_sizes(meta, splits, args.imgsz, args.min_cells, log)

    verdicts = build_verdicts(meta, global_size_rows, size_rows, ndup_stats, size_stats, args)

    # ---- 落盘 --------------------------------------------------------- #
    manifests = []
    if args.emit_manifests:
        manifests = write_manifests(meta, ndup_stats, splits, out_dir, names, args, log)

    result = {
        "data": desc,
        "class_names": names,
        "n_images": len(meta),
        "native_resolution": global_size_rows,
        "native_resolution_by_split": size_rows,
        "integrity": {k: v for k, v in ndup_stats.items() if k != "dup_in_train"},
        "size_strata": {"bands": band_names, "min_cells": args.min_cells, "imgsz": args.imgsz, "rows": size_stats},
        "manifests": manifests,
        "verdicts": verdicts,
    }
    (out_dir / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / "near_duplicate_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_a", "split_a", "file_a", "id_b", "split_b", "file_b", "dhash", "phash", "ssim"])
        for gi, gj, dd, dp, s in near[:200000]:
            w.writerow([gi, meta[gi]["split"], Path(meta[gi]["path"]).name,
                        gj, meta[gj]["split"], Path(meta[gj]["path"]).name, dd, dp, round(s, 4)])

    # ---- markdown 报告 ------------------------------------------------ #
    L = []
    L.append("# 数据集 split 完整性审查\n")
    L.append(f"- 数据: `{desc}`")
    L.append(f"- 图像总数: {len(meta)}")
    L.append(f"- 类别: {names}\n")

    L.append("## 1. 原生分辨率 census\n")
    L.append("### 1.1 全局\n")
    L.append("| 宽x高 | 张数 | 占比 |")
    L.append("|---:|---:|---:|")
    gtot = sum(r["count"] for r in global_size_rows) or 1
    for r in global_size_rows[:25]:
        L.append(f"| {r['width']}x{r['height']} | {r['count']} | {r['count']/gtot*100:.1f}% |")
    L.append("")
    L.append("### 1.2 按 split\n")
    L.append("| split | 宽x高 | 张数 | 占比 |")
    L.append("|---|---:|---:|---:|")
    tot = sum(r["count"] for r in size_rows) or 1
    for r in size_rows[:25]:
        L.append(f"| {r['split']} | {r['width']}x{r['height']} | {r['count']} | {r['count']/tot*100:.1f}% |")
    L.append("")

    L.append("## 2. 重复与泄漏\n")
    L.append(f"- SHA256 完全重复组: {len(exact_groups)} 组 / {exact_pair_count} 对; train 内冗余 {within_train_exact} 张")
    L.append(f"- 近重复确认对: {len(near)} 对")
    L.append(f"- 跨 split 同源编号(相差<=3)的组: {len(cross_groups)} 组\n")
    L.append("| 方向 (左侧为查询侧) | 查询侧图数 | 命中数 | 占比 |")
    L.append("|---|---:|---:|---:|")
    for pair, st in pair_stats.items():
        L.append(f"| {pair} | {st['a_total']} | {st['a_matched']} | {st['rate']*100:.1f}% |")
    L.append("")
    L.append("| split 内部 | 图数 | 有近重复图数 | 占比 |")
    L.append("|---|---:|---:|---:|")
    for sp, st in within.items():
        L.append(f"| {sp} | {st['n']} | {st['n_with_near_dup']} | {st['rate']*100:.1f}% |")
    L.append("")

    L.append("## 3. letterbox 后的 GT 尺寸分层\n")
    L.append("| split | imgsz | class | 框数 | <P3下限 | <P2下限 | " + " | ".join(band_names) + " |")
    L.append("|---|---:|---:|---:|---:|---:|" + "---:|" * len(band_names))
    for r in size_stats:
        L.append(
            f"| {r['split']} | {r['imgsz']} | {r['class_id']} | {r['n_boxes']} | "
            f"{r['frac_lt_stride8_floor']*100:.1f}% | {r['frac_lt_stride4_floor']*100:.1f}% | "
            + " | ".join(f"{r[f'frac_{b}']*100:.1f}%" for b in band_names) + " |"
        )
    L.append("")

    L.append("## 4. 判定\n")
    for v in verdicts:
        tag = {"error": "**[严重]**", "warn": "**[注意]**", "info": "·"}.get(v["level"], "·")
        L.append(f"- {tag} `{v['id']}` {v['text']}")
    L.append("")
    if manifests:
        L.append("## 5. 已生成子集清单\n")
        L.append("| 子集 | 图数 | yaml |")
        L.append("|---|---:|---|")
        for m in manifests:
            L.append(f"| {m['subset']} | {m['n_images']} | `{m['yaml']}` |")
        L.append("")
    L.append("> 边界声明: 哈希距离只产生候选, 不证明同场景或同内容; 低纹理图像可能碰撞。")
    L.append("> 本审查不自动删除任何图像, 不据此推断官方 split 有误, 文件名聚类不等于 scene ID。\n")

    (out_dir / "audit.md").write_text("\n".join(L), encoding="utf-8")

    print("\n=== 判定摘要 ===")
    for v in verdicts:
        print(f"[{v['level']:5s}] {v['id']}: {v['text']}")
    print(f"\n输出目录: {out_dir.resolve()}")
    print("  audit.json / audit.md / near_duplicate_pairs.csv" + (" / manifests/*" if manifests else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
