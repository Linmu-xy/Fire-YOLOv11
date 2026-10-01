#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stratified_ap.py
================
按目标尺寸分层评估检测器 —— 直接检验"P2 到底有没有用"这个假设。

为什么需要它
------------
P2(stride 4) 相对 P3(stride 8) 的**唯一**独占作用域，是"经过 letterbox 之后
长边仍落在约 8–16px 之间的目标"。所以"P2 有没有用"等价于两个可测量的命题:

    (a) 数据里有没有足够多的目标落在这个尺寸段?          -> 用尺寸分布回答
    (b) P2 在落在这个尺寸段的目标上，是否真的比 P3 基线强? -> 用本脚本回答

如果 (b) 的答案是"不强"，那么无论换多高分辨率的数据集，P2 都不会有效 ——
因为机制本身被否证了。这一步不需要任何新数据集，也不需要重训。

口径说明（对齐 docs/EXPERIMENT_PROTOCOL.md 的边界）
--------------------------------------------------
- `--basis long_lb`（默认）: 按 **letterbox 之后的长边像素** 分层，命名形如 `lb8-16px`。
  这是回答"P2 有没有用"的正确口径，**不要**把它叫成 COCO APsmall。
- `--basis coco_area`: 按 **原图像素面积** 分层，且用 COCO 阈值(32²=1024, 96²=9216)，
  命名 `APs/APm/APl`，这才是可正当地称为 COCO APsmall 的口径。
- 面积范围之外的 GT **按 COCO 规则忽略**：与它匹配上的检测不算 FP，也不计入召回。
  这一点是关键 —— 否则大目标的预测会把小目标那一段的指标冲掉。

自校验
------
不带 `--strata-edges`（即只有"全部"一档）跑出来的 mAP50-95，应当与
`yolo val` 的数字接近。相差超过 1pp 就先查口径，别急着下结论。

前置条件
--------
本仓库的 checkpoint 里 pickle 了对自定义模块 `firesmoke` 的引用, 直接
`python analysis/stratified_ap.py ...` 会报 `ModuleNotFoundError: No module named 'firesmoke'`
(因为 sys.path[0] 变成了 analysis/)。**必须把仓库根目录加进 PYTHONPATH**:

    cd ~/Documents/yolo11/Fire-YOLOv11
    export PYTHONPATH=$PWD
    ~/miniconda3/envs/yolo11/bin/python analysis/stratified_ap.py \
        --weights outputs/paper-v1/dfire/p2/seed0/weights/best.pt \
        --data artifacts/data/dfire/data.yaml --split val --imgsz 640 --edges 8,16,32

用法
----
# 1) 先测尺寸分布（零成本，用已有脚本）
python gt_size_distribution.py --data artifacts/data/dfire/data.yaml --imgsz 640

# 2) 分层评估，P2 与同协议基线各跑一次
python stratified_ap.py --weights runs/p2_seed0/weights/best.pt \
    --data artifacts/data/dfire/data.yaml --split val --imgsz 640 \
    --edges 8,16,32 --out eval_p2

python stratified_ap.py --weights runs/baseline_640_seed0/weights/best.pt \
    --data artifacts/data/dfire/data.yaml --split val --imgsz 640 \
    --edges 8,16,32 --out eval_base
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # 0.50:0.05:0.95
COCO_AREA_EDGES = [0.0, 32.0**2, 96.0**2, float("inf")]


# --------------------------------------------------------------------------- #
# 核心评估（纯函数，便于单独测试）
# --------------------------------------------------------------------------- #
def iou_matrix(dets: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """dets:(D,4), gts:(G,4), 均为 xyxy，返回 (D,G) 的 IoU。"""
    if dets.size == 0 or gts.size == 0:
        return np.zeros((len(dets), len(gts)), dtype=np.float64)
    d = dets[:, None, :]
    g = gts[None, :, :]
    ix1 = np.maximum(d[..., 0], g[..., 0])
    iy1 = np.maximum(d[..., 1], g[..., 1])
    ix2 = np.minimum(d[..., 2], g[..., 2])
    iy2 = np.minimum(d[..., 3], g[..., 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ad = np.clip(d[..., 2] - d[..., 0], 0, None) * np.clip(d[..., 3] - d[..., 1], 0, None)
    ag = np.clip(g[..., 2] - g[..., 0], 0, None) * np.clip(g[..., 3] - g[..., 1], 0, None)
    union = ad + ag - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def voc_ap_101(rec: np.ndarray, prec: np.ndarray) -> float:
    """COCO 式 all-point 插值 AP。"""
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_image(dets: np.ndarray, scores: np.ndarray,
                gt_boxes: np.ndarray, gt_counted: np.ndarray,
                iou_thr: float) -> np.ndarray:
    """单图单类的贪心匹配。返回每个检测的标记: 1=TP, 0=FP, -1=忽略。

    gt_counted[i] 为 True 表示该 GT 属于当前尺寸档(计入召回);
    为 False 表示同类别但不在该档 —— 按 COCO 规则忽略, 匹配上它既不算 TP 也不算 FP。
    """
    n = len(dets)
    flags = np.zeros(n, dtype=np.int8)
    if n == 0:
        return flags
    if len(gt_boxes) == 0:
        return flags  # 全 FP
    iou = iou_matrix(dets, gt_boxes)
    order = np.argsort(-scores, kind="stable")
    used = np.zeros(len(gt_boxes), dtype=bool)
    for k in order:
        row = np.where(used, -1.0, iou[k])
        j = int(np.argmax(row)) if row.size else -1
        if j >= 0 and row[j] >= iou_thr:
            used[j] = True
            flags[k] = 1 if gt_counted[j] else -1
        else:
            flags[k] = 0
    return flags


def compute_stratified_ap(all_dets: list[dict], all_gts: list[dict],
                          edges: list[float], basis: str, n_classes: int) -> list[dict]:
    """all_dets: [{img_id, cls, score, box(native xyxy)}]
       all_gts:  [{img_id, cls, box(native xyxy), size(native px 长边), size_lb(letterbox 后长边)}]

       返回每个尺寸档 × 每个类别的指标。
    """
    n_bands = len(edges) - 1
    out = []

    def band_of(g):
        v = g["size_lb"] if basis == "long_lb" else (g["size"] ** 2)
        for b in range(n_bands):
            if edges[b] <= v < edges[b + 1]:
                return b
        return None

    gt_band = [band_of(g) for g in all_gts]

    dets_by_img_cls = defaultdict(list)
    for d in all_dets:
        dets_by_img_cls[(d["img_id"], d["cls"])].append(d)
    gts_by_img_cls = defaultdict(list)
    for g, b in zip(all_gts, gt_band):
        gts_by_img_cls[(g["img_id"], g["cls"])].append((g, b))

    # 预处理每张图每个类的数组
    prep = {}
    for key, dl in dets_by_img_cls.items():
        dl = sorted(dl, key=lambda x: -x["score"])
        prep.setdefault(key, {})["dets"] = (
            np.array([x["box"] for x in dl], dtype=np.float64),
            np.array([x["score"] for x in dl], dtype=np.float64),
        )
    for key, gl in gts_by_img_cls.items():
        prep.setdefault(key, {})["gts"] = (
            np.array([g["box"] for g, _ in gl], dtype=np.float64),
            np.array([b for _, b in gl], dtype=object),
        )

    band_names = []
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        if basis == "long_lb":
            band_names.append(f"lb{lo:.0f}-{hi:.0f}px" if np.isfinite(hi) else f"lb>={lo:.0f}px")
        else:
            band_names.append(f"area{lo:.0f}-{hi:.0f}" if np.isfinite(hi) else f"area>={lo:.0f}")
    if basis == "coco_area":
        band_names = ["APs", "APm", "APl"] + band_names[3:]

    for b in range(n_bands):
        for c in range(n_classes):
            n_gt = sum(1 for g, gb in zip(all_gts, gt_band) if g["cls"] == c and gb == b)
            aps = []
            for thr in IOU_THRESHOLDS:
                recs, precs = [], []
                total_tp, total_fp = 0, 0
                records = []
                for (img_id, cls), parts in prep.items():
                    if cls != c:
                        continue
                    if "dets" not in parts:
                        continue
                    dets, scores = parts["dets"]
                    if "gts" in parts:
                        gboxes, gbands = parts["gts"]
                        counted = np.array([gb == b for gb in gbands], dtype=bool)
                    else:
                        gboxes = np.zeros((0, 4))
                        counted = np.zeros(0, dtype=bool)
                    flags = match_image(dets, scores, gboxes, counted, thr)
                    for sc, fl in zip(scores, flags):
                        if fl == -1:
                            continue
                        records.append((float(sc), int(fl)))
                if n_gt == 0:
                    continue
                records.sort(key=lambda x: -x[0])
                if records:
                    tp = np.array([r[1] for r in records], dtype=np.float64)
                    tp_c = np.cumsum(tp)
                    fp_c = np.cumsum(1.0 - tp)
                    rec = tp_c / max(n_gt, 1)
                    prec = tp_c / np.maximum(tp_c + fp_c, 1e-12)
                    aps.append(voc_ap_101(rec, prec))
                else:
                    aps.append(0.0)
            aps_arr = np.array([a for a in aps]) if aps else np.zeros(0)
            out.append({
                "band": band_names[b], "band_index": b, "class_id": c,
                "n_gt": n_gt,
                "AP50": float(aps_arr[0]) if aps_arr.size else None,
                "AP50-95": float(aps_arr.mean()) if aps_arr.size else None,
                "AP75": float(aps_arr[5]) if aps_arr.size > 5 else None,
            })
    return out


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def find_split_dir(data_arg: str, split: str) -> tuple[Path | None, Path | None]:
    cand = Path(data_arg)
    roots = []
    yml = None
    if cand.is_file() and cand.suffix.lower() in {".yaml", ".yml"}:
        yml = cand
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(cand.read_text(encoding="utf-8"))
            p = cfg.get("path")
            if p:
                pp = Path(str(p))
                roots.append(pp if pp.is_absolute() else (cand.parent / pp).resolve())
            v = cfg.get(split)
            if v:
                vp = Path(str(v))
                if not vp.is_absolute():
                    for r in list(roots) + [cand.parent, cand.parent.parent]:
                        if (r / vp).exists():
                            vp = (r / vp).resolve()
                            break
                # 图像清单模式: data.yaml 里 split 指向一个 .txt
                if vp.is_file() and vp.suffix.lower() == ".txt":
                    imgs = [Path(x.strip()) for x in vp.read_text(encoding="utf-8").splitlines() if x.strip()]
                    imgs = [p if p.is_absolute() else (vp.parent / p) for p in imgs]
                    imgs = [p for p in imgs if p.suffix.lower() in IMAGE_EXT and p.exists()]
                    ld = derive_labels_dir(imgs[0]) if imgs else None
                    return imgs, ld
                if vp.is_dir():
                    return list_images(vp), find_labels_for(vp)
        except Exception:
            pass
        roots += [cand.parent, cand.parent.parent]
    else:
        roots.append(cand)

    for r in roots:
        for img_d, lbl_d in ((r / "images" / split, r / "labels" / split),
                             (r / split / "images", r / split / "labels")):
            if img_d.is_dir():
                return list_images(img_d), (lbl_d if lbl_d.is_dir() else None)
    return None, None


def list_images(d: Path) -> list[Path]:
    return [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXT]


def derive_labels_dir(img_path: Path) -> Path | None:
    """从图像路径推导标签目录: .../images/<split>/x.jpg -> .../labels/<split>"""
    parts = list(img_path.parts)
    if "images" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("images")
    cand = Path(*parts[:i]) / "labels" / Path(*parts[i + 1:]).parent
    return cand if cand.is_dir() else None


def find_labels_for(img_dir: Path) -> Path | None:
    cand = img_dir.parent / "labels" / img_dir.name
    if cand.is_dir():
        return cand
    cand = img_dir.parent.parent / "labels"
    return cand if cand.is_dir() else None


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not path or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 5:
            try:
                out.append((int(float(p[0])), *[float(x) for x in p[1:5]]))
            except Exception:
                pass
    return out


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description="按目标尺寸分层评估检测器")
    ap_.add_argument("--weights", required=True)
    ap_.add_argument("--data", required=True, help="data.yaml 或数据集根目录")
    ap_.add_argument("--split", default="val")
    ap_.add_argument("--imgsz", type=int, default=640)
    ap_.add_argument("--batch", type=int, default=16)
    ap_.add_argument("--chunk", type=int, default=128,
                     help="每次交给 predict 的图像数; 太大会被 ultralytics 预加载成一个巨 batch 导致显存爆掉")
    ap_.add_argument("--device", default="")
    ap_.add_argument("--conf", type=float, default=0.001, help="低阈值以拿到完整 PR 曲线")
    ap_.add_argument("--edges", default="8,16,32",
                     help="letterbox 后长边的分档边界, 逗号分隔; 留空则只算「全部」一档")
    ap_.add_argument("--basis", default="long_lb", choices=["long_lb", "coco_area"])
    ap_.add_argument("--limit", type=int, default=0)
    ap_.add_argument("--expect", type=float, default=None,
                     help="已知的 yolo val mAP50-95, 用于自校验口径是否一致")
    ap_.add_argument("--out", default="stratified_out")
    a = ap_.parse_args(argv)

    images, labels = find_split_dir(a.data, a.split)
    if not images:
        print(f"[fatal] 在 {a.data} 下找不到 split={a.split} 的图像", file=sys.stderr)
        return 2
    if labels is None:
        print("[fatal] 找不到对应 labels 目录, 无法做分层评估", file=sys.stderr)
        return 2
    if a.limit:
        images = images[:a.limit]
    print(f"[init] {a.split}: {len(images)} 张图, labels={labels}")

    # GT + 尺寸
    all_gts = []
    from PIL import Image
    for i, p in enumerate(images):
        try:
            with Image.open(p) as im:
                W, H = im.size
        except Exception:
            continue
        scale = a.imgsz / max(W, H)
        for cls, x, y, bw, bh in read_label(labels / f"{p.stem}.txt"):
            w_px, h_px = bw * W, bh * H
            long_native = max(w_px, h_px)
            x1 = (x - bw / 2) * W
            y1 = (y - bh / 2) * H
            all_gts.append({
                "img_id": i, "cls": cls,
                "box": [x1, y1, x1 + w_px, y1 + h_px],
                "size": long_native, "size_lb": long_native * scale,
            })
        if (i + 1) % 1000 == 0:
            print(f"[gt]   {i+1}/{len(images)}")
    print(f"[gt] 共 {len(all_gts)} 个框")

    # 推理
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[fatal] 无法导入 ultralytics: {e}", file=sys.stderr)
        return 2
    print("[predict] 开始推理 ...")
    model = YOLO(a.weights)
    all_dets = []
    total = len(images)
    chunk = max(1, int(a.chunk))
    for base in range(0, total, chunk):
        part = images[base:base + chunk]
        kw = dict(source=[str(p) for p in part], imgsz=a.imgsz, conf=a.conf,
                  iou=0.7, batch=a.batch, save=False, verbose=False, stream=True)
        if a.device:
            kw["device"] = a.device
        for k, r in enumerate(model.predict(**kw)):
            i = base + k  # 列表源保持输入顺序
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(xyxy, confs, clss):
                all_dets.append({"img_id": i, "cls": int(c), "score": float(s),
                                 "box": [float(v) for v in b]})
        print(f"[predict]   {min(base + chunk, total)}/{total}")
    print(f"[predict] {len(all_dets)} 个检测")

    # 分档
    if a.basis == "coco_area":
        edges = list(COCO_AREA_EDGES)
    else:
        edges = [0.0] + [float(v) for v in str(a.edges).split(",") if v.strip()] + [float("inf")]
    n_classes = max([g["cls"] for g in all_gts] + [d["cls"] for d in all_dets] + [0]) + 1

    rows = compute_stratified_ap(all_dets, all_gts, edges, a.basis, n_classes)

    # 自校验: 全部目标、无尺寸过滤
    if a.expect is not None and len(edges) > 2:
        all_rows = compute_stratified_ap(all_dets, all_gts, [0.0, float("inf")], a.basis, n_classes)
        vals = [r["AP50-95"] for r in all_rows if r["AP50-95"] is not None]
        mine = float(np.mean(vals)) if vals else float("nan")
        print(f"\n[自校验] 本脚本全量 mAP50-95 = {mine:.5f}; 你给的 yolo val = {a.expect:.5f}; "
              f"差 {abs(mine - a.expect)*100:.3f}pp")
        if abs(mine - a.expect) > 0.01:
            print("[自校验] 差值 >1pp, 先核对 conf/iou/imgsz/rect 口径, 再解读分档结果。")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stratified.json").write_text(
        json.dumps({"weights": a.weights, "imgsz": a.imgsz, "basis": a.basis,
                    "edges": edges, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    with (out_dir / "stratified.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["band"])
        w.writeheader()
        w.writerows(rows)

    L = [f"# 分层评估: {Path(a.weights).parent.name}", "",
         f"- 权重: `{a.weights}`", f"- imgsz: {a.imgsz}, basis: `{a.basis}`",
         f"- 分档边界: {edges}", f"- GT 框: {len(all_gts)}, 检测: {len(all_dets)}", "",
         "| 档位 | class | GT 数 | AP50 | AP50-95 | AP75 |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        f2 = lambda v: "-" if v is None else f"{v:.5f}"
        L.append(f"| {r['band']} | {r['class_id']} | {r['n_gt']} | "
                 f"{f2(r['AP50'])} | {f2(r['AP50-95'])} | {f2(r['AP75'])} |")
    L.append("")
    L.append("## 怎么读")
    L.append("")
    L.append("- 把 P2 和同协议基线的两张表并排放: **只在 `lb8-16px` 这一档上比较**。")
    L.append("  P2 的机制预言它在这一档上明显更强; 若它在这一档上也不强, 机制被否证。")
    L.append("- 若某一档 `GT 数` 很小(比如几十个), 该档的数字不可信, 不要据此下结论 ——")
    L.append("  这本身就说明**当前数据集不足以检验 H5**, 需要换尺度分布更合适的数据集。")
    L.append("- `lb8-16px` 以下的档中若 GT 极少, 说明 P2 的作用域是空集, 收益趋近 0 是预期的。")
    L.append("")
    L.append("> 本评估不改动权重、不产生新训练结果。分档口径见脚本头部说明。")
    (out_dir / "stratified.md").write_text("\n".join(L), encoding="utf-8")

    print("\n=== 分层结果 ===")
    for r in rows:
        f2 = lambda v: "-" if v is None else f"{v:.5f}"
        print(f"[{r['band']:>12s}] cls={r['class_id']} n_gt={r['n_gt']:>6d} "
              f"AP50={f2(r['AP50'])} AP50-95={f2(r['AP50-95'])}")
    print(f"\n输出: {out_dir.resolve()}  (stratified.json / stratified.csv / stratified.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
