# -*- coding: utf-8 -*-
"""跨域尺寸分层：验证"fire 在 FASDD 上不降反升"是迁移能力还是尺寸构成差异。

动机
----
零样本结果：fire 的 AP50-95 在域内 0.42 → 跨域 FASDD 0.44（不降反升），而 smoke 从 0.53 崩到 0.24。
但两个域的**目标尺寸构成完全不同**（FASDD 在 640 下被降采样 0.52×，目标相对更大），
而"大目标本来就好检"。所以必须**在同一个尺寸档内**比较域内 vs 跨域：
  - 若逐档的跨域 AP 都 ≤ 域内 → "不降反升" 纯属尺寸构成效应，不代表迁移能力
  - 若在某些档上跨域反而更高 → 那才需要另找解释

做法
----
**完全离线**：直接读 `evaluate` 已产出的 `predictions.json`（归一化 xywh + conf），
补上每张图的原始尺寸算出 letterbox 后长边，复用 `stratified_ap.compute_stratified_ap`
（内含 COCO 面积范围忽略规则）。**不需要任何新的推理。**

用法
----
cd <repo>; export PYTHONPATH=$PWD
python analysis/cross_size_strata.py --out analysis/out_cross_strata
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from stratified_ap import compute_stratified_ap  # noqa: E402

EDGES = [0.0, 8.0, 16.0, 32.0, 64.0, float("inf")]
BAND_LABELS = ["<8px", "8-16px", "16-32px", "32-64px", ">=64px"]
CLASSES = {0: "fire", 1: "smoke"}

# 各域的 predictions 与图像根目录
# 注意：`id` 形如 "fasdd/test/xxx.jpg"，实际文件在 artifacts/data/<ds>/images/<split>/<name>
DOMAINS = {
    "dfire_val": ("analysis/out_zeroshot/src_p2_s0/predictions.json", "artifacts/data"),
    "fasdd_test": ("analysis/out_zeroshot/tgt_p2_s0/predictions.json", "artifacts/data"),
}


def resolve_image(img_root: str, img_id: str) -> Path:
    """把 predictions.json 里的 id 映射成真实图像路径。

    id 形如 "fasdd/test/xxx.jpg"；prepared 布局为
    <img_root>/<dataset>/images/<split>/<name>。找不到时退回 <img_root>/<id>。
    """
    root = Path(img_root)
    parts = str(img_id).split("/")
    cands = []
    if len(parts) >= 3:
        cands.append(root / parts[0] / "images" / parts[1] / "/".join(parts[2:]))
        cands.append(root / parts[0] / parts[1] / "/".join(parts[2:]))
    cands.append(root / img_id)
    for c in cands:
        if c.is_file():
            return c
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="analysis/out_cross_strata")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    size_cache: dict[str, tuple[int, int]] = {}

    def img_size(p: Path):
        k = str(p)
        if k not in size_cache:
            try:
                with Image.open(p) as im:
                    size_cache[k] = im.size
            except Exception:
                size_cache[k] = (0, 0)
        return size_cache[k]

    summary = {}
    for name in [d.strip() for d in a.domains.split(",") if d.strip()]:
        if name not in DOMAINS:
            print(f"[skip] 未知域 {name}")
            continue
        pred_rel, img_root = DOMAINS[name]
        pp = Path(pred_rel)
        if not pp.is_file():
            print(f"[skip] 缺 {pp}")
            continue
        art = json.loads(pp.read_text())
        gts, dets = [], []
        miss = 0
        first_resolved = None
        for i, im in enumerate(art["images"]):
            fp = resolve_image(img_root, im["id"])
            if first_resolved is None:
                first_resolved = (str(fp), fp.is_file())
            W, H = img_size(fp)
            if W <= 0 or H <= 0:
                miss += 1
                continue
            s = a.imgsz / max(W, H)
            for t in im["truth"]:
                cls, cx, cy, w, h = t[0], t[1], t[2], t[3], t[4]
                wpx, hpx = w * W, h * H
                long_n = max(wpx, hpx)
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                gts.append({"img_id": i, "cls": int(cls), "box": [x1, y1, x1 + wpx, y1 + hpx],
                            "size": long_n, "size_lb": long_n * s})
            for p in im["predictions"]:
                cls, cx, cy, w, h, conf = p[0], p[1], p[2], p[3], p[4], p[5]
                wpx, hpx = w * W, h * H
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                dets.append({"img_id": i, "cls": int(cls), "score": float(conf),
                             "box": [x1, y1, x1 + wpx, y1 + hpx]})
        print(f"[{name}] 图 {len(art['images'])}（尺寸缺失 {miss}），GT {len(gts)}，检测 {len(dets)}", flush=True)
        print(f"[{name}] 首个 id 解析: {first_resolved}", flush=True)
        if not gts:
            print(f"[{name}] !! GT 为空，路径解析有问题，跳过", flush=True)
            continue
        rows = compute_stratified_ap(dets, gts, EDGES, "long_lb", 2)
        summary[name] = {"n_gt": len(gts), "n_det": len(dets), "rows": rows}

    # ---------- 报告 ----------
    L = [f"# 跨域尺寸分层（逐档比较域内 vs 跨域）\n",
         f"- imgsz = {a.imgsz}，分档依据 = letterbox 后长边",
         "- 数据来自已有 predictions.json，**无新推理**；档外 GT 按 COCO 规则忽略",
         "- 配置：P2 seed0（域内 = D-Fire val，跨域 = FASDD test）\n"]

    for cls_id, cname in CLASSES.items():
        L.append(f"## {cname}\n")
        L.append("| 档位 | D-Fire val GT | 占比 | FASDD test GT | 占比 | 域内 AP50-95 | 跨域 AP50-95 | Δ |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        totals = {k: sum(r["n_gt"] for r in v["rows"] if r["class_id"] == cls_id)
                  for k, v in summary.items()}
        for b, lab in enumerate(BAND_LABELS):
            cells = {}
            for name in summary:
                rr = [r for r in summary[name]["rows"] if r["class_id"] == cls_id and r["band_index"] == b]
                cells[name] = rr[0] if rr else None
            nd = cells.get("dfire_val")
            nt = cells.get("fasdd_test")
            n_d = nd["n_gt"] if nd else 0
            n_t = nt["n_gt"] if nt else 0
            ad = nd["AP50-95"] if nd else None
            at = nt["AP50-95"] if nt else None
            d = (at - ad) * 100 if (ad is not None and at is not None) else None
            L.append(f"| {lab} | {n_d} | {(n_d/totals.get('dfire_val',1)*100):.1f}% | "
                     f"{n_t} | {(n_t/totals.get('fasdd_test',1)*100):.1f}% | "
                     f"{ad:.4f} | {at:.4f} | {d:+.2f}pp |" if ad is not None and at is not None
                     else f"| {lab} | {n_d} | — | {n_t} | — | — | — | — |")
        L.append("")

    L.append("## 判读\n")
    L.append("- 关键看**逐档的 Δ 列**。若每一档的 Δ 都 ≤ 0（或接近 0），说明")
    L.append("  「fire 跨域不降反升」完全是**尺寸构成效应**（FASDD 的目标更大），并非迁移能力更强。")
    L.append("- 同时对比两个域的**占比列**：如果 FASDD 在小档位上的占比明显低于 D-Fire，")
    L.append("  而小档位的 AP 又极低，那域内的整体 AP 就会被小目标拖低 —— 这是构成效应的直接来源。")
    L.append("- 只有出现**明显为正且量级可观**的 Δ 档位，才需要另找迁移性解释。\n")
    L.append("> 本报告不含任何新训练或新推理。")

    (out / "cross_size_strata.md").write_text("\n".join(L), encoding="utf-8")
    (out / "cross_size_strata.json").write_text(
        json.dumps({"imgsz": a.imgsz, "edges": EDGES, "domains": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
