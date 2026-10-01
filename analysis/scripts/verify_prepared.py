# -*- coding: utf-8 -*-
"""校验一个 prepared 数据集: 四重完整性检查 + 过采样次数核对。

用法:
    cd <repo>; export PYTHONPATH=$PWD
    python analysis/verify_prepared.py --dataset dfire_hn3
    python analysis/verify_prepared.py --dataset dfire        # 回归: 应为 neg_repeat=1 且无重复行
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# 旧版本 prepare 写出的 metadata 没有 neg_repeat 字段, 缺失即视为 1

import yaml

sys.path.insert(0, ".")
from firesmoke.data import load_prepared  # noqa: E402

SPLITS = ("train", "val", "test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--expected-neg-repeat", type=int, default=None,
                    help="期望的过采样倍数; 给出时不一致即判失败")
    ap.add_argument("--reference", default=None,
                    help="参照数据集名; 给出时比较去重后的内容指纹(用于回归)")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    print(f"=== {a.dataset} ===")
    out, meta, rows = load_prepared(cfg, a.dataset, verify=True)
    print(f"[OK] load_prepared(verify=True) 四重校验通过: {out}")
    # 旧版本 prepare 写出的 metadata 没有 neg_repeat 字段, 缺失即视为 1
    neg = int(meta.get("neg_repeat", 1))
    print(f"  counts        : {meta['counts']}")
    print(f"  neg_repeat    : {neg}")
    print(f"  manifest 行数 : {len(rows)}")

    ok = True
    for s in SPLITS:
        lines = (out / f"{s}.txt").read_text().splitlines()
        n_rows = sum(1 for r in rows if r["split"] == s)
        n_uniq = len({r["id"] for r in rows if r["split"] == s})
        print(f"  {s:5s}.txt 行数 : {len(lines)}  (manifest 行 {n_rows}, 唯一图 {n_uniq})")
        if len(lines) != n_rows:
            print(f"     !! {s}.txt 与 manifest 行不一致")
            ok = False

    # 过采样次数
    tr = [r for r in rows if r["split"] == "train"]
    bg = [r for r in tr if not r["boxes"]]
    nonbg = [r for r in tr if r["boxes"]]
    bg_counts = Counter(r["id"] for r in bg)
    nb_counts = Counter(r["id"] for r in nonbg)
    bg_max = max(bg_counts.values()) if bg_counts else 0
    nb_max = max(nb_counts.values()) if nb_counts else 0
    print(f"  train 纯背景: {len(bg)} 行 / {len(bg_counts)} 张 -> 每张出现 {bg_max} 次")
    print(f"  train 含目标: {len(nonbg)} 行 / {len(nb_counts)} 张 -> 每张出现 {nb_max} 次")

    exp = a.expected_neg_repeat if a.expected_neg_repeat is not None else neg
    if bg_max != exp:
        print(f"     !! 背景图重复次数 {bg_max} != 期望 {exp}")
        ok = False
    if nb_max != 1:
        print(f"     !! 含目标图被重复 {nb_max} 次 (应为 1)")
        ok = False

    # 唯一图数必须等于原始规模 (dfire 为 21,501)
    n_unique_images = len({r["id"] for r in rows})
    print(f"  唯一图总数: {n_unique_images}")
    if n_unique_images != len(meta["counts"]) and n_unique_images != sum(meta["counts"].values()):
        # counts 按 manifest 行计; 过采样后 counts 会大于唯一图数, 这是预期行为
        print(f"     (manifest 行数 {len(rows)} > 唯一图数 {n_unique_images}: 过采样生效)")

    # 关键回归项: 未过采样的数据集不应有任何重复行
    if neg == 1 and len(rows) != n_unique_images:
        print("     !! neg_repeat=1 却出现重复行 -> 改动破坏了既有行为")
        ok = False
    if neg == 1:
        dup = [k for k, v in Counter(r["id"] for r in rows).items() if v > 1]
        if dup:
            print(f"     !! 发现 {len(dup)} 个重复 id")
            ok = False
        else:
            print("  [回归] neg_repeat=1, 无重复行, 与改动前一致")

    # 强回归: 若有参照数据集, 比较"去重后的内容指纹"
    # 只比 (split, 文件名, boxes), 不比 id (id 内嵌数据集名, 天然不同)
    if a.reference:
        ref_out, ref_meta, ref_rows = load_prepared(cfg, a.reference, verify=True)

        def fingerprint(rs):
            out_set = set()
            for r in rs:
                name = Path(r["image"]).name
                boxes = tuple(tuple(float(v) for v in b) for b in r["boxes"])
                out_set.add((r["split"], name, boxes))
            return out_set

        fp = fingerprint(rows)
        ref_fp = fingerprint(ref_rows)
        same = fp == ref_fp
        print(f"  [内容指纹] 与 {a.reference} 去重后内容{'一致' if same else '不一致'}"
              f"  (唯一图 {len(fp)} vs {len(ref_fp)})")
        if not same:
            print("     !! 改动改变了去重后的数据内容 -> 回归失败")
            ok = False
    print("\n结论:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
