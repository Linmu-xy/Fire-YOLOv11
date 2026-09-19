#!/usr/bin/env python3
"""Paired DCBR launcher. Defaults to preflight; training requires --execute."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = "configs/protocol-dcbr.yaml"
METHODS = ("p2_dcbr", "p2_dcbr_semantic", "p2_dcbr_whole")


def paired_errors(parent, spec, metadata, pretrained_sha256):
    """Fail closed when a paired experiment lacks comparable evidence."""
    errors = []
    if (parent.get("dataset"), parent.get("method"), parent.get("seed")) != (
            "dfire", "p2", spec["seed"]):
        errors.append("父级必须为相同 seed 的 dfire/p2")
    for key in ("training", "initialization", "selection"):
        if parent.get(key) != spec[key]:
            errors.append(f"父级 {key} 不一致")
    if not pretrained_sha256 or parent.get("pretrained_sha256") != pretrained_sha256:
        errors.append("预训练权重指纹不一致或缺失")
    old = parent.get("data_metadata", {})
    for key in ("portable_inventory_sha256", "counts", "names", "label_policy"):
        if not metadata.get(key) or old.get(key) != metadata[key]:
            errors.append(f"数据 {key} 不一致或缺失")
    expected = {"architecture": "p2", "background_alpha": 0.0, "style": False,
                "topk": spec["research"]["topk"]}
    if parent.get("research") != expected:
        errors.append("父级研究变量与冻结 P2 对照不一致")
    return errors


def preflight(cfg, spec, archive):
    from firesmoke.common import digest
    from firesmoke.data import load_prepared

    errors, notes = [], []
    for label, p in (("输出", Path(spec["output"])), ("归档", archive)):
        if p is not None and p.exists():
            errors.append(f"{label}目录已存在: {p}；请换 --tag")
    metadata = {}
    try:
        _, metadata, rows = load_prepared(cfg, "dfire")
        notes.append(f"数据清单和标签验证通过: {len(rows)} 张图")
    except Exception as error:
        errors.append(f"数据检查失败: {error}")
    parent_file = ROOT / f"experiments/yolo11n/D-Fire_p2_seed{spec['seed']}/run.json"
    weights = Path(spec["weights"])
    try:
        parent = json.loads(parent_file.read_text(encoding="utf-8"))
        weight_hash = digest(weights)
        errors.extend(paired_errors(parent, spec, metadata, weight_hash))
        graph = "configs/models/yolo11n-p2.yaml"
        recorded = parent.get("provenance", {}).get("source_sha256", {}).get(graph)
        if not recorded or recorded != digest(ROOT / graph):
            errors.append("父级 P2 结构源码指纹不一致或缺失")
        notes.append(f"父级参考: {parent_file}")
    except Exception as error:
        errors.append(f"父级/预训练权重检查失败: {error}")
    try:
        from firesmoke.common import offline_runtime
        offline_runtime()
        import torch
        notes.append(f"ultralytics={importlib.metadata.version('ultralytics')}, torch={torch.__version__}")
        device = str(spec["training"]["device"])
        if device != "cpu":
            if not torch.cuda.is_available() or not device.isdigit() or int(device) >= torch.cuda.device_count():
                errors.append(f"CUDA device={device} 不可用；请在训练环境中复查")
        font = Path.home() / ".config/Ultralytics/Arial.ttf"
        if not font.is_file():
            errors.append(f"离线绘图字体缺失: {font}")
    except Exception as error:
        errors.append(f"运行环境检查失败: {error}")
    return notes, errors


def main(argv=None):
    from firesmoke.data import load_protocol
    from firesmoke.experiment import plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--method", choices=METHODS, default="p2_dcbr")
    parser.add_argument("--tag", default="module-dcbr-v1")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true", help="仅预检（也是默认行为）")
    actions.add_argument("--execute", action="store_true", help="预检通过后启动训练")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_protocol(ROOT / CONFIG)
    spec = plan(cfg, "dfire", args.method, args.seed, args.tag)
    archive = ROOT / f"experiments/yolo11n/D-Fire_{args.method}_seed{args.seed}_{args.tag}"
    notes, errors = preflight(cfg, spec, None if args.no_archive else archive)
    print(f"DCBR 候选: {args.method}, seed={args.seed}, tag={args.tag}")
    print(f"输出: {spec['output']}")
    if not args.no_archive:
        print(f"成功后归档: {archive}")
    for note in notes:
        print(f"[OK] {note}")
    for error in errors:
        print(f"[FAIL] {error}")
    if not args.execute:
        print("[dry-run] 未启动训练；训练需显式 --execute")
        return int(bool(errors))
    if errors:
        print("预检未通过，未启动训练")
        return 1
    cmd = [sys.executable, "-m", "firesmoke", "--config", CONFIG, "train",
           "--dataset", "dfire", "--method", args.method, "--seed", str(args.seed),
           "--tag", args.tag, "--execute"]
    if subprocess.run(cmd, cwd=ROOT).returncode:
        print("训练未成功结束，保留原始输出，未归档")
        return 1
    if not args.no_archive:
        shutil.copytree(spec["output"], archive)
        print(f"归档完成: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
