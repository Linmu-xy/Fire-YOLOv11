#!/usr/bin/env python3
"""Dry-run or launch the first SRDG candidate-module experiment.

The paired parent is the completed protocol P2 seed0 run.  This script keeps
its data, optimization and initialization policy fixed and changes only the P2
graph by inserting SemanticResidualDetailGate.

    python scripts/train_srdg_seed0.py --dry-run  # never trains
    python scripts/train_srdg_seed0.py            # explicit researcher action
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG = "configs/protocol-v2.yaml"
DATASET = "dfire"
METHOD = "p2_srdg"
SEED = 0
DEFAULT_TAG = "module-srdg-v1"
ULTRA_REQUIRED = "8.4.42"
ARCHIVE = ROOT / "experiments" / "yolo11n" / "D-Fire_p2_srdg_seed0"
PARENT_RUN = ROOT / "experiments" / "yolo11n" / "D-Fire_p2_seed0" / "run.json"


def spec_for(tag: str):
    from firesmoke.data import load_protocol
    from firesmoke.experiment import plan

    cfg = load_protocol(ROOT / CONFIG)
    return cfg, plan(cfg, DATASET, METHOD, SEED, tag)


def command(spec: dict) -> list[str]:
    return [sys.executable, "-m", "firesmoke", "--config", CONFIG, "train",
            "--dataset", DATASET, "--method", METHOD, "--seed", str(SEED),
            "--tag", spec["tag"], "--execute"]


def preflight(cfg: dict, spec: dict) -> tuple[list[str], list[str]]:
    from firesmoke.common import digest
    from firesmoke.data import load_prepared

    report: list[str] = []
    blocking: list[str] = []

    def check(ok: bool, text: str) -> None:
        report.append(f"[{'OK ' if ok else 'FAIL'}] {text}")
        if not ok:
            blocking.append(text)

    weights = Path(spec["weights"])
    check(weights.is_file(), f"预训练权重: {weights}")
    check(Path(spec["data"]).is_file(), f"规范化数据配置: {spec['data']}")
    metadata = None
    try:
        _, metadata, rows = load_prepared(cfg, DATASET)
    except Exception as error:
        check(False, f"规范化数据完整性: {error!r}")
    else:
        check(True, f"规范化数据完整性: {len(rows)} 张图，清单与标签检查通过")

    check(not Path(spec["output"]).exists(),
          f"协议输出目录空闲: {spec['output']}（不覆盖、不续训）")
    check(not ARCHIVE.exists(), f"归档目录空闲: {ARCHIVE}")

    parent = None
    try:
        parent = json.loads(PARENT_RUN.read_text(encoding="utf-8"))
    except Exception as error:
        check(False, f"父级 P2 seed0 运行记录: {error!r}")
    else:
        identity = (parent.get("dataset"), parent.get("method"), parent.get("seed"))
        check(identity == (DATASET, "p2", SEED),
              f"父级 P2 身份: dataset/method/seed={identity}")
        check(parent.get("training") == spec["training"],
              "父级 P2 与 SRDG 的训练超参（含 seed）完全一致")

    if parent is not None and metadata is not None:
        parent_metadata = parent.get("data_metadata", {})
        check(parent_metadata.get("portable_inventory_sha256")
              == metadata.get("portable_inventory_sha256"),
              "父级 P2 与 SRDG 使用相同 portable 数据清单")
        check(parent_metadata.get("counts") == metadata.get("counts"),
              "父级 P2 与 SRDG 的 train/val/test 样本数一致")
    if parent is not None and weights.is_file():
        check(parent.get("pretrained_sha256") == digest(weights),
              "父级 P2 与 SRDG 使用相同预训练权重")

    try:
        version = importlib.metadata.version("ultralytics")
    except Exception as error:
        check(False, f"ultralytics 版本可读: {error!r}")
    else:
        check(version == ULTRA_REQUIRED,
              f"ultralytics={version}（要求 {ULTRA_REQUIRED}）")

    font = Path.home() / ".config" / "Ultralytics" / "Arial.ttf"
    check(font.is_file(), f"离线字体: {font}")

    device = str(spec["training"]["device"])
    if device.lower() == "cpu":
        report.append("[OK ] device=cpu；跳过 CUDA 检查")
    else:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except Exception as error:
            check(False, f"CUDA 检查: {error!r}")
        else:
            check(cuda, f"CUDA 可用（device={device!r}）")
    return report, blocking


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2 + SRDG 候选模块 seed0 训练")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help=f"输出 tag（默认 {DEFAULT_TAG}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只完成数据、环境和父级可比性检查，不训练")
    parser.add_argument("--no-archive", action="store_true",
                        help="训练成功后不复制到 experiments/yolo11n/")
    args = parser.parse_args(argv)

    cfg, spec = spec_for(args.tag)
    research = spec["research"]
    print(f"SRDG 候选模块训练 —— seed={SEED} tag={spec['tag']}")
    print("  研究变量  : P2 -> P2+SRDG；其余数据、初始化与训练超参冻结")
    print(f"  方法配置  : architecture={research['architecture']}, "
          f"background_alpha={research['background_alpha']}, "
          f"style={research['style']}, topk={research['topk']}")
    print(f"  协议输出  : {spec['output']}")
    print(f"  归档副本  : {ARCHIVE}" + ("（--no-archive）" if args.no_archive else ""))
    print(f"  父级参考  : {PARENT_RUN}")

    report, blocking = preflight(cfg, spec)
    print("  检查:")
    for line in report:
        print(f"    {line}")
    print("  命令:")
    print("    " + " ".join(command(spec)))

    if args.dry_run:
        print("  [dry-run] 未启动训练。")
        return 0 if not blocking else 1
    if blocking:
        print(f"  [ERROR] {len(blocking)} 项前置检查未通过，未启动训练。")
        return 1
    if subprocess.run(command(spec), cwd=ROOT).returncode != 0:
        print("  [ERROR] firesmoke train 非零退出，未归档。")
        return 1
    if args.no_archive:
        print("  [OK ] 训练结束；--no-archive，跳过归档。")
        return 0
    shutil.copytree(Path(spec["output"]), ARCHIVE)
    print(f"  [OK ] 训练结束，已归档到 {ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
