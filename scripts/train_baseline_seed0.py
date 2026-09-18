#!/usr/bin/env python3
"""Run the protocol-controlled D-Fire baseline for seed 0.

This is the paired control for experiments/yolo11n/D-Fire_p2_seed0.  Both arms
use the canonical D-Fire subset, SGD, AMP disabled, and the same transfer rule:
shared YOLO11 layers 0..22 are loaded while the Detect head is initialized from
scratch.  The only intended difference is the detection graph (P3/P4/P5 versus
P2/P3/P4/P5).

The default invocation starts training.  Always inspect the dry run first:

    python scripts/train_baseline_seed0.py --dry-run
    python scripts/train_baseline_seed0.py
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

DATASET = "dfire"
METHOD = "baseline"
SEED = 0
DEFAULT_TAG = "paper-v1"
ULTRA_REQUIRED = "8.4.42"
ARCHIVE = ROOT / "experiments" / "yolo11n" / "D-Fire_protocol_baseline_seed0"
PAIRED_REFERENCE = ROOT / "experiments" / "yolo11n" / "D-Fire_p2_seed0" / "run.json"
CRITICAL_SOURCES = (
    "configs/protocol.yaml",
    "firesmoke/data.py",
    "firesmoke/experiment.py",
    "firesmoke/models.py",
)


def spec_for(tag: str):
    from firesmoke.data import load_protocol
    from firesmoke.experiment import plan

    cfg = load_protocol(ROOT / "configs/protocol.yaml")
    return cfg, plan(cfg, DATASET, METHOD, SEED, tag)


def command(spec: dict) -> list[str]:
    return [
        sys.executable,
        "-m",
        "firesmoke",
        "--config",
        "configs/protocol.yaml",
        "train",
        "--dataset",
        DATASET,
        "--method",
        METHOD,
        "--seed",
        str(SEED),
        "--tag",
        spec["tag"],
        "--execute",
    ]


def preflight(cfg: dict, spec: dict) -> tuple[list[str], list[str]]:
    """Run read-only checks that establish a valid paired comparison."""
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
          f"协议输出目录空闲: {spec['output']}（v1 不覆盖、不续训）")
    check(not ARCHIVE.exists(), f"归档目录空闲: {ARCHIVE}")

    reference = None
    try:
        reference = json.loads(PAIRED_REFERENCE.read_text(encoding="utf-8"))
    except Exception as error:
        check(False, f"成对 P2 seed0 运行记录: {error!r}")
    else:
        identity = (reference.get("protocol"), reference.get("dataset"),
                    reference.get("method"), reference.get("seed"))
        check(identity == ("firesmoke-v1", DATASET, "p2", SEED),
              f"成对 P2 运行身份: protocol/dataset/method/seed={identity}")
        check(reference.get("training") == spec["training"],
              "成对 P2 的训练超参（含 seed）与本次 baseline 完全一致")

    if reference is not None and metadata is not None:
        reference_metadata = reference.get("data_metadata", {})
        check(reference_metadata.get("portable_inventory_sha256")
              == metadata.get("portable_inventory_sha256"),
              "成对 P2 与本次 baseline 使用相同 portable 数据清单")
        check(reference_metadata.get("counts") == metadata.get("counts"),
              "成对 P2 与本次 baseline 的 train/val/test 样本数一致")

    if reference is not None and weights.is_file():
        check(reference.get("pretrained_sha256") == digest(weights),
              "成对 P2 与本次 baseline 使用相同预训练权重")
        source_hashes = reference.get("provenance", {}).get("source_sha256", {})
        mismatches = [name for name in CRITICAL_SOURCES
                      if source_hashes.get(name) != digest(ROOT / name)]
        check(not mismatches,
              "成对 P2 与本次 baseline 的关键训练源码一致"
              + (f"；不一致: {', '.join(mismatches)}" if mismatches else ""))

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
    parser = argparse.ArgumentParser(description="协议化 baseline seed0 成对对照训练")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help=f"输出 tag（默认 {DEFAULT_TAG}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只完成数据、环境和成对可比性检查，不训练")
    parser.add_argument("--no-archive", action="store_true",
                        help="训练成功后不复制到 experiments/yolo11n/")
    args = parser.parse_args(argv)

    cfg, spec = spec_for(args.tag)
    research = spec["research"]
    print(f"协议化 baseline 成对训练 —— seed={SEED} tag={spec['tag']}")
    print("  对照问题  : P3/P4/P5 baseline 与 P2/P3/P4/P5 在相同协议下的结构差异")
    print(f"  方法配置  : architecture={research['architecture']}, "
          f"background_alpha={research['background_alpha']}, "
          f"style={research['style']}, topk={research['topk']}")
    print(f"  协议输出  : {spec['output']}")
    print(f"  归档副本  : {ARCHIVE}" + ("（--no-archive）" if args.no_archive else ""))
    print(f"  成对参考  : {PAIRED_REFERENCE}")

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
