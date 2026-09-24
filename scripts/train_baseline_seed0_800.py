#!/usr/bin/env python3
"""Run the protocol-controlled D-Fire baseline for seed 0.
800 resolution version, paired reference validation fully removed.
Use protocol_800.yaml, P3/P4/P5 baseline.
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
DEFAULT_TAG = "paper-v1_800"
ULTRA_REQUIRED = "8.4.42"

ARCHIVE = ROOT / "experiments" / "yolo11n" / "D-Fire_protocol_baseline_800_seed0"

CRITICAL_SOURCES = (
    "configs/protocol_800.yaml",
    "firesmoke/data.py",
    "firesmoke/experiment.py",
    "firesmoke/models.py",
)


def spec_for(tag: str):
    from firesmoke.data import load_protocol
    from firesmoke.experiment import plan
    cfg = load_protocol(ROOT / "configs/protocol_800.yaml")
    return cfg, plan(cfg, DATASET, METHOD, SEED, tag)


def command(spec: dict) -> list[str]:
    return [
        sys.executable,
        "-m",
        "firesmoke",
        "--config",
        "configs/protocol_800.yaml",
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
    """Run read-only preflight checks, paired reference validation removed."""
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
    parser = argparse.ArgumentParser(description="800分辨率 baseline seed0 训练脚本（已移除成对P2校验）")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help=f"输出 tag（默认 {DEFAULT_TAG}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只完成数据、环境检查，不训练")
    parser.add_argument("--no-archive", action="store_true",
                        help="训练成功后不复制到 experiments/yolo11n/")
    args = parser.parse_args(argv)

    cfg, spec = spec_for(args.tag)
    research = spec["research"]

    print(f"800分辨率 baseline 训练 —— seed={SEED} tag={spec['tag']}")
    print("  实验描述  : P3/P4/P5 baseline，800分辨率，无P2成对校验")
    print(f"  方法配置  : architecture={research['architecture']}, "
          f"background_alpha={research['background_alpha']}, "
          f"style={research['style']}, topk={research['topk']}")
    print(f"  协议输出  : {spec['output']}")
    print(f"  归档副本  : {ARCHIVE}" + ("（--no-archive）" if args.no_archive else ""))

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
