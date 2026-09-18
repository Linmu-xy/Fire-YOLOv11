#!/usr/bin/env python3
"""P2 消融三轮训练（seed 0/1/2）的共享实现。

训练对象是仓库里实际存在的 P2 图 configs/models/yolo11n-p2.yaml：从 neck P3（第 16 层）
上采样后与 backbone P2（第 2 层）拼接，经 DWConv + 1x1 Conv + C3k2 得到 stride=4 的
检测输入，Detect 接 [27, 16, 19, 22] 四个尺度。该图只用上游原生算子，不含注意力层。

训练交给论文协议框架 firesmoke（等价命令 python -m firesmoke train ... --execute），
因此具备协议要求的：只迁移共享层 0..22 权重 + Detect 头随机初始化、SGD/lr0=0.01/
amp=false、100 epochs/batch 32、run.json 与 weight_transfer.json 元数据，以及
"输出目录已存在即拒绝" 的保护。

不要用通用 yolo train 或历史 scripts/train_seed*.py 启动本项：那是完整 .pt 微调 +
optimizer=auto + amp=true，初始化政策与优化器和协议不同，结果不可与 firesmoke 的
baseline 混比。

输出：
  outputs/<tag>/dfire/p2/seed<k>/            协议原始输出（含 run.json，训练前不清理）
  experiments/yolo11n/D-Fire_p2_seed<k>/     训练成功后的归档副本（可 --no-archive 跳过）

用法（直接调用共享实现，需要显式给 seed）:
    python scripts/train_p2_common.py --seed 0 --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Fire-YOLOv11/
sys.path.insert(0, str(ROOT))  # 支持 `python scripts/train_p2_common.py` 直接运行

DATASET = "dfire"
METHOD = "p2"
GRAPH = "configs/models/yolo11n-p2.yaml"
DEFAULT_TAG = "paper-v1"
ULTRA_REQUIRED = "8.4.42"


def spec_for(seed, tag):
    """按 configs/protocol.yaml 推导本次运行的路径与超参（纯计算，不训练、不写文件）。"""
    from firesmoke.data import load_protocol
    from firesmoke.experiment import plan

    cfg = load_protocol(ROOT / "configs/protocol.yaml")
    return plan(cfg, DATASET, METHOD, seed, tag)


def archive_dir(seed):
    return ROOT / "experiments" / "yolo11n" / f"D-Fire_{METHOD}_seed{seed}"


def command(spec):
    """与 docs/QUICKSTART.md 的 P2 消融命令一致；-m firesmoke 需要 cwd=仓库根。"""
    return [sys.executable, "-m", "firesmoke", "--config", "configs/protocol.yaml", "train",
            "--dataset", DATASET, "--method", METHOD, "--seed", str(spec["seed"]),
            "--tag", spec["tag"], "--execute"]


def preflight(spec):
    """训练前的只读检查；返回 (报告行, 阻断原因列表)。"""
    report, blocking = [], []

    def check(ok, text):
        report.append(f"[{'OK ' if ok else 'FAIL'}] {text}")
        if not ok:
            blocking.append(text)

    check(Path(spec["weights"]).is_file(), f"预训练权重: {spec['weights']}")
    check(Path(spec["data"]).is_file(),
          f"规范化数据: {spec['data']}"
          f"（缺失时先运行 python -m firesmoke prepare --dataset {DATASET}）")
    check(not Path(spec["output"]).exists(),
          f"协议输出目录空闲: {spec['output']}（已存在请换 --tag；v1 不覆盖也不续训）")
    check(not archive_dir(spec["seed"]).exists(), f"归档目录空闲: {archive_dir(spec['seed'])}")

    try:
        import importlib.metadata

        version = importlib.metadata.version("ultralytics")
    except Exception as error:  # 环境不对时给明确报错，而不是训练中途炸
        check(False, f"ultralytics 版本可读: {error!r}")
    else:
        check(version == ULTRA_REQUIRED,
              f"ultralytics 版本 {version}（适配器要求 {ULTRA_REQUIRED}，见 firesmoke/common.py）")

    device = str(spec["training"]["device"])
    if device.lower() == "cpu":
        report.append("[OK ] device=cpu，跳过 CUDA 检查（CPU 训练会非常慢）")
    else:
        note = ""
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except Exception as error:
            cuda, note = False, f"（导入 torch 失败: {error!r}）"
        check(cuda, f"CUDA 可用（configs/protocol.yaml 的 device={device!r}）{note}；"
                    "不可用时先修驱动/重启，或临时把 protocol.yaml 的 device 改成 'cpu'")
    return report, blocking


def main(seed, argv=None):
    parser = argparse.ArgumentParser(
        description=f"P2 消融训练 —— dataset={DATASET} method={METHOD} seed={seed}")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"输出 tag（默认 {DEFAULT_TAG}）")
    parser.add_argument("--dry-run", action="store_true", help="只打印检查结果和将执行的命令，不训练")
    parser.add_argument("--no-archive", action="store_true",
                        help=f"不复制结果到 experiments/yolo11n/D-Fire_{METHOD}_seed{seed}/")
    args = parser.parse_args(argv)

    spec = spec_for(seed, args.tag)
    research = spec["research"]
    print(f"P2 消融训练 —— seed={seed} tag={spec['tag']}")
    print(f"  图        : {GRAPH}（architecture={research['architecture']}, "
          f"background_alpha={research['background_alpha']}, style={research['style']}, "
          f"topk={research['topk']}）")
    print(f"  协议输出  : {spec['output']}")
    print(f"  归档副本  : {archive_dir(seed)}" + ("（--no-archive）" if args.no_archive else ""))

    report, blocking = preflight(spec)
    print("  检查:")
    for line in report:
        print(f"    {line}")
    print("  命令:")
    print("    " + " ".join(command(spec)))

    if args.dry_run:
        print("  [dry-run] 未启动训练。")
        return 0
    if blocking:
        print(f"  [ERROR] {len(blocking)} 项前置检查未通过，未启动训练。")
        return 1

    if subprocess.run(command(spec), cwd=ROOT).returncode != 0:
        print("  [ERROR] firesmoke train 非零退出，未归档。")
        return 1

    if args.no_archive:
        print("  [OK ] 训练结束；--no-archive，跳过归档。")
        return 0
    destination = archive_dir(seed)
    if destination.exists():
        print(f"  [WARN] 归档目标已存在，跳过复制: {destination}")
        return 0
    shutil.copytree(Path(spec["output"]), destination)
    print(f"  [OK ] 训练结束，已归档到 {destination}")
    return 0


if __name__ == "__main__":
    seeding = argparse.ArgumentParser(add_help=False)
    seeding.add_argument("--seed", type=int, required=True)
    known, rest = seeding.parse_known_args()
    raise SystemExit(main(known.seed, rest))
