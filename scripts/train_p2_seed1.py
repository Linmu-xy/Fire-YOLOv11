#!/usr/bin/env python3
"""P2 消融第二轮 —— seed 1。

结构 = configs/models/yolo11n-p2.yaml（P2/P3/P4/P5 四尺度，纯卷积，无注意力层）。
流程走论文协议框架 firesmoke：共享层 0..22 迁移 + Detect 头随机初始化、SGD、amp=false。
结果落在 results/dfire/paper-v1/p2/seed1/（训练成功即最终位置，无二次归档）。

运行: python scripts/train_p2_seed1.py              # 训练
      python scripts/train_p2_seed1.py --dry-run    # 只预览检查与命令
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_p2_common import main

SEED = 1

if __name__ == "__main__":
    raise SystemExit(main(SEED))
