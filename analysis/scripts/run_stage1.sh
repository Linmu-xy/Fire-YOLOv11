#!/bin/bash
# 阶段 1 训练队列（顺序执行，约 5 × 1.9h ≈ 9.5 GPU 小时）
#
# 设计依据见 docs/实验与训练设计_v1.md
#   A0 对照: 补齐同协议 baseline 的 seed1 / seed2（现有只有 seed0）
#   A1 实验: dfire_hn3 = D-Fire train 的纯背景图在 manifest 中重复 3 次
#            只作用于 train（val/test 保持每张一次，避免按图像求均值的指标被重复行偏置
#
# 固定: imgsz=640 / batch=32 / lr0=0.01 / epochs=100 / deterministic=true（沿用 protocol.yaml）
# 唯一自变量: 数据构成
set -u
cd /home/cmj/Documents/FireSmoke/Fire-YOLOv11 || exit 1
export PYTHONPATH=$PWD
PY=/home/cmj/miniconda3/envs/yolo11/bin/python

run () {
  echo "=== $(date '+%F %T') START: $* ==="
  "$PY" -m firesmoke --config configs/protocol.yaml train "$@" --execute
  rc=$?
  echo "=== $(date '+%F %T') DONE rc=$rc: $* ==="
  if [ $rc -ne 0 ]; then
    echo "!!! 队列因失败终止: $*"
    exit $rc
  fi
}

# ---- A0: 对照补齐 ----
run --dataset dfire     --method baseline --seed 1 --tag paper-v1
run --dataset dfire     --method baseline --seed 2 --tag paper-v1

# ---- A1: hard-negative 加强版 ----
run --dataset dfire_hn3 --method baseline --seed 0 --tag hn-v1
run --dataset dfire_hn3 --method baseline --seed 1 --tag hn-v1
run --dataset dfire_hn3 --method baseline --seed 2 --tag hn-v1

echo "=== $(date '+%F %T') 阶段 1 队列全部完成 ==="
