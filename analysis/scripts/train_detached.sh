#!/bin/bash
# 后台启动一次训练（fd 安全 + GPU 占用检查），日志按 run 分文件。
#
# 用法:
#   bash analysis/scripts/train_detached.sh <dataset> <method> <seed> <tag> [repo_root]
# 例:
#   bash analysis/scripts/train_detached.sh dfire baseline 2 paper-v1
#
# 为什么需要它:
#   * 直接在 SSH 会话里 setsid 会把 fd 泄漏给通道，导致远端调用永不返回
#     -> 这里统一用 `setsid bash -c "exec ... >> log 2>&1 </dev/null"` 打开全部标准流
#   * 实测过 llama-server 之类的进程会与训练抢 GPU，白白拖慢数小时
#     -> 启动前检查显存占用，超过阈值就拒绝启动
set -eu

DS="${1:?需要 dataset}"; METHOD="${2:?需要 method}"; SEED="${3:?需要 seed}"; TAG="${4:?需要 tag}"
ROOT="${5:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
[ -d firesmoke ] && [ -d configs ] || { echo "!! 不在仓库根目录: $PWD"; exit 1; }

PY="$HOME/miniconda3/envs/yolo11/bin/python"
[ -x "$PY" ] || PY="python3"

# --- GPU 占用检查: 有别的进程占着大显存就不启动 ---
if command -v nvidia-smi >/dev/null 2>&1; then
  BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true)
  if [ -n "$BUSY" ]; then
    echo "!! GPU 上已有计算进程，拒绝启动（避免争用拖慢）:"
    echo "$BUSY" | sed 's/^/     /'
    echo "   确认可以共存后，用 FORCE=1 跳过检查，或先停掉对方进程。"
    [ "${FORCE:-0}" = "1" ] || exit 3
  fi
fi

mkdir -p analysis/logs
LOG="analysis/logs/${TAG}__${DS}__${METHOD}__seed${SEED}.log"
OUT="outputs/${TAG}/${DS}/${METHOD}/seed${SEED}"
if [ -e "$OUT" ]; then
  echo "!! 输出目录已存在，封装层会拒绝覆盖: $OUT"; exit 4
fi

export PYTHONPATH="$ROOT"
CMD=( "$PY" -m firesmoke --config configs/protocol.yaml train
      --dataset "$DS" --method "$METHOD" --seed "$SEED" --tag "$TAG" --execute )

echo "启动: ${CMD[*]}"
echo "  日志: $LOG"
echo "  输出: $OUT"
setsid bash -c "exec ${CMD[*]} >> '$ROOT/$LOG' 2>&1 < /dev/null" </dev/null >/dev/null 2>&1 &
sleep 20
if pgrep -f "dataset $DS --method $METHOD --seed $SEED --tag $TAG" >/dev/null; then
  echo "  状态: 已在后台启动"
  tail -c 400 "$LOG" | tr '\r' '\n' | tail -3
else
  echo "  状态: 未检测到进程，请看日志"; tail -20 "$LOG" 2>/dev/null
fi
