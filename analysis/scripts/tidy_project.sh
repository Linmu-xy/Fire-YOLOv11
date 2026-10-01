#!/bin/bash
# 整理 Fire-YOLOv11 的 analysis/ 与备份文件，保持项目结构工整。
#
# 目标结构:
#   analysis/
#     scripts/    所有分析脚本 (.py / .sh)
#     reports/    出具的报告 (.md / 小型 .json)，体积小、值得留档与版本化
#     work/       重中间产物 (predictions.json 等)，可由 checkpoint 重建
#     logs/       运行日志，体积大且含进度条刷屏，不入版本库
#   backups/      被修改过的源码文件的原始备份（与源码目录分开）
#   artifacts/data/<ds>/  prepared 数据集（仓库已 gitignore）——不动
#   outputs/             训练输出（仓库已跟踪）——不动
#
# 注意: 移动正在被打开写的日志文件是安全的（同一文件系统内 fd 保留），
#       但本脚本会在结束后校验日志仍在增长。
set -eu
# 本脚本位于 analysis/scripts/ 下 -> 仓库根 = 上两级；也允许用参数显式指定
ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT" || exit 1
if [ ! -d firesmoke ] || [ ! -d configs ]; then
  echo "!! 看起来不在仓库根目录: $PWD"; exit 1
fi
echo "仓库根目录: $PWD"

echo "--- 1) 建立目标目录 ---"
mkdir -p analysis/scripts analysis/reports analysis/work analysis/logs backups

echo "--- 2) 脚本 -> analysis/scripts ---"
for f in analysis/*.py analysis/*.sh; do
  [ -e "$f" ] || continue
  echo "    $(basename "$f")"
  mv "$f" analysis/scripts/
done

echo "--- 3) 报告（.md / .txt / .png）-> analysis/reports ---"
for f in analysis/*.md analysis/*.txt analysis/*.png; do
  [ -e "$f" ] || continue
  echo "    $(basename "$f")"
  mv "$f" analysis/reports/
done

echo "--- 3b) 清理 __pycache__ ---"
for d in analysis/__pycache__ analysis/scripts/__pycache__; do
  [ -d "$d" ] && rm -rf "$d" && echo "    已删 $d"
done
true

echo "--- 4) 产物目录 -> analysis/work，并把 .md 报告提到 reports ---"
for d in analysis/out_* analysis/rel_val; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  # 先把报告提出来（避免重产物目录被误删后报告丢失）
  while IFS= read -r md; do
    [ -n "$md" ] || continue
    dst="analysis/reports/${name}__$(basename "$md")"
    cp "$md" "$dst"
    echo "    报告: $dst"
  done < <(find "$d" -maxdepth 1 -name '*.md' 2>/dev/null)
  # 再把重产物移走；rel_val 是冒烟测试残留，直接删除
  if [ "$name" = "rel_val" ]; then
    rm -rf "$d"
    echo "    删除冒烟残留: $name"
  else
    rm -rf "analysis/work/$name"
    mv "$d" "analysis/work/$name"
    echo "    产物: analysis/work/$name"
  fi
done

echo "--- 5) 日志 -> analysis/logs ---"
for f in analysis/*.log; do
  [ -e "$f" ] || continue
  echo "    $(basename "$f")"
  mv "$f" analysis/logs/
done

echo "--- 6) 源码备份 -> backups/ ---"
for f in configs/*.orig_20261001 firesmoke/*.orig_20261001; do
  [ -e "$f" ] || continue
  echo "    $f"
  mv "$f" backups/
done

echo "--- 7) 忽略日志与重产物（若尚未配置）---"
touch .gitignore
for pat in "analysis/logs/" "analysis/work/"; do
  if ! grep -qxF "$pat" .gitignore; then
    printf '%s\n' "$pat" >> .gitignore
    echo "    .gitignore += $pat"
  else
    echo "    已存在: $pat"
  fi
done

echo
echo "=== 整理后 ==="
ls -la analysis/
echo
du -sh analysis/* backups 2>/dev/null | sort -h
