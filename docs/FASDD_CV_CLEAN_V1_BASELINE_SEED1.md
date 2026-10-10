# FASDD_CV clean v1：cmj baseline seed1

## 本次实验

- 工作站：cmj（100.89.55.70）。
- 仓库：`/home/cmj/Documents/FireSmoke/Fire-YOLOv11`。
- 数据：`fasdd_cv_clean_v1`，train 44,113 / val 29,804 / test 15,488。
- 内容 SHA256：`56d5ca7e69646edd5b9a3351c86bfa24b190cf37e01af1335721955e8d6a1ca2`。
- 协议：`configs/protocol-fasdd-cv-clean-v1.yaml`，与 lxy baseline seed0 同一协议。
- 方法：YOLO11n baseline，随机种子 1，100 epochs，imgsz 640，batch 32，workers 8，SGD，GPU 0，AMP=false，deterministic=true。
- 初始化：本地 `weights/yolo11n.pt` 的共享层 0..22 迁移，检测头随机初始化。
- 权重选择：验证集 mAP50-95；训练过程中使用 val，不自动执行 test 评估。
- 数据版本的感知近重复候选仍待核验；本轮为同协议单种子基线。

## 保存位置

```text
scripts/train_fasdd_cv_clean_v1_baseline_seed1.py
docs/FASDD_CV_CLEAN_V1_BASELINE_SEED1.md
results/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed1/
  args.yaml
  run.json
  protocol.yaml
  results.csv
  weights/best.pt
  weights/last.pt
  launch/                 # 启动脚本、配置、源码快照与进程记录
```

日志和运行缓存位于仓库外：

`/home/cmj/Documents/FireSmoke/Fire-YOLOv11-runtime/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed1/<UTC时间>/`

入口默认只预检。显式执行后创建与 SSH 会话分离的后台训练进程：

```bash
cd /home/cmj/Documents/FireSmoke/Fire-YOLOv11
/home/cmj/miniconda3/envs/yolo11/bin/python scripts/train_fasdd_cv_clean_v1_baseline_seed1.py --execute
```

已有结果目录会拒绝覆盖。最终运行成败由日志目录和结果目录 `launch/status.json` 记录。

## 启动状态

- 启动时间：2026-10-08 12:27:34（北京时间，UTC+8）。
- 监督进程 PID：4913；训练进程 PID：4918。
- 具体日志：`/home/cmj/Documents/FireSmoke/Fire-YOLOv11-runtime/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed1/20261008T042734_221918Z/train.log`。
- 完整结果目录：`/home/cmj/Documents/FireSmoke/Fire-YOLOv11/results/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed1/`。
- 已观察到 epoch 1/100 的第 196/1379 个训练批次，损失值有限；GPU 利用率 97%，显存占用 8,765 MiB。
- `run.json` 确认 seed=1、`training_started=true`；`launch/` 已归档脚本、配置、源码快照、启动信息和训练进程记录。
- 与 lxy seed0 的实际 `run.json` 对照，训练参数（随机种子除外）、方法设置、预训练权重、数据内容、训练核心源码、注册配置及关键包版本均一致。两台机器驱动版本不同，环境信息分别保存，不能保证跨驱动逐位一致。

这是启动阶段的状态记录，训练尚未完成。本地保存本次启动元数据和 seed0/seed1 对照结果。
