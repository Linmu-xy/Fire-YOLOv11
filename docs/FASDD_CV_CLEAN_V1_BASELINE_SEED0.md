# FASDD_CV clean v1：baseline seed0

## 实验登记

- 数据版本：`fasdd_cv_clean_v1`，train 44,113 / val 29,804 / test 15,488。
- 数据内容 SHA256：`56d5ca7e69646edd5b9a3351c86bfa24b190cf37e01af1335721955e8d6a1ca2`。
- 协议：`configs/protocol-fasdd-cv-clean-v1.yaml`。
- 启动脚本：`scripts/train_fasdd_cv_clean_v1_baseline_seed0.py`。
- 方法：YOLO11n baseline，P3/P4/P5，seed 0。
- 初始化：从本地 `weights/yolo11n.pt` 迁移共享层 0..22；检测头随机初始化，与仓库研究协议一致。
- 训练：100 epochs，imgsz 640，batch 32，GPU 0，workers 8，SGD，lr0 0.01，AMP=false，deterministic=true。
- 选择权重：验证集 mAP50-95；不自动执行 test 评估。
- 定位：先运行单种子基线，作为管线和性能观察；感知近重复候选仍待审查，不能据此宣称场景独立。

## 目录规范

按 `PROJECT_LAYOUT.md`：

```text
scripts/train_fasdd_cv_clean_v1_baseline_seed0.py
configs/protocol-fasdd-cv-clean-v1.yaml
docs/FASDD_CV_CLEAN_V1_BASELINE_SEED0.md
results/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed0/
  run.json                  # 框架写入环境、源码指纹、完整训练参数、数据元信息
  protocol.yaml             # 框架保存实际协议
  args.yaml                 # Ultralytics 实际参数
  results.csv               # 每个 epoch 的训练 / val 指标
  weights/best.pt
  weights/last.pt
  launch/                   # 本次启动脚本、配置、源码归档、进程与最终状态
```

控制台日志和运行缓存放在仓库外：`<仓库父目录>/Fire-YOLOv11-runtime/fasdd_cv_clean_v1/<tag>/baseline/seed0/<UTC时间>/`。其中 `train.log` 为持续日志，`process.json` 为启动进程记录，`worker.json` 保存监督进程和训练进程 PID，`status.json` 在训练结束时记录成功或失败。它们不会作为仓库临时文件提交。

启动时保存 Python 源码、配置、依赖文件和目录规范的归档及 SHA256；实验目录建立后同步保存到 `launch/`。后台进程与 SSH 会话分离。

## 命令

在仓库根目录，使用已安装的 `yolo11` Python 环境：

```bash
# 只预检，默认不训练
python scripts/train_fasdd_cv_clean_v1_baseline_seed0.py --dry-run

# 用户已授权的 seed0 基线训练：后台启动，输出日志位置和 PID
python scripts/train_fasdd_cv_clean_v1_baseline_seed0.py --execute
```

已有输出目录会拒绝覆盖。失败记录保留；另一次独立实验可显式指定新 `--tag`。启动器不自动续训、不自动降低 batch、不修改原协议。

查看启动打印出的日志路径：

```bash
tail -f <日志绝对路径>/train.log
```

运行中的 `launch/` 记录用于追溯；最终成败以 `status.json` 为准。训练开始和首批次的状态另记在本节下方。

## 本次启动记录

- 工作站：lxy（100.94.208.28）。
- 仓库：`/home/lxy/Documents/yolo11/Fire-YOLOv11`。
- 启动时间：2026-10-08 11:43:43（北京时间，UTC+8）。
- 监督进程 PID：5452；训练进程 PID：5458。
- 结果目录：`/home/lxy/Documents/yolo11/Fire-YOLOv11/results/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed0/`。
- 日志：`/home/lxy/Documents/yolo11/Fire-YOLOv11-runtime/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed0/20261008T034343_274638Z/train.log`。
- 已观察到 epoch 1/100 的第 128/1379 个训练批次，损失为有限值；GPU 利用率 97%，总显存占用 8,735 MiB。
- `run.json` 的 `training_started` 为 true；实际 `args.yaml` 确认 seed=0、epochs=100、batch=32、imgsz=640、optimizer=SGD、AMP=false、split=val。
- `launch/` 内已保存启动脚本、冻结协议、源码归档、启动信息和训练 PID。

以上是启动阶段的状态快照，不代表训练已完成。没有启动 seed1/seed2，也没有进行 test 评估。
