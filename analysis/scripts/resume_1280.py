# -*- coding: utf-8 -*-
"""续训 paper-v1_1280 / baseline / seed0 的第 55-100 轮。

背景
----
这次训练在 2026-09-27 18:04 启动、22:04 停在第 55 轮（epoch 索引 54）。
封装层 firesmoke 有意不支持 resume（experiment.py 抛 FileExistsError + resume 硬编码 False），
但该 run 的 background_alpha=0 → init_criterion 实测为原版 v8DetectionLoss，
且 style=False，因此用原生 ultralytics resume 与封装层在目标函数上完全等价。

行为说明
--------
- save_dir 由 checkpoint 的 train_args 解析回原目录，不会新建 seed02
- results.csv 以追加模式写入（ultralytics 仅在文件不存在时写表头）→ 结束后应为 100 行
- 注意：ResearchTrainer 的 on_train_epoch_end 回调不会生效，
  所以 background_exposure.jsonl 将停在 55 条，与 results.csv 不同步。
  该文件只记录曝光统计，baseline(alpha=0) 不参与损失，不影响训练正确性。
- 续训结束后需要手动补一份 provenance 记录到 run.json（跨进程）。
"""
import time

from ultralytics import YOLO

WEIGHTS = "outputs/paper-v1_1280/dfire/baseline/seed0/weights/last.pt"

print(f"[resume] 开始 {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"[resume] weights = {WEIGHTS}", flush=True)
model = YOLO(WEIGHTS)
print(f"[resume] 载入完成, epoch(0-based) = {model.ckpt.get('epoch')}, "
      f"best_fitness = {model.ckpt.get('best_fitness')}", flush=True)
model.train(resume=True)
print(f"[resume] 结束 {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
