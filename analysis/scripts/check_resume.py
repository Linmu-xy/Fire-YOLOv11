# -*- coding: utf-8 -*-
"""非破坏性验证: 检查某个 run 的 last.pt 能否被 ultralytics 正确 resume。
只做解析与加载, 不构造 optimizer, 不执行任何训练步骤。"""
import os
import sys

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

last = sys.argv[1]

print("ultralytics version =", ultralytics.__version__)
print("checkpoint =", last)

ck = torch.load(last, map_location="cpu", weights_only=False)
print("\n--- checkpoint 内容 ---")
print("  epoch (0-based) =", ck.get("epoch"), "-> 已完成", (ck.get("epoch") or -1) + 1, "轮")
print("  best_fitness    =", ck.get("best_fitness"))
print("  updates         =", ck.get("updates"))
print("  optimizer state =", ck.get("optimizer") is not None)
print("  ema state       =", ck.get("ema") is not None)

y = YOLO(last)
m = y.model
ta = dict((ck.get("train_args") or y.overrides or {}))
print("\n--- 模型对象 ---")
print("  ckpt model class =", type(m).__name__)
em = ck.get("ema")
print("  ema class        =", type(em).__name__ if em is not None else "缺失")
print("  research_alpha   =", getattr(m, "research_alpha", "NOT-SET"))
try:
    print("  init_criterion   =", type(m.init_criterion()).__name__)
except Exception as e:
    print("  init_criterion   = 失败:", type(e).__name__, e)
cfg = get_cfg(ta)
print("\n--- resume 参数解析 ---")
print("  epochs        =", cfg.epochs)
print("  imgsz / batch =", cfg.imgsz, "/", cfg.batch)
print("  close_mosaic  =", cfg.close_mosaic)
print("  lr0 / lrf     =", cfg.lr0, "/", cfg.lrf)
print("  optimizer     =", cfg.optimizer)
print("  save_dir      =", cfg.save_dir, "| 存在:", os.path.isdir(str(cfg.save_dir)))
print("  project / name=", cfg.project, "/", cfg.name)
print("  exist_ok      =", cfg.exist_ok)

print("\n--- 路径可达性 ---")
for k in ("data", "model", "weights"):
    v = ta.get(k)
    if v:
        ex = os.path.exists(str(v))
        print(f"  {k:8s} = {v}  存在={ex}")
        if not ex and k == "model":
            print("           （resume 用 checkpoint 内的模型对象, 通常不需要这个 yaml; 但值得记录）")

done = (ck.get("epoch") or -1) + 1
print(f"\n--- 结论 ---\n  已完成 {done} 轮, 目标 {cfg.epochs} 轮, 还剩 {cfg.epochs - done} 轮")
ok = (ck.get("optimizer") is not None) and (ck.get("ema") is not None) and os.path.isdir(str(cfg.save_dir))
print("  可续训:", "是" if ok else "否（缺 optimizer/ema 或 save_dir 不存在）")
