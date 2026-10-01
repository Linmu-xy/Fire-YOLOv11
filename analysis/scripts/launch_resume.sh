#!/bin/bash
# 用 setsid 完全脱离 SSH 会话启动续训, 避免连接波动导致 SIGHUP 杀掉训练。
cd /home/lxy/Documents/yolo11/Fire-YOLOv11 || exit 1
export PYTHONPATH=/home/lxy/Documents/yolo11/Fire-YOLOv11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/lxy/miniconda3/envs/yolo11/bin/python -u analysis/resume_1280.py
