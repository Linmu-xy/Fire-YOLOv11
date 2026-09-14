#!/usr/bin/env python3
"""yolo11n 在 D-Fire 上的种子训练 —— seed 0。

输出: experiments/yolo11n/D-Fire_seed0/
参数与基线 experiments/yolo11n/D-Fire/ 完全一致，仅 seed 不同，保证可比。

运行: python scripts/train_seed0.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # Fire-YOLOv11/
SEED = 0


def main() -> None:
    model = YOLO(str(ROOT / "weights" / "yolo11n.pt"))
    model.train(
        data=str(ROOT / "datasets" / "D-Fire" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=32,
        patience=100,
        close_mosaic=10,
        seed=SEED,
        device=0,
        workers=8,
        project=str(ROOT / "experiments" / "yolo11n"),
        name=f"D-Fire_seed{SEED}",
    )


if __name__ == "__main__":
    main()
