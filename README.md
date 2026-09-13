# Fire-YOLOv11

火焰 / 烟雾检测的论文实验仓库：在同一批数据集上对比不同检测算法。

## 目录结构

```
Fire-YOLOv11/
├── datasets/                     # 数据集（各算法共享，只读）
│   ├── D-Fire/                   # 21527 张，类别 0=smoke, 1=fire
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── data.yaml
│   └── FASDD_CV/                 # 95314 张，类别 0=fire, 1=smoke
│       ├── images/
│       ├── annotations/YOLO_CV/{labels,train.txt,val.txt,test.txt}
│       └── data.yaml
├── weights/                      # 预训练权重
│   ├── yolo11n.pt
│   └── yolo26n.pt
├── experiments/                  # 实验结果，按 <算法>/<数据集>/ 分开
│   └── yolo11n/D-Fire/
│       ├── results.csv
│       ├── results.png / *PR_curve.png / confusion_matrix*.png
│       └── weights/{best.pt,last.pt}
└── scripts/
    └── prepare_dataset.py        # D-Fire 划分脚本（历史记录，原始数据已清理）
```

## 环境

```bash
conda activate yolo11     # ultralytics 8.4.42, torch 2.11.0+cu130
```

## 训练

`project` 指向算法目录、`name` 用数据集名，结果就自动落在 `experiments/<算法>/<数据集>/`：

```bash
# yolo11n on D-Fire
yolo train model=weights/yolo11n.pt data=datasets/D-Fire/data.yaml \
  epochs=100 imgsz=640 batch=32 device=0 workers=8 \
  project=experiments/yolo11n name=D-Fire

# yolo26n on D-Fire
yolo train model=weights/yolo26n.pt data=datasets/D-Fire/data.yaml \
  epochs=100 imgsz=640 batch=32 device=0 workers=8 \
  project=experiments/yolo26n name=D-Fire
```

## 评估（测试集）

```bash
yolo val model=experiments/yolo11n/D-Fire/weights/best.pt \
  data=datasets/D-Fire/data.yaml split=test imgsz=640
```

## 已完成的基线

| 算法 | 数据集 | epochs | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| yolo11n | D-Fire | 100 | 0.797 | 0.477 | 0.769 | 0.752 |

（D-Fire 论文中 YOLOv8n 的 mAP50 为 0.787，可作为参照。）

## 注意事项

1. **两个数据集的类别顺序相反**，不要混用标签：
   - `D-Fire`：`0=smoke`, `1=fire`
   - `FASDD_CV`：`0=fire`, `1=smoke`
2. **D-Fire 含 9838 张背景图**（空标签文件），是官方 "None" 类，属正常样本。
3. **离线环境已本地化**：预训练权重放在 `weights/`，Arial 字体放在 `~/.config/Ultralytics/Arial.ttf`（来自 Liberation Sans），无需联网。
4. `datasets/FASDD_CV/annotations/YOLO_CV/images` 是指向 `../../images` 的软链，供 `train.txt` 里的 `./images/...` 解析，**移动数据集时整个目录一起移动即可保持有效**。
5. 训练命令用 `yolo train`，不要用旧的 `yolo detect train` —— 后者会自动插入一层 `detect/` 目录，导致输出路径变成 `runs/detect/...`。
