# FASDD_CV 新数据版本：fasdd_cv_clean_v1

已在 lxy（100.94.208.28）构建完成。新版本的数据完整性已校验，尚未启动训练；感知近重复与场景独立性仍待核验。

仓库：`/home/lxy/Documents/yolo11/Fire-YOLOv11`

## 数据规模

| 划分 | 图片 | 背景图 | 火焰框 | 烟雾框 |
|---|---:|---:|---:|---:|
| train | 44,113 | 19,525 | 33,137 | 22,869 |
| val | 29,804 | 13,019 | 20,984 | 15,474 |
| test | 15,488 | 6,501 | 8,312 | 7,733 |
| 合计 | **89,405** | **39,045** | **62,433** | **46,076** |

类别统一为 `0: fire`、`1: smoke`。本次仅使用 FASDD_CV，不包含 FASDD_UAV、FASDD_RS。

## 清理规则与排除记录

原始 FASDD_CV 共 95,314 张，本版本排除 5,909 张。下列原因互斥，每张图片只计入一个排除原因。

| 排除原因 | 数量 |
|---|---:|
| 与 D-Fire 任意划分的文件字节完全一致 | 5,733 |
| 与 D-Fire 的解码 RGB 像素一致、文件字节不同 | 6 |
| FASDD 跨划分文件字节重复 | 60 |
| FASDD 跨划分解码像素重复 | 11 |
| FASDD 同划分文件字节重复 | 90 |
| FASDD 同划分解码像素重复 | 6 |
| 原始无效标注图片 | 3 |

对 FASDD 内部的重复组，保留顺序为 test > val > train，再按原始 ID 排序；保留图片仍属于原来的官方划分，不重新分配。对与 D-Fire 重复的组，排除其中的 FASDD 图片，D-Fire 保持原样。

像素指纹包含 EXIF 方向校正后的 RGB 尺寸和完整像素。感知相似度仅用于筛查，不直接决定删除。

## 已完成的检查

- 对规范化 FASDD 和 D-Fire 的 116,812 张图片完成全量解码与文件、像素 SHA256 扫描，没有无法解码的图片。
- 新版本生成后，对全部 89,405 张保留图片和 D-Fire 参照图片再次校验文件 SHA256。
- 校验新版本 manifest、规范化标签、划分清单与图片链接。
- 新版本内部，以及与规范化 D-Fire 参照之间，字节和解码像素完全一致的冲突均为零。
- 原数据、原配置及原划分清单的哈希保持一致；现有仓库修改保留。
- 构建脚本的 9 项单元检查通过。训练计划能够读取新配置，返回 `training_started: false`。

图片使用软链接，标签为独立文件。这里的版本冻结是指清单和指纹固定，原图仍是共享文件，不能随意修改。跨机器迁移需复制原数据后重新建立链接。

## 近重复候选：尚未解决的场景泄漏问题

筛查条件为 pHash 汉明距离 ≤ 4、dHash 汉明距离 ≤ 4、32×32 灰度图全局 SSIM ≥ 0.95。共得到 **168,098 对候选**，并非已经确认的重复图片。

| 候选范围 | 配对数 |
|---|---:|
| FASDD train ↔ val | 111,089 |
| FASDD train ↔ test | 15,588 |
| FASDD val ↔ test | 39,386 |
| FASDD ↔ D-Fire（所有划分） | 2,035 |

涉及 FASDD train 11,434 张、val 10,319 张、test 4,928 张。没有把这些候选自动删掉，也没有据此宣称场景独立。该筛查不能保证发现所有裁剪、变形或相邻视频帧。

本版本可以作为可追溯的精确去重基线。正式实验前应优先核验跨 train/test、val/test 的候选，并据来源或视频场景建立分组划分。现有 FASDD/D-Fire 测试集此前已被查看，新版本不等于新的盲测集。

## 远端产物

- 数据入口：`/home/lxy/Documents/yolo11/Fire-YOLOv11/data/prepared/fasdd_cv_clean_v1/data.yaml`
- 训练协议：`/home/lxy/Documents/yolo11/Fire-YOLOv11/configs/protocol-fasdd-cv-clean-v1.yaml`
- 版本元数据：`data/prepared/fasdd_cv_clean_v1/version.json`
- 审计目录：`data/prepared/fasdd_cv_clean_v1/audit/`
- 构建脚本：`scripts/prepare_fasdd_clean_v1.py`
- 版本说明：`docs/FASDD_CV_CLEAN_V1.md`

审计目录含完整内容指纹、逐图排除理由、保留来源 ID、感知候选 CSV 与汇总信息。本地已保存配置、版本元数据、排除记录和候选 CSV 的副本。

内容指纹：`56d5ca7e69646edd5b9a3351c86bfa24b190cf37e01af1335721955e8d6a1ca2`

## 使用方式

以下使用远端 yolo11 环境，在仓库根目录执行。预览不启动训练：

```bash
/home/lxy/miniconda3/envs/yolo11/bin/python -m firesmoke \
  --config configs/protocol-fasdd-cv-clean-v1.yaml plan \
  --dataset fasdd_cv_clean_v1 --method baseline --seed 0 --tag fasdd-cv-clean-v1
```

日后检查数据是否发生变化：

```bash
/home/lxy/miniconda3/envs/yolo11/bin/python scripts/prepare_fasdd_clean_v1.py --verify-only
```

当前协议沿用 100 epochs、640 输入、batch 32、SGD、seed 0/1/2。尚未进行 GPU 训练，因此不代表这些参数已通过实际训练验证。应先完成场景候选核验，再确定正式训练版本与协议。
