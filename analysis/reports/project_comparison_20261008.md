# cmj / lxy 项目比较与 FASDD_CV clean v1 重建

比较日期：2026-10-08（北京时间）。比较使用实际文件 SHA256，包含核心代码、协议、训练脚本、测试、文档、分析脚本/报告、预训练权重及已完成实验结果。

## 项目状态

| 项目 | cmj | lxy |
|---|---|---|
| 仓库路径 | `/home/cmj/Documents/FireSmoke/Fire-YOLOv11` | `/home/lxy/Documents/yolo11/Fire-YOLOv11` |
| 当前分支 | main | main |
| 当前提交 | `21daf8c7943c872d8dbdd36b3dd283f3ca98301f` | 同左 |
| 比较前工作区 | 干净 | 4 个已跟踪文件修改，新增方案/数据版本/训练结果文件 |
| 代码、配置、文档和预训练文件 | 110 个 | 119 个 |
| 已完成结果文件 | 517 个 | 517 个 |

两边共同拥有的 110 个代码/配置/文档/预训练文件中，**106 个逐字节一致，4 个不同**。全部 517 个已完成结果文件逐字节一致，包括已有模型权重。lxy 正在运行的 FASDD seed0 输出单独处理，未纳入稳定结果的内容比较。

训练核心 `firesmoke/`、共有模型定义、已有训练启动器、预训练权重一致。没有发现 cmj 的核心训练实现落后。

## 四项已有文件差异

| 文件 | 差异与影响 |
|---|---|
| `configs/protocol.yaml` | lxy 增加 Arm B 的 `dfire_fsneg3` 数据配置；其余已有数据和训练设置一致。新 FASDD 协议仅保留 dfire 与新版本数据，因此可在 cmj 独立构建相同的新协议。 |
| `analysis/scripts/zeroshot_eval.py` | cmj 仍使用 `outputs/`、`experiments/` 旧权重路径，与当前目录规范不符，可能静默跳过权重；lxy 改为 `results/`，补齐 baseline seed1/2 与 Arm B 配置，增加阈值来源和方差组参数。 |
| `README.md` | lxy 补充 Arm B 方案入口。 |
| `docs/NEXT_TRAINING.md` | lxy 增加将原方案归档、转向 Arm B 的说明。 |

评估脚本是最明显的落后项。lxy 的版本仍有已发现的评价口径问题：在逐模型源域报警统计中复用阈值来源模型的数组，且将置信区间半宽称作 MDD。不能把 lxy 文件较新理解为这些问题已修复；本次比较不自动用它替换 cmj 版本。

## lxy 独有的九个文件（比较前）

```text
configs/protocol-fasdd-cv-clean-v1.yaml
docs/FASDD_CV_CLEAN_V1.md
docs/FASDD_CV_CLEAN_V1_BASELINE_SEED0.md
docs/FASDD_CV_CLEAN_V1_NOTES_ZH.md
docs/PLAN_ARMB.md
scripts/prepare_armb_dataset.py
scripts/prepare_fasdd_clean_v1.py
scripts/train_fasdd_cv_clean_v1_baseline_seed0.py
tests/test_fasdd_clean_v1.py
```

Arm B 方案属于另一条实验路线；本次在 cmj 构建 FASDD 新版本，无需更改其默认 `protocol.yaml` 或复制 Arm B 原始数据。

数据目录也有差异：lxy 另有 `data/raw/FireSmoke-YOLO/`、`data/raw/D-Fire-FSneg/`，以及 `data/prepared/fsyolo/`、`data/prepared/dfire_fsneg3/`；cmj 没有这些数据。这不会阻止 FASDD 基线，但 cmj 不能直接运行这些数据对应的训练。两边虽然都保存了 fsyolo 的历史结果，保存结果不代表原始数据也已同步。

## 数据与环境

- 两边原始 FASDD_CV 图片数均为 95,314，官方 train/val/test 清单 SHA256 一致。
- 两边已有规范化 fasdd 的 portable inventory 指纹一致：`af1a353ee854c3143710cc7540863af35ab23f270e332fa173a4a9d27386f7ca`。
- 两边 dfire 的 portable inventory 指纹一致：`033e970451ffc052cf13f66113bb65dfc48e82c3984f89a0a73e0a34f8bbaf29`。
- 已有清单一致不等于全部原图已逐字节比较；新版本重建会执行全量解码、内容指纹和保留图片校验，并与 lxy 的版本指纹对照。
- 关键包版本一致：ultralytics 8.4.42、torch 2.11.0、torchvision 0.26.0、numpy 2.2.6、Pillow 12.2.0、PyYAML 6.0.3。
- 两边 GPU 均为 RTX 3090。驱动不同：cmj 580.82.07（报告 CUDA 13.0），lxy 595.91.07（报告 CUDA 13.2）；包版本一致不代表驱动环境完全相同。
- 两机操作系统时区不同，因此报告统一使用北京时间；版本与内容比较不以文件显示时间判断新旧。

## 重建状态

目标：cmj 的 `data/prepared/fasdd_cv_clean_v1/`，89,405 张图片，内容指纹应为：

`56d5ca7e69646edd5b9a3351c86bfa24b190cf37e01af1335721955e8d6a1ca2`

**cmj 已构建完成，状态为 `built_and_integrity_verified`，完整构建扫描耗时 468.88 秒（约 7 分 49 秒，不含网络中断等待）。** 上传阶段本机 Tailscale 一度返回 `NoState`，用户重启服务后连接恢复。

构建已于北京时间 2026-10-08 11:58:42 在 cmj 启动。监督 PID 3361，构建 PID 3362，日志目录为：

`/home/cmj/Documents/FireSmoke/Fire-YOLOv11-runtime/dataset-builds/fasdd_cv_clean_v1/20261008T035842Z/`

已完成构建与逐项核对：

| 核对项 | 结果 |
|---|---|
| FASDD 规范化来源 95,311 张的文件、RGB 像素、尺寸和 ID 摘要 | 两边一致 |
| 新版本 89,405 张的 ID、划分、标注、文件与像素指纹 | 两边一致 |
| train / val / test 数量 | 44,113 / 29,804 / 15,488，两边一致 |
| 内容、生成协议、构建脚本 SHA256 | 两边一致 |
| 逐图排除清单、保留来源 ID 清单 | 文件逐字节一致 |
| 近重复候选的图片配对 | 168,098 对，全部一致 |
| cmj 全量图片再校验、标签/清单/链接校验 | 通过 |
| cmj 基线启动器 `--dry-run` | 通过；GPU 0 可用，训练未启动 |
| 原 cmj 配置、原数据清单 | 构建前后保持一致 |

cmj 数据入口：

`/home/cmj/Documents/FireSmoke/Fire-YOLOv11/data/prepared/fasdd_cv_clean_v1/data.yaml`

新增 FASDD 配置、构建脚本、seed0 启动脚本与对应文档已按规范保存。此次在 cmj 构建数据并预检，实际 seed0 训练仍在 lxy。

### 额外发现：D-Fire test 有 11 张图片内容不同

21,501 张规范化 D-Fire 参照中，**21,490 张的文件和像素一致，11 张的文件 SHA256 与 RGB 像素 SHA256 不同，全部属于 test**。其 ID、尺寸与标注一致；其中 WEB11315 的 dHash 也有差异。这解释了为什么两边近重复候选配对相同，但部分相似度数值和候选 CSV 文件哈希不同。

```text
WEB11243.jpg  WEB11244.jpg  WEB11245.jpg
WEB11313.jpg  WEB11315.jpg  WEB11316.jpg
WEB11319.jpg  WEB11320.jpg  WEB11321.jpg
WEB11322.jpg  WEB11323.jpg
```

抽查 WEB11243 与 WEB11315：cmj 文件分别为 262,646 / 308,506 字节，lxy 分别为 112,621 / 130,504 字节；两边 JPEG 量化表不同，cmj 文件末尾为 `ffd9`，lxy 文件末尾有附加内容且保留 EXIF/XMP/ICC 等信息。**推测 cmj 的文件曾被 JPEG 修复流程重编码**，历史原因尚未直接证实。当前安装的 Ultralytics 会对末尾不是 `ffd9` 的 JPEG 调用 Pillow，以 quality=100、subsampling=0 重新保存；这一机制与抽查特征一致。

这 11 张 D-Fire test 的差异没有改变本次 FASDD 新版本的选图、排除清单或内容指纹。跨机器严格复现 D-Fire 评估时应固定同一份图片版本，并记录这项差异；直接运行默认图像扫描也可能触发 JPEG 修复。当前保留两边原图与历史审计，不用某一份图片覆盖另一份。

`version-identity-comparison.json` 的全证据一致标志为 false，原因是这项 D-Fire 参照差异及其候选数值差异，**并非 FASDD clean v1 构建失败或训练数据不一致**。

## 原始证据

本地比较目录保存 `cmj-before.json`、`lxy-before.json`、`comparison-before.json` 和双方差异文件副本；仓库外也分别保存两台机器的比较前完整文件清单。未覆盖历史实验结果，未修改 lxy 正在训练的核心代码。

构建后再次清点：cmj 120 个、lxy 122 个代码/配置/文档/预训练文件；**116 个共同文件一致，仍有上述 4 个已有文件差异，lxy 独有文件只剩 Arm B 的两项**（`docs/PLAN_ARMB.md`、`scripts/prepare_armb_dataset.py`）。517 个已完成结果文件仍全部一致。两边的 FASDD 配置、脚本和文档已经同步。

本地另保存 `cmj-version.json`、两边的版本身份摘要、`version-identity-comparison.json`、`dfire-source-differences.json`、`candidate-pair-comparison.json`、`cmj-preflight.txt`、两边构建后摘要和 `comparison-after.json`。完整构建后清单分别保存在各自仓库外 `Fire-YOLOv11-runtime/project-comparison-20261008/`。

在 lxy seed0 运行期间再次执行只读 `--verify-only`，全量保留图片与 D-Fire 参照文件校验仍通过，训练数据与构建时的指纹一致。训练日志没有 JPEG 自动修复记录。
