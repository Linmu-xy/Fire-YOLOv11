# 操作手册

本框架不自动启动训练。下面的 `--execute` 命令由研究者确认配置后手动执行。
从仓库根目录运行；所有配置中的相对路径相对于仓库根目录解析。

## 环境

```bash
conda activate yolo11
python -m firesmoke --help
python -m unittest discover -s tests -v
```

已在 Python 3.10、Ultralytics 8.4.42、PyTorch 2.11.0 环境测试。
适配器固定 `ultralytics==8.4.42`，其他版本会报错，以免上游损失接口变化造成静默错误。
已有环境无需安装；可选 `python -m pip install -e . --no-deps` 注册 `firesmoke` 命令。
新环境先按硬件安装匹配的 PyTorch/torchvision，再安装本项目。历史 `requirements.txt`
是原环境完整快照，包含平台相关 CUDA 包，不建议在另一台机器直接照搬。

`weights/yolo11n.pt` 必须预先存在。默认离线，不隐式下载权重或安装依赖。
当前训练入口仅支持 CPU/单 GPU，不支持未验证的 DDP、断点续训。

## 数据审查与准备

原始数据保持不变。默认输出目录 `artifacts/` 不进入 Git。

```bash
# 标签、数量、同路径 split 重叠检查；不加载图片像素。
python -m firesmoke audit --output artifacts/audit/metadata-user

# 正式论文必须另外完成内容/近重复审查，可能耗时较长。
python -m firesmoke audit --output artifacts/audit/content-user --hashes --radius 4

# 创建软链接图片、独立的统一标签、清单及元数据。
python -m firesmoke prepare --dataset dfire
python -m firesmoke prepare --dataset fasdd
```

规范化后均为 `0=fire, 1=smoke`。严禁把原 D-Fire 权重的输出编号直接解释为该顺序。
原始空标签保留。当前配置排除包含零面积/非法归一化框的**整张图像**，排除清单保存在
`artifacts/data/<dataset>/excluded.json`。原因、原标注和原 split 均可追溯。
几何上略跨图像边缘、但 xywh 各分量合法的框保留并报告，不静默裁剪标签。

输出目录必须不存在，防止覆盖已经用于实验的数据。重新准备时在配置中使用新版本目录。
数据软链接是本机绝对路径；搬到新机器必须重新 prepare，不能直接复制生成的软链接。
`load_prepared` 会检查清单哈希、split 列表、标签和图片路径。内容审查是另外一项必要检查，
准备步骤不声称验证过所有图片解码，也不声称已消除近重复泄漏。

## 预览与 CPU 模型检查

```bash
python -m firesmoke plan --dataset dfire --method full --seed 0
python -m firesmoke check-model --method baseline --output artifacts/checks/baseline-user.json
python -m firesmoke check-model --method full --output artifacts/checks/p2-user.json
```

`plan` 不要求准备完数据，不导入训练器。`check-model` 只做随机张量前向、共享权重加载和
参数量/FLOPs 检查，既不创建优化器，也不进行参数更新。

## 手动训练

先审查 `configs/protocol.yaml`，尤其是设备、batch、epochs、优化器和标注处理规则。
下面命令**会训练**，本次开发未执行。

```bash
python -m firesmoke train --dataset dfire --method baseline --seed 0 --tag paper-v1 --execute
python -m firesmoke train --dataset dfire --method p2 --seed 0 --tag paper-v1 --execute
python -m firesmoke train --dataset dfire --method background --seed 0 --tag paper-v1 --execute
python -m firesmoke train --dataset dfire --method p2_background --seed 0 --tag paper-v1 --execute
python -m firesmoke train --dataset dfire --method full --seed 0 --tag paper-v1 --execute
```

正式实验分别换成 seeds 1、2，以及 dataset `fasdd`。完整矩阵为 2 数据集 × 5 方法 × 3 seeds
= 30 次训练；建议先在源域验证集上检查 baseline/background 是否值得继续，不用测试集挑模块。
如显存不足，统一下调全部方法的 batch 并重新登记协议。不要只为某个方法更改分辨率或训练步数。

输出示例：

```text
outputs/paper-v1/dfire/full/seed0/
  run.json                    # 配置、源码哈希、环境、Git、权重/数据清单指纹
  protocol.yaml               # 原始协议快照
  weight_transfer.json        # 实际迁移的权重键
  background_exposure.jsonl   # 每轮经过增强后真正为空的图像数量
  args.yaml / results.csv
  weights/best.pt / last.pt
```

目录已经存在就拒绝启动，请使用新 tag；不自动续训或覆盖。不要使用通用 `yolo train` 启动本框架
的消融，否则自定义训练期损失和增强不会生效。原 `scripts/train_seed*.py` 仅作历史基线保留。

## 冻结模型后评估

以下以 D-Fire 源域、full、seed0 为例。其他方法和 seeds 使用相同流程。

```bash
python -m firesmoke evaluate \
  --weights outputs/paper-v1/dfire/full/seed0/weights/best.pt \
  --dataset dfire --split val --device 0 --output artifacts/eval/dfire-full-s0-val

python -m firesmoke calibrate \
  --predictions artifacts/eval/dfire-full-s0-val/predictions.json \
  --max-fpr 0.01 --output artifacts/eval/dfire-full-s0-calibration.json

python -m firesmoke evaluate \
  --weights outputs/paper-v1/dfire/full/seed0/weights/best.pt \
  --dataset dfire --split test --device 0 --output artifacts/eval/dfire-full-s0-test

python -m firesmoke evaluate \
  --weights outputs/paper-v1/dfire/full/seed0/weights/best.pt \
  --dataset fasdd --split test --device 0 --output artifacts/eval/dfire-to-fasdd-full-s0-test

python -m firesmoke reliability \
  --predictions artifacts/eval/dfire-to-fasdd-full-s0-test/predictions.json \
  --calibration artifacts/eval/dfire-full-s0-calibration.json \
  --bootstrap 1000 --output artifacts/eval/dfire-to-fasdd-full-s0-reliability.json
```

同域测试也使用同一源域 calibration 文件。反方向实验在 FASDD 上训练、在 FASDD val 校准、
在 D-Fire test 测试。禁止使用目标域标签重新调阈值后称为 zero-shot。
评价器要求权重旁有 `run.json`，并拒绝旧类别顺序权重，防止静默混用历史实验。

评估分两次推理：上游验证器计算 AP，另一次预测保存逐图结果（含没有预测框的背景图）。
两次使用相同 conf=0.001、NMS IoU=0.7、max_det=300、rect=false、FP32、无 TTA。
置信度截断和 max_det 的影响必须在论文中说明。

## 汇总与速度

```bash
python -m firesmoke summarize \
  --metrics artifacts/eval/dfire-full-s0-test/metrics.json \
            artifacts/eval/dfire-full-s1-test/metrics.json \
            artifacts/eval/dfire-full-s2-test/metrics.json \
  --output artifacts/tables/dfire-full.json

python -m firesmoke benchmark \
  --weights outputs/paper-v1/dfire/full/seed0/weights/best.pt \
  --device cpu --output artifacts/bench/full-cpu.json
```

汇总要求 seeds 恰好为 0、1、2，拒绝同一权重冒充独立 seed。
默认基准是 batch=1、FP32、融合后模型、50 次预热和 200 次测量，保存所有样本。
它仅测 **forward**，不包含预处理、NMS、解码与报警；不能把该 FPS 称为系统端到端 FPS。
真实边缘硬件、视频报警时延和量化是后续实验，尚未在本版本自动化。
