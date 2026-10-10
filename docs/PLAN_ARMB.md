# Arm B 训练方案：D-Fire + FireSmoke-YOLO 训练负样本 ×3

> 设计日期：2026-10-02。所有数字来自本仓库已有 run / 标签 / 报告的实测，不含推测。
> 这是**待验证的数据侧候选**，不承诺精度提升，也不代表已满足任何期刊要求。
> 上游诊断见 `analysis/reports/实测诊断结论.md` 与 `analysis/reports/实验与训练设计_v1.md`。

## 0. 一句话

下一轮训练：`dfire_fsneg3` / `baseline` / seeds 0-2 / tag `armb-v1`
—— 在 D-Fire train 中并入 FireSmoke-YOLO 的 1,554 张训练负样本、每张重复 3 份，
**只改训练行集合，其余训练配置全部冻结**；主终点是跨域（FASDD_CV test）背景告警比例。

## 1. 为什么是这一轮

| # | 已实测的结论 | 数字 | 来源 |
|---|---|---|---|
| 1 | 结构模块路线已封闭：效应全部低于测量门槛 | P2 +0.68 / SRDG +0.39 / DCBR +0.45 / ACR +0.15pp vs 门槛 0.92–1.05pp | `out_power__power.md`、`out_modules__module_effects.md` |
| 2 | P2 有算术上界，不是运气问题 | 作用域 = val 框的 5.9%；修到完美对整体 mAP 的贡献 ≈0.14pp | `实测诊断结论.md` §5 |
| 3 | 分辨率已到顶 | 640→960 +1.21pp（唯一越过门槛）、1280 −2.69pp | 分辨率阶梯 4 点 |
| 4 | 缺口 G2：跨域火侧误报 | FASDD test 背景告警 9.5–12.3%，fire:smoke = 3.4:1 | `out_zeroshot__zeroshot_eval.md`、`out_false_alarm__false_alarm_attribution.md` |
| 5 | G2 的机制已人工确认 | 红发肖像 / 红球花 / 暖色花盆 / 橙色隧道灯带 / 雾中枯枝 —— 缺的是场景上下文，不是容量 | 最难负样本人工查看（6 张） |
| 6 | 重复 D-Fire 自己的背景**没有信息增量** | 源域 val 790 张背景图触发 **0** 张；真正惹祸的暖色物体不在 D-Fire 背景集里 | `实验与训练设计_v1.md` §1-B（`dfire_hn3` 已停） |
| 7 | 而 FireSmoke-YOLO 的负样本携带**新**失效模式 | 触发率 val 2.42% / test 3.33%，比 D-Fire 背景（0.25%）难约 10 倍 | `probe_negatives.py` |

结论：唯一还没被否证、且数据已在本地、成本可控的方向，是**用外来负样本提高背景梯度权重**。
`dfire_hn3`（重复 D-Fire 背景）与它的差别不在剂量，而在**样本是否携带新信息**。

## 2. 数据设计：`dfire_fsneg3`

| 项 | 值 |
|---|---|
| D-Fire train | 15,485 行（其中纯背景 7,043 = 45.5%），symlink 原文件 |
| FireSmoke-YOLO train 负样本 | 1,554 张（空标签），**每张重复 3 份** = 4,662 行 |
| 合并后 train | **20,147 行**（+30.1%），背景占比 45.5% → **58.1%** |
| val / test | 与 `dfire` **逐图一致**（1,722 / 4,294），不并入任何 FS 图 |
| 生成方式 | `scripts/prepare_armb_dataset.py`（D-Fire 部分 symlink；FS 负样本 hardlink，实测 4,662/4,662 全部 hardlink，零额外占用） |
| 近重复审计（接入前必做） | 1,554 张 FS 负样本 vs 15,499 张 D-Fire train：**0 张完全重复**（sha256）；20 张（1.3%）有 dHash ≤4 近邻候选，集中在少数低纹理 D-Fire 图上，属 dHash 假阳性，人工核验图见 `analysis/work/armb_duplicate_contact_sheet.png`。按仓库策略**不自动删除，只记录** |

**为什么不用 `neg_repeat`（关键实现约束）**
`firesmoke/data.py` 的 `neg_repeat` 会重复 train 划分内**所有**无 GT 框的图像。
D-Fire train 自带 7,043 张背景，用它会把这批无信息增量的图也一起 ×3，
把一个干净的「外来负样本剂量」实验变成两个变量的混合实验。
因此本项目改为**在源目录层面只重复 FS 负样本**，`dfire_fsneg3` 的 `neg_repeat` 保持默认 1
（`metadata.json` 已登记 `neg_repeat: 1`，见 `load_prepared` 的自校验）。

**类别顺序**：D-Fire 是 `0=smoke, 1=fire`，FireSmoke-YOLO 是 `0=fire, 1=smoke`。
本方案只并入 FS 的**空标签**图，不含任何类别标注，顺序差异不适用；
将来若并入 FS 的正样本标注，必须先在源目录层面换序（PROJECT_LAYOUT 坑 2）。

## 3. 训练配置（冻结，来自 `configs/protocol.yaml`）

与 `results/dfire/paper-v1/baseline` 完全相同，**唯一变量是训练行集合**：

```
epochs=100  imgsz=640  batch=32  optimizer=SGD  lr0=0.01  lrf=0.01
momentum=0.937  weight_decay=0.0005  warmup_epochs=3.0  amp=false
deterministic=true  close_mosaic=10  mosaic=1.0  mixup=0.0  scale=0.5
translate=0.1  fliplr=0.5  hsv_h=0.015  hsv_s=0.7  hsv_v=0.4
共享 YOLO11 层 0..22 迁移；检测头随机初始化；selection = best val mAP50-95
```

## 4. 对照与可比性（已核验，2026-10-02）

对照 = `results/dfire/paper-v1/baseline/seed{0,1,2}`（`paper-v1` 配置组），复用而不重训。

| 核验项 | 结果 |
|---|---|
| 数据清单（id + boxes）跨机器一致 | `portable_inventory_sha256 = 033e970451ffc052…`，三种子与当前 `data/prepared/dfire` **相同** |
| 预训练权重 | `sha256 = 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`，与三种子 `run.json` 的 `pretrained_sha256` 一致 |
| 训练超参 | epochs/imgsz/batch/optimizer/lr0/amp/deterministic/close_mosaic 全部一致（逐项比对 run.json） |
| seed0 与 seed1/2 之间的源码差异 | 仅涉及**新增** `configs/protocol_800.yaml`、`protocol_960.yaml` 与对应启动脚本；`firesmoke/`、`configs/protocol.yaml`、`tests/` 未变 → 三种子可并池 |
| **遗留差异（必须记录）** | 三种子的 `manifest_sha256` 互不相同（e57ad5e3… / fd72743e…）：manifest 里存的是**绝对路径**，跨机器/跨次 prepare 就会变。判定口径统一用 `portable_inventory_sha256`，不用 manifest sha |

## 5. 预登记：终点、门槛、判定与停止规则

**冻结阈值**（协议要求：只能在源域 val 上选，不做任何目标域调整）
沿用 `out_zeroshot__zeroshot_eval.md` 的口径 —— `p2/seed0` 的源域 D-Fire val 1% FPR：
`fire = 0.3143`、`smoke = 0.6057`。换成别的来源（如 baseline/seed0 的 0.3148 / 0.6060）**必须整表重算**。

| 角色 | 指标 | 门槛（MDD, α=.05） | 基线（待本轮补齐到三种子） |
|---|---|---|---|
| **主终点** | 跨域 `background_alarm_fraction`（FASDD test） | **1.37pp**（P2 组 σ=0.5522pp） | baseline seed0 = 9.81%；P2 三种子均值 = 10.25% |
| 守卫 | 域内 val `mAP50-95` 下降 | ≤ **0.92pp**（P2 组 σ=0.37pp） | 0.47706 / 0.47307 / 0.46929 |
| 守卫 | 跨域 `fire` 召回下降 | ≤ 6.32pp（σ=2.545pp，门槛很宽） | 0.9360 |
| 副终点 | 跨域 `smoke` 召回、跨域 `mAP50-95` | 2.26pp（后者） | 0.4211 / 0.3406 |

**判定（先登记，避免看着结果找理由）**
1. 背景告警改善 **≥3pp**（≈2×门槛）**且**域内 mAP50-95 下降 ≤0.92pp → 判定有效，进入阶段 2（换更强负样本，如 MS-FSDB 的 8,983 张）。
2. 改善 **<1.37pp** → 判定不可分辨，**停止「重复源域负样本」这条线**，转向任务重定义或换数据来源。
3. 落在 1.37–3pp 之间 → 记为边缘信号，需补第 4/5 个种子或换 MS-FSDB 后再判，不得直接宣称提升。
4. 域内 mAP50-95 下降 >0.92pp → 即使告警指标更好也**不采用**（必须两者同时报）。
5. 任何小于 MDD 的改善**不得称为提升**；三 seed 报 mean ± 种子 σ 与每 seed 原始值。

**统计口径**：配对设计（同 seed 的 Arm B − baseline），先用 `max(baseline 组 σ, P2 组 σ)` 估门槛，
训完后再用配对差本身的 σ 复核。检验统计量取 t₁₋α/₂(n−1)，n=3 时 t=4.303。

## 6. 执行顺序与命令

```bash
cd /home/lxy/Documents/yolo11/Fire-YOLOv11
P=~/miniconda3/envs/yolo11/bin/python
export PYTHONPATH=$PWD

# ① 数据准备（已完成，留档可复现；换机器时重跑，见 PROJECT_LAYOUT 坑 1）
$P scripts/prepare_armb_dataset.py          # 生成 data/raw/D-Fire-FSneg
$P scripts/prepare_armb_dataset.py --audit  # 近重复审计（接入前必做）→ analysis/work/armb_duplicate_audit.json
$P -m firesmoke --config configs/protocol.yaml prepare --dataset dfire_fsneg3

# ② 预检（只打印计划，不训练、不写盘）
$P -m firesmoke --config configs/protocol.yaml plan \
   --dataset dfire_fsneg3 --method baseline --seed 0 --tag armb-v1

# ③ 零 GPU：先把对照组的跨域基线补到三种子（锁定门槛，训练前做）
$P analysis/scripts/zeroshot_eval.py --out analysis/work/out_zeroshot_armb \
   --threshold-from p2_s0 --variance-groups p2,baseline640

# ④ 训练（GPU，由研究者手动启动；每个 seed 一次）
$P -m firesmoke --config configs/protocol.yaml train \
   --dataset dfire_fsneg3 --method baseline --seed 0 --tag armb-v1 --execute
#   seed 1 / seed 2 同法；输出落在 results/dfire_fsneg3/armb-v1/baseline/seed<N>/

# ⑤ 评估（训练后，零新增训练）
$P analysis/scripts/zeroshot_eval.py --out analysis/work/out_zeroshot_armb \
   --threshold-from p2_s0 --variance-groups p2,baseline640,armb --reuse
```

## 7. 成本

| 项 | 量 |
|---|---|
| 每轮 batch 数 | 630（20,147 ÷ 32，原 484） |
| 实测速率参照 | `dfire` 484 batch ≈ 67.6s/轮；`dfire_hn3` 924 batch ≈ 120s/轮 |
| 单次训练 | **≈2.2–2.5 h**（100 epochs） |
| 三种子合计 | **≈6.6–7.5 GPU 小时** |
| 评估 | ≈0.5 h（FASDD test 15,884 张 + 源域 val，零训练） |

## 8. 前置条件与风险

1. ⚠️ **本机 GPU 驱动当前不可用**：`nvidia-smi` 报 `couldn't communicate with the NVIDIA driver`，
   `/dev/nvidia*` 不存在（2026-10-02 05:30 实测）。训练前必须先恢复驱动；
   数据准备与预检不受影响。
2. ⚠️ **FASDD 的 train/test scene-disjoint 审计尚未做**（`analysis/scripts/split_integrity_audit.py`）。
   目标域结论的强度依赖它 —— 建议在训练跑的同时零 GPU 完成。
3. ⚠️ 训练期间确认没有其它进程占显存（cmj 上曾出现 `llama-server` 占 20GB、导致 5 小时里损失约 4 小时的先例）。
4. ❗ **诚实先验**：FS 负样本是「较弱版本」的目标失效模式（2.9% vs 目标域 ~10%，且只有 1,554 张）。
   预期改善 1–4pp，有较大概率可分辨，但**不像「必定有效」**。
5. ❗ FS 图像带中新网 / Alamy 水印，且与 D-Fire/FASDD 的同源关系未明。
   已实测：与 D-Fire train **无一张字节级重复**，20 张仅 dHash 近邻（候选，非确认）；
   结论与人工核验图要一并写进报告，水印在论文里要注明。
6. ❗ 若判定有效但幅度小（1.4–3pp），下一步是换 MS-FSDB 的 8,983 张负样本（独立数据集，
   不破坏 FASDD 的未见域地位），而不是继续加 D-Fire 自己的背景。
7. 回滚方式：训练输出目录不可覆盖（协议保护），换 `--tag` 即可另起一次；数据侧重跑脚本加 `--force`。

## 9. 本轮明确不做

- ❌ 不再加注意力 / 结构模块（P2、SRDG、DCBR、ACR 的效应全在门槛内，路线已封闭）
- ❌ 不重复 D-Fire 自身的背景图（`dfire_hn3`，实测无信息增量，已停）
- ❌ 不动输入分辨率（800–960 已到顶，1280 掉点）
- ❌ 不做小目标专项优化（8–16px 档 AP 2–3%，几何上不可达）
- ❌ 不用 FASDD train 当训练数据（会毁掉最重要的未见域；要用必须先另备未见域）
- ❌ 不靠单种子结论、不靠测试集调参、不因结果不利而删种子

## 10. 与仓库其他文档的关系

- 上游诊断与实验设计：`analysis/reports/实测诊断结论.md`、`analysis/reports/实验与训练设计_v1.md`
- 跨域评估：`analysis/reports/out_zeroshot__zeroshot_eval.md`、`out_reliability__fixed_threshold_probe.md`
- 历史（已被本轮取代）：`docs/NEXT_TRAINING.md`（P2 成对设计）、`docs/MODULE_*.md`、`docs/PLAN_ACR.md`
