# 已归档训练设计：协议化 baseline seed0

2026-09-19 更新：P2 seeds 0/1/2 和 SRDG seed0 均已完整归档；下文是历史快照。
用户已说明 baseline 在另一工作站完成，本轮不重跑。当前下一训练方案为
[DCBR 候选](MODULE_DCBR.md)，其比较父级为已完成的同协议 P2。

日期：2026-09-18。用户确认 baseline 已在另一台电脑完成，本机不再重复运行。本文保留为成对
实验设计记录；当前候选模块与下一训练入口见 [MODULE_SRDG.md](MODULE_SRDG.md)。另一台电脑的
baseline 结果同步后仍需核对本文定义的数据、权重、源码和训练指纹，不能仅凭目录名认定可比。

## 当前证据

| 运行 | 状态 | 最佳 val mAP50 | 最佳 val mAP50-95 | 备注 |
|---|---:|---:|---:|---|
| 历史 YOLO11n seed0 | 100 epochs | 0.79683 | 0.47758 | 完整 `.pt` 微调、optimizer=auto、AMP=true |
| 历史 YOLO11n seed1 | 100 epochs | 0.79679 | 0.47280 | 与论文协议初始化不同 |
| 历史 YOLO11n seed2 | 100 epochs | 0.78965 | 0.47219 | 与论文协议初始化不同 |
| 协议 P2 seed0 | 100 epochs | 0.80345 | 0.48384 | SGD、AMP=false、共享层 0..22 迁移 |
| 协议 P2 seed2 | 7 个已写入 epoch | 0.50274 | 0.24888 | 未完成，不纳入正式比较 |

历史三种子最佳 mAP50-95 为 `0.47419 ± 0.00295`（样本标准差）。P2 seed0 表面上高
`0.00965`，但两组同时改变了初始化、优化器、AMP 和规范化数据清单，因此不能把差值归因于 P2。
P2 还把理论计算量从 6.44 增至 10.14 GFLOPs@640（约 +57.4%），小幅精度收益未必足以支持
轻量部署主张。

## 与近期研究方向的对应

当前更有说服力的证据不是单数据集单次最高分，而是数据污染审查、多种子稳定性、跨数据集外部
验证以及精度—效率联合报告。多场景基准工作把数据构建与统一评估视为火烟检测的主要瓶颈；近期
跨数据集研究还报告了明显的 zero-shot 性能下降及 train/test 近重复污染风险。2026 年一项
YOLO11 火烟研究也采用双基准、多种子与部署 profiling，并展示了小幅改进可能对 seed 敏感。

因此本项目当前不增加新的注意力模块或损失堆叠，而先建立可归因的 baseline→P2 成对证据。相关
阅读：

- [Benchmarking Multi-Scene Fire and Smoke Detection](https://arxiv.org/abs/2410.16631)
- [Cross-Dataset Evaluation of YOLOv8 for UAV Fire and Smoke Detection](https://doi.org/10.3390/drones10080635)
- [Lightweight Fire and Smoke Detection with YOLO11: A Two-Benchmark, Multi-Seed Study](https://doi.org/10.3390/fire9090386)

## 决策

下一次运行 **D-Fire / baseline / seed0 / paper-v1**。它与已经完成的 P2 seed0 构成成对结构对照：

- 相同 canonical D-Fire 清单及类别映射；
- 相同预训练权重 SHA256；
- 相同共享层 0..22 迁移和随机 Detect 头；
- 相同 seed、SGD、学习率、增强、100 epochs、batch 32、AMP=false；
- 唯一预期变量是检测结构：baseline 使用 P3/P4/P5，P2 使用 P2/P3/P4/P5。

训练入口会检查成对 P2 的 `run.json`、portable 数据指纹、样本数、预训练权重、关键源码和完整
prepared 数据。任一关键项不一致就拒绝启动。

## 命令

先只检查，不训练：

```bash
python scripts/train_baseline_seed0.py --dry-run
```

研究者确认检查全部通过后手动训练：

```bash
python scripts/train_baseline_seed0.py
```

原始输出为 `outputs/paper-v1/dfire/baseline/seed0/`，成功后归档为
`experiments/yolo11n/D-Fire_protocol_baseline_seed0/`。旧的 `D-Fire_seed0` 是历史实验，不得覆盖或
改名冒充协议 baseline。

## 训练后判读

1. checkpoint 只按 val mAP50-95 选择；不从不同 epoch 拼接最好 P、R、AP。
2. 先比较同 seed 的 baseline 与 P2，报告 mAP50、mAP50-95、fire/smoke 分类别 AP 和训练时间。
3. 同时执行参数量、GFLOPs 和目标设备延迟；P2 的精度变化必须与额外计算成本一起呈现。
4. 不用 test 结果决定是否保留模块。结构冻结后才运行 D-Fire test 与 FASDD_CV zero-shot test。
5. 单个 seed 只能用于管线筛查。正式结论仍需 baseline/P2 的 seeds 0、1、2 均值、样本标准差及
   每 seed 原始值。
6. H1 若表述为“小目标改进”，必须先实现固定面积口径的尺寸分层 AP；总体 mAP 上升不能替代
   小目标证据。

## 后续顺序

1. 完成协议 baseline seeds 0、1、2。
2. 完成 P2 seed1；P2 seed2 当前目录是失败/中断运行，保留证据后以全新输出目录重跑，禁止把
   7 个 epoch 当作正式重复或静默续训。
3. 完成三种子配对后再决定 P2 是否进入 `p2_background` 与 `full`；若精度收益不能补偿计算开销，
   P2 只保留为负面消融，不作为主方法。
4. 背景损失优先在 baseline 上做 `background` 对照。现有 P2 seed0 每轮平均约 3806/15485 张
   增强后空图，说明损失有实际暴露样本，但 alpha 的有效性仍需验证集和固定误报预算检验。
