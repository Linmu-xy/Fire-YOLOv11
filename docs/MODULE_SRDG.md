# SRDG 候选模块设计

2026-09-19 状态更新：seed0 已完成 100 epochs，最佳 val mAP50-95=.48098（epoch94），
同 seed 父级 P2 为 .48384；P 上升、R 下降，尚未证明固定工作点的误报改善。
新一轮候选及证据修正见 [MODULE_DCBR.md](MODULE_DCBR.md)。下文保留原始训练前设计记录。

## 研究动机

P2 检测层提高了空间分辨率，但浅层特征语义较弱，容易把灯光、云雾、反光和高对比纹理当成
小火焰或烟雾。经典 FPN 工作已指出高分辨率浅层特征存在语义不足；本项目的 P2 seed0 只证明
训练可行，尚未证明小目标收益，也未证明背景误报下降。

本轮不增加普通通道/空间注意力，而设计 **Semantic Residual Detail Gate（SRDG，语义残差细节
门控）**：利用已有 P3 语义对 P2 浅层细节做局部调制，专门处理 P2 引入的语义—细节失衡。

## 模块

令 `S` 为上采样到 P2 分辨率的 P3 语义特征，`D` 为 backbone P2 细节特征。两者各 64 通道。

```text
A  = tanh(Conv1x1(S))
D' = D * (1 + g*A),  g = 0.5
Y  = Concat(S, D')
```

`Conv1x1` 的权重和偏置初始化为 0，因此训练开始时 `A=0`、`D'=D`，模型严格退化为原 P2，
不会因随机门控破坏初始特征。`tanh` 和 `g=0.5` 把调制范围限制在 `[0.5, 1.5]`：模块可以抑制
语义不支持的纹理，也不能把弱细节完全归零。随后仍使用原 P2 的 DWConv、1×1 Conv 和 C3k2。
门控构造使用隔离的 RNG 上下文，不消耗全局初始化随机流；同一 seed 下，门控之后的 P2 与
Detect 参数和父级 P2 逐张量一致，使首轮差异只来自可学习门控本身。

实现文件：

- `firesmoke/models.py::SemanticResidualDetailGate`
- `configs/models/yolo11n-p2-srdg.yaml`
- `configs/protocol-v2.yaml`

自定义类只注册到当前 Python 进程中的 Ultralytics YAML 解析命名空间，不修改 site-packages。
上游版本继续固定为 8.4.42。

## 结构成本

CPU 随机张量前向与上游 profiler 检查结果：

| 结构 | 参数量 | GFLOPs@640 | stride |
|---|---:|---:|---|
| baseline | 2,590,230 | 6.4417 | 8/16/32 |
| P2 | 2,638,312 | 10.1378 | 4/8/16/32 |
| P2+SRDG | 2,642,472 | 10.3475 | 4/8/16/32 |

SRDG 相对 P2 增加 4,160 参数（约 0.16%）和 0.2097 GFLOPs（约 2.07%）。它没有解决 P2 相对
baseline 的总体计算开销，因此不能仅凭“模块轻量”宣称整个模型轻量。

## 可检验假设

- H-SRDG1：相对 P2，SRDG 降低纯背景图上的 alarm fraction 和 FP/image。
- H-SRDG2：在源域验证集固定 1% 经验 FPR 预算时，smoke 图像级 recall 不下降超过 1 个百分点。
- H-SRDG3：若总体 mAP 改善，收益在三个 seed 上方向一致，而非单 seed 偶然波动。
- H-SRDG4：冻结模型后，D-Fire→FASDD_CV zero-shot 的误报改善仍存在；否则只能称同域抑制。

H-SRDG2 的 1 个百分点是预先登记的工程非劣界值，不是统计显著性阈值。三种子仍需逐值报告，
不能只报告均值或只保留有利 seed。

## 第一轮实验

父级对照固定为已完成的 `D-Fire_p2_seed0`。第一轮只运行 `p2_srdg/seed0`，用于排除训练不稳定、
门控塌缩和明显 recall 损失，不作为论文结论。

```bash
python scripts/train_srdg_seed0.py --dry-run
python scripts/train_srdg_seed0.py
```

默认输出为 `outputs/module-srdg-v1/dfire/p2_srdg/seed0/`，成功后归档至
`experiments/yolo11n/D-Fire_p2_srdg_seed0/`。训练脚本会检查完整 prepared 数据、父级 P2 的
portable 数据指纹、样本数、预训练权重和全部训练超参。

## 判读与后续消融

1. seed0 只做稳定性筛查；通过后补 P2 与 P2+SRDG 的 seeds 1、2。
2. checkpoint 仍只按 val mAP50-95 选择；模块主终点是背景报警与 smoke recall，不以 mAP 替代。
3. 使用相同 checkpoint 同时报告 mAP50-95、每类 AP/P/R、负图报警、FP/image、ECE/Brier、GFLOPs
   和目标硬件延迟。
4. 模型选择完成前不查看 test；冻结后再做 D-Fire test 与 FASDD_CV zero-shot test。
5. 若 SRDG 有效，再新增 `P2+SRDG+background loss` 组合；在此之前不把多个模块同时加入，避免
   无法归因。
6. 若要把“小目标”写成贡献，必须补尺寸分层 AP。若要把门控写成机制贡献，应保存门控热图，
   对真小目标、灯光、云雾、反光等错误类别做盲法案例分析。

## 研究边界

SRDG 与语义引导、多尺度融合和门控思想存在大量先行工作。目前只能称“面向本课题假设的候选
实现”，不能在未完成系统检索和对照前声称首创。需要至少与原 P2、参数量匹配的普通 1×1 变换、
以及常见轻量注意力对照，才能判断收益来自语义门控机制还是额外参数。

阅读起点：

- [Feature Pyramid Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)
- [Cross-Dataset Evaluation of YOLOv8 for UAV Fire and Smoke Detection](https://doi.org/10.3390/drones10080635)
- [Decoupled Classification Refinement: Hard False Positive Suppression](https://arxiv.org/abs/1810.04002)
