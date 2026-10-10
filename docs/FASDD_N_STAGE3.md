# FASDD YOLO11n stage3

登记：2026-10-10。详细证据与实验设计：`analysis/reports/FASDD_N_STAGE3_REDESIGN_20261010.md`。

V2 context/detail_norm 的 seed0 收益为+0.219/+0.100点。新阶段主假设为定位质量评分错配；baseline 的 GT 质量排序诊断53.263%→59.857%仅为理想化对照，不是模型成绩。

| 实验 | 目标 | 当前状态 |
|---|---|---|
| Q0 frozen_quality_probe | 冻结原 n，只学习1,802参数的显式质量评分 | 已完成；主结果−0.501点，未通过筛选 |
| T1 reg_transfer | 独立检验回归塔/DFL预训练迁移 | cmj 已启动 seed0，100轮 |
| T2 compatible | 在T1后检验额外兼容头状态迁移 | 入口已准备，保留后续消融 |
| Q1 joint_quality | Q0有效后，同初始化的100轮配对实验 | 条件设计，未实施 |
| D0 train_label_audit | 按统一政策审核300张train图 | 清单已生成，人工审核pending |

可执行的初始化对照配置：`configs/protocol-fasdd-n-stage3.yaml`。
完整实验登记：`configs/experiment-fasdd-n-stage3.yaml`；后者是实验矩阵，不能直接作为所有行的训练入口。

```bash
python scripts/train_fasdd_n_stage3.py --method reg_transfer --seed 0 --dry-run
python scripts/train_fasdd_n_stage3.py --method reg_transfer --seed 0 --execute
python scripts/train_fasdd_n_quality_stage3.py --check
python scripts/train_fasdd_n_quality_stage3.py --execute
```

部署仍为n，结果保存到`results/fasdd_cv_clean_v1/fasdd-n-stage3/<method>/seedN/`。启动器前台运行，日志由外部supervisor保存到项目外runtime；每次正式训练独立保存源码快照和协议，T1另存迁移逐键记录。实际启动时间、PID、日志和验证范围见 `docs/FASDD_N_STAGE3_SEED0_RUNS_20261010.md`。

Q0 配置：`configs/protocol-fasdd-n-quality-stage3.yaml`。真实batch32已验证冻结模型及BN保持不变、优化器只更新readout、关闭质量与exponent=0逐元素精确复现、NMS前框保持、NMS后框/类别保持；三种模式另通过8张train内部图的原生验证器检查。这些小批次AP不是正式验证集成绩。

终局与新增只读审计：`analysis/reports/FASDD_STAGE3_FAILURE_AND_STAGE4_DESIGN_20261010.md`。Q0完整val主模式52.7624%，基线53.2633%；保留原框、只按q重排原TP50也退步0.869点。下一设计为轻量边界坐标修正与有可靠同数据教师后进行的定位蒸馏，见`configs/experiment-fasdd-n-stage4-design.yaml`，尚未实现或启动。
