# FASDD n refinement v2：实现与验证

日期：2026-10-09。结论：两台RTX3090的工程检查通过，context与detail_norm的seed0正式训练已启动。

## 改动

- `firesmoke/detail_refinement_v2.py`：PixelUnshuffle后的短卷积旁路，含BN/SiLU，输出加入P3回归；保持原检测头和初始化RNG。
- `firesmoke/checkpoint_state.py`及`models.py`：完整状态恢复纳入ResearchTrainer，初始化与resume分别处理。
- `firesmoke/refinement_monitor.py`：每轮前两个训练batch记录残差、任务梯度和参数RMS。
- `configs/protocol-fasdd-n-refinement-v2.yaml`、`scripts/train_fasdd_n_refinement_v2.py`：固定新tag、原640/batch32/100轮配方和数据指纹。
- `scripts/resume_fasdd_n_refinement.py`：读取对应tag的登记协议，保留输入检查点副本，使用Trainer的严格恢复证明。

## 新分支预算

detail_norm r16新增5,328参数，总2,595,558，较baseline增加0.206%；工具计算640输入6.5106944 GFLOPs。原三层64/128/256输入、分类宽度64、reg_max16和预测位置保持原设计。

## 验证结果

两台均执行 `analysis/scripts/validate_fasdd_n_refinement_v2.py`：

| 检查 | 结果 |
|---|---|
| 初始预测恒等、原头张量与全局RNG保持 | 通过 |
| CUDA batch32 /640实际检测损失与反向 | 通过，所有损失/梯度有限 |
| 新分支输出投影任务梯度 | L2=4.66538，非零 |
| batch32峰值分配显存 | 7,747,188,224 bytes，约7.22 GiB |
| 保存重载与完整模型重建 | 512个状态张量一致 |
| 推理融合 | 最大绝对误差9.1553e-5，在登记容差内 |
| ONNXRuntime FP32对齐 | 最大绝对误差9.1553e-5，在登记容差内 |

真实Trainer夹具每个方法使用16张train、16张val，先运行第一轮，在optimizer剥离之前保存输入检查点，再resume至第3轮：

| 恢复核对 | context /cmj | detail_norm /lxy |
|---|---:|---:|
| 训练模型状态张量完全一致 | 517 | 512 |
| EMA状态张量完全一致 | 517 | 512 |
| 优化器状态张量完全一致 | 273 | 262 |
| EMA updates | 4 | 4 |
| 恢复start_epoch（零基） | 1 | 1 |
| 完成轮数 | 3 | 3 |
| 诊断记录数 | 6 | 6 |
| 最大内部任务梯度RMS | 4.1260e-5 | 0.005521 |
| 最大输出投影任务梯度RMS | 0.013934 | 0.217706 |

本检查证明恢复时模型/EMA/优化器/轮次的一致性，没有声称中断与连续训练整条轨迹逐位相同。夹具AP不用于判断候选准确度。两种不同模块的梯度幅值也不能直接当作性能排序。

## 正式训练预检

两台确认89,405个prepared清单成员的标签、路径和划分一致；train44,113、val29,804、test15,488。内容SHA256为`56d5ca7e69646edd5b9a3351c86bfa24b190cf37e01af1335721955e8d6a1ca2`，原始数据协议SHA为`2aeeb813192fb5d2a93db778e77c5588ed024c3fd2ca5038b970ed11f1369f12`。

预训练权重两边SHA256均为`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。新协议稳定摘要`930f907a030f00f04e2b8656746c5b3ba76afe0351fa41d6f54e9806484a0713`相同；配方、数据指纹与原基线一致。

正式训练启动前再次核对全部发布文件SHA256与验证时一致。代码与小夹具备份在两边项目外的`Fire-YOLOv11-runtime/v2_rollout_20261009_*`、`validate_v2_*`。

## 证据保存

本地`analysis/artifacts/fasdd_n_refinement_v2_20261009/`包括：

- cmj/lxy_inventory.json：修改前的核心源码和运行状态。
- source_manifest.json、v2_source.tar.gz：发布代码与摘要。
- cmj/lxy_rollout.json：远端备份、验证进程和日志位置。
- cmj/lxy_validation.json：完整工程验证结果。
- cmj/lxy_preflight.txt：正式训练预检。
- cmj/lxy_launch.json、cmj/lxy_startup.json：正式训练启动及实际批次运行证据。

启动记录另见`docs/FASDD_N_REFINEMENT_V2_SEED0_RUNS_20261009.md`。
