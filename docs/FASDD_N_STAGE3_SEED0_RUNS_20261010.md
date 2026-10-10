# FASDD n stage3：两台机器独立验证

用户授权：2026-10-10，让两台机器分别验证不同方向。
固定数据：FASDD clean-v1，内容 SHA256 `56d5ca7e69646edd5b9a3351c86bfa24b190cf37e01af1335721955e8d6a1ca2`。
统一类别0=fire、1=smoke，部署模型保持 YOLO11n。

## cmj：T1 回归头预训练迁移

- 启动时间：2026-10-10 13:44:18（北京时间）。训练 PID：21636。
- 脚本：`scripts/train_fasdd_n_stage3.py`；配置：`configs/protocol-fasdd-n-stage3.yaml`。
- 方向：保留原0–22层迁移，再迁移43个兼容cv2/DFL状态张量、382,678个状态元素。原分类头初始化保持与控制一致；模型参数数仍2,590,230。
- seed0，640，batch32，SGD，100轮，AMP=false；训练配方与历史基线一致。
- 输出：`/home/cmj/Documents/FireSmoke/Fire-YOLOv11/results/fasdd_cv_clean_v1/fasdd-n-stage3/reg_transfer/seed0/`。
- 日志：`/home/cmj/Documents/FireSmoke/Fire-YOLOv11-runtime/fasdd_cv_clean_v1/fasdd-n-stage3/reg_transfer/seed0/supervisor_20261010T054418_070360Z/console.log`。
- 实际迁移政策已从结果中的`weight_transfer.json`核对；第一轮已完成，进入后续训练。

复现启动命令（已运行，不要重复启动或覆盖）：

```bash
python -u scripts/train_fasdd_n_stage3.py --method reg_transfer --seed 0 --execute
```

## lxy：Q0 冻结基线的质量评分

- 启动时间：2026-10-10 13:50:39（北京时间）。训练 PID：45006。
- 脚本：`scripts/train_fasdd_n_quality_stage3.py`；配置：`configs/protocol-fasdd-n-quality-stage3.yaml`。
- 冻结父模型：历史baseline seed0的best.pt；文件SHA256 `d29e5872ab7d9b4a7f33fe5bb29cd06ca883c7c01d36f655e0dbdbf77743a0e2`。
- 父模型的参数和BN均冻结为eval；只更新新增1,802参数。原n+readout共2,592,032参数，约增加0.070%。
- seed0，640，batch32，AdamW lr=.001、decay=.0001，固定5轮，无颜色或几何增强。
- 从train的44,113张图按manifest ID哈希划分：39,790拟合、4,323内部检查；原val/test不用于拟合或选择权重。划分SHA256 `ea3bbeebab1f3d668063660b4ce98dd5d22d3512472cc63e020bd42d43e4da80`。
- 目标：原TAL正样本与类别分数top128并集，对每类预测同类GT最大IoU，使用逐图归一化软BCE。
- 选择：固定5轮结束后，使用内部检查BCE最小的readout；不根据官方val挑轮次。
- 评估：固定640/rect=true/batch64/FP32/conf=.001/NMS=.7/max_det300，在完整29,804张val上依次关闭评分、原NMS后评分、NMS前评分。NMS前模式为预先指定主结果；NMS后保留原候选框作排序诊断，重评分后不再按conf阈值二次裁剪。
- 关闭评分的mAP50–95必须复现53.2633%（容差0.001个百分点），否则自动停止后续解释。
- 输出：`/home/lxy/Documents/yolo11/Fire-YOLOv11/results/fasdd_cv_clean_v1/fasdd-n-stage3/frozen_quality_probe/seed0/`。
- 日志：`/home/lxy/Documents/yolo11/Fire-YOLOv11-runtime/fasdd_cv_clean_v1/fasdd-n-stage3/frozen_quality_probe/seed0/supervisor_20261010T055039_465168Z/console.log`。
- 已进入实际拟合，首轮BCE下降是拟合信号，不能当作mAP提升。

复现命令（检查可重复；execute已运行，输出存在时拒绝覆盖）：

```bash
python scripts/train_fasdd_n_quality_stage3.py --check
python -u scripts/train_fasdd_n_quality_stage3.py --execute
```

## 检查证据与保存范围

真实CUDA batch32通过：冻结父参数和BN状态完全一致、父梯度为空、readout梯度有限、优化器只改变readout、质量关闭及exponent=0的检测张量精确复现、NMS前坐标精确保持、NMS后坐标与类别精确保持。另在8张train内部图上经过原生AutoBackend融合及三种模式的指标流程。

完整检查JSON保存在训练结果`run.json`及项目外fit/check runtime。8张图的AP仅验证代码流程，不是正式val结果。没有评估test。

两边均保存独立源码快照、启动器、协议、启动记录与终态；cmj另保存迁移逐键决定。Q0每轮保存last/best readout、优化器和RNG等状态，CSV逐轮flush，父模型每轮检查不变。当前没有实现或验证Q0自动续训入口，暂停后需按保存状态恢复，不能直接重跑execute。

训练脚本、配置和核心代码已同步到两台机器；运行时各自使用已加载代码和启动时快照。cmj在Q0集成完成前启动，其快照保留当时版本。所有日志/临时文件在项目外，旧实验结果保留。

全val完成后再判断收益：主mAP50–95至少+0.5点才考虑扩大训练预算，并同时查看fire AP75及smoke退步。此门槛是预算规则，不是统计显著性或+5点承诺；多seed确认仍需后续进行。
