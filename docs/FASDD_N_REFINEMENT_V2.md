# FASDD n 模型实验 v2

登记日期：2026-10-09。依据：`analysis/reports/FASDD_N_REFINEMENT_POSTMORTEM_20261009.md`。

## 首轮执行

| 机器 | 方法 | seed | 目的 |
|---|---|---:|---|
| cmj | context | 0 | 原 context 算法的干净重跑 |
| lxy | detail_norm | 0 | 验证带归一化的短细节旁路是否能够学习 |

tag 固定为 `fasdd-n-refinement-v2`，结果位于 `results/fasdd_cv_clean_v1/fasdd-n-refinement-v2/<method>/seed0/`。

参照已完成 baseline seed0，mAP50–95=53.265%；baseline seed1=53.530%仅供种子差异参考。v1 被续训重置污染的模块结果保留为历史记录。

## 固定条件

保持 FASDD clean-v1 内容 SHA256 `56d5ca7e69646edd5b9a3351c86bfa24b190cf37e01af1335721955e8d6a1ca2`、官方保留划分、fire=0/smoke=1。

YOLO11n，640、batch32、100轮、SGD、lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup3、AMP=false、mosaic1、close_mosaic10、deterministic=true。0–22层从相同预训练权重迁移，预测头从头初始化。

训练验证维持官方 Trainer 的 rect=true、batch64、conf=.001、NMS IoU=.7、max_det300、half=false。候选由val选取。

## detail_norm 结构

```text
P2 stride4 /64ch
 → PixelUnshuffle2 /256ch
 → 1×1 Conv + BN + SiLU /16ch
 → 3×3 depthwise Conv + BN + SiLU /16ch
 → 1×1 zero-initialized output /64ch
 → 加入 P3 回归特征
```

保留原检测器的三层64/128/256通道、stride8/16/32、分类宽度64、reg_max16和8,400个预测位置。分类输入直接保持原特征。新增5,328参数，总2,595,558（约+0.206%）。

这是短路径与归一化一起改变的设计候选；本轮不能分别归因两者贡献。隐藏宽度16、零输出初始化固定，r32、小幅非零输出、分类旁路和门控分别留待独立对照。

安装模块保持原预测器张量和全局初始化RNG；初始预测与父模型逐元素一致。

## 学习监测

两种方法每轮前两个真实训练 batch 记录至结果目录的 `refinement_learning.jsonl`：

- 原特征RMS、回归/分类残差RMS及比例。
- 模块参数RMS。
- 参数反向钩子捕获的任务梯度RMS，位于优化器权重衰减和裁剪之前；训练使用AMP=false。

统计不读取额外训练批次、不使用随机抽样，不改变损失和优化器。验证/warmup前向不计入训练信号。早期零输出会令第一步上游梯度为零，关注后续是否出现有效任务梯度和持续非零残差。

## 续训

`ResearchTrainer.get_model` 在 resume 模式先构造完整图，再 strict=True 恢复所有状态张量；首次初始化才使用 shared_transfer。`scripts/resume_fasdd_n_refinement.py` 支持新tag，继续严格检查暂停记录、源代码指纹、数据、完整检查点和CSV轮次。

未来续训会在仓库外runtime保存输入检查点 `resume_input.pt`，并写入完整权重恢复证明 `model_restore.json`。暂停时应先安全保存检查点及其副本，保持本版本源码不变。

## 入口与保存

```bash
python scripts/train_fasdd_n_refinement_v2.py --method context --seed 0 --dry-run
python scripts/train_fasdd_n_refinement_v2.py --method context --seed 0 --execute

python scripts/train_fasdd_n_refinement_v2.py --method detail_norm --seed 0 --dry-run
python scripts/train_fasdd_n_refinement_v2.py --method detail_norm --seed 0 --execute
```

配置为 `configs/protocol-fasdd-n-refinement-v2.yaml`。启动器、协议、源码指纹与源码快照保存至各实验launch目录；原始日志、代码备份和小夹具验证检查点在项目外runtime。

验证入口 `analysis/scripts/validate_fasdd_n_refinement_v2.py`：初始恒等、RNG/头权重保持、CUDA batch32/640损失与梯度、序列化/融合、可选ONNX，以及真实Trainer检查点恢复。夹具仅16张train/16张val，不以其AP判断模型提升。

本轮重点比较整体和fire AP75/AP50–95、后续按16–64px火焰与混合烟雾分组。seed0有收益后再登记配对seed1/2；不会把单seed结果当作稳定5点收益。
