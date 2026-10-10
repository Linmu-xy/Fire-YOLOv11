# FASDD：YOLO11n与现有模块检查

日期：2026-10-08。用户明确限制在n规模上提升，后续以n结构模块实验为主。本文是读取代码、完成检查点和历史记录的静态检查；本轮未做模型前向、训练或实现测试。

## 1. 实际检查对象

- lxy：`/home/lxy/Documents/yolo11/Fire-YOLOv11`，FASDD clean v1 baseline seed0 best.pt。
- cmj：`/home/cmj/Documents/FireSmoke/Fire-YOLOv11`，同协议baseline seed1 best.pt。
- 两边`firesmoke/*.py`及9份现有模型YAML的SHA一致，head配置一致。Ultralytics8.4.42。
- 读取`models.py`、`experiment.py`、`aligned_context.py`、`band_refinement.py`、P2/SRDG/DCBR/ACR配置，以及安装版本Detect/C2PSA/检测损失/TaskAlignedAssigner的代码。
- 数据错误分布沿用完整val诊断，不重新训练，也不读取test筛选新模块。

## 2. 真实n模型结构

| 层 | 作用 | 通道 | 640方形输入下空间尺度 |
|---|---|---:|---:|
| 2 | backbone P2 | 64 | 160×160，stride4 |
| 16 | neck P3 | 64 | 80×80，stride8 |
| 19 | neck P4 | 128 | 40×40，stride16 |
| 22 | neck P5 | 256 | 20×20，stride32 |
| 23 | Detect | 输入64/128/256 | 三层共8,400位置 |

两个完成检查点均为**2,590,230参数**；分类中间宽度64/64/64；回归reg_max=16，end2end=false，legacy=false。Detection head本身431,062参数。baseline并未缓存layer2，现有save索引为4/6/10/13/16/19/22；使用浅层旁路必须显式增加layer2缓存。

P5已有SPPF和C2PSA注意力，neck已有上下行融合。不能把新模块动机写成“原模型没有上下文或多尺度融合”。需要检验的是如何让已有语义和浅层细节更适合具体分类/定位任务。

## 3. 现有模块的作用位置和证据

| 模块 | 显式修改位置 | 已完成历史结果 | 检查结论 |
|---|---|---|---|
| P2 | 新增stride4分支 | D-Fire三seed相对baseline平均+0.586点 | 参数+1.86%，历史FLOPs估计约+57.4%；计算成本需单独看 |
| SRDG | P2拼接后，仅调制细节 | seed0相对P2 −0.286点 | 粗语义1×1门控，没有直接修改P3/P4/P5 |
| DCBR | P2细节的平滑残差 | seed0 −0.224、seed2 +0.468点 | 方向不一致；不是已证明的火/烟频带分离 |
| ACR | P2局部对齐与多尺度细节 | seed0相对P2 −0.530点 | 只直接修改P2，内部消融未完成，不能认定对齐贡献 |

原P2输出32通道，导致该版本Detect所有分类中间层从64变32；这是确定的结构混杂，准确度影响尚未被单变量消融证明。SRDG/DCBR/ACR父级共有这个配置，通道变化不能直接解释三者之间的差异。

上表是D-Fire历史实验。FASDD clean v1目前只有n基线，不能给出“FASDD模块均失败”的结论。现有模块可保留作机制或参考对照，但新方案应覆盖原三层检测特征。

## 4. 当前FASDD对模块设计的要求

- 总体val mAP50–95两seed均值53.3975%，+5点目标暂为58.3975%。fire AP50–95=44.808%，smoke=61.985%。
- fire AP50=74.370%，AP75=45.945%。conf=.25时fire IoU.5/.75召回约67.94%/49.46%，检出与严格定位都有缺口。
- 长边投影到640后，<16px火焰仅占3.68%，16–64px占37.83%。新模块需要覆盖中小目标和框质量。
- 混合场景烟雾在seed1的IoU.5召回57.91%，纯烟89.42%；≥128px组仍为66.57%/91.52%。这是关联证据，不证明共现的因果关系。
- 多火簇的合并/拆分标注和预测边界不一致，部分工业烟羽被列为背景，需明确标注政策。模块无法保证解决任务定义或标签粒度矛盾。
- 同一640模型直接800/960验证导致mAP50–95降低；纯模块对照首先固定640。

完整数字与图例见`FASDD_BASELINE_DIAGNOSIS_20261008.md`。

## 5. 接入与归因需要注意的实际代码限制

1. 当前迁移只复制0..22，所有检测头重置。首轮模块实验沿用相同政策，以现有baseline为对照；“所有兼容权重迁移”单独做n对照，不能与模块变化一起归因。
2. 新Detect候选应保持三层64/128/256输入、分类宽度64、reg_max16和stride8/16/32。读取layer2不应被标准四输入Detect误当作第四个预测层。
3. 安装版`Detect.forward_head(x, box_head, cls_head)`返回`boxes/scores/feats`字典。损失用`feats`生成anchor；即便输入增加P2，返回feats也必须仍是三层对应尺度。
4. 仅将自定义类放进`ultralytics.nn.tasks`全局，不能保证parse_model对多输入/输出通道与Detect专用参数正确处理。优先在baseline构建后安装明确的Detect子类适配器，保存层索引、原head权重与元数据。
5. 首轮只支持当前end2end=false。不能在没有实现相应分支时宣称兼容所有Ultralytics检测头、融合推理和导出路径。
6. 当前分类BCE目标已包含TAL根据分类与重叠构造的软质量权重。额外IoU质量头可能重复已有监督；需要评分与定位错配证据才作为后续独立候选。

## 6. 建议的模块方向

设计**检测前的细节与上下文残差**：P2的压缩细节送入P3回归；P3/P4/P5之间的低维上下文经空间权重融合，分别输出分类/回归残差。保持n主干及原三层预测结构。设计、预算和消融见`docs/FASDD_N_MODULE_DESIGN_V1.md`。

预算算术结果：隐藏宽度16时预计增加26,889参数，总计2,617,119，约+1.04%；隐藏宽度32时增加56,841，总计2,647,071，约+2.19%。这些是待实现结构的算术估计，不是实际网络profile结果。

## 7. 证据归档与复现

- `analysis/scripts/audit_fasdd_n_module_20261008.py`：读取真实检查点与源码哈希，不做前向。
- `analysis/scripts/estimate_fasdd_n_refinement_budget_20261008.py`：计算设计中的卷积权重数及MAC，不构建模型。
- `analysis/artifacts/fasdd_n_module_20261008/static_model_audit_lxy.json`、`static_model_audit_cmj.json`、`two_host_comparison.json`。
- 同目录`source_snapshot.zip`及`proposed_module_budget*.json`。

```bash
python analysis/scripts/estimate_fasdd_n_refinement_budget_20261008.py --hidden 16 --output analysis/artifacts/fasdd_n_module_20261008/budget_reproduced.json
```

本次只完成检查、结构设计、预算估算和文档保存。没有改动核心训练代码、完成权重或数据，也未启动新训练。
