# SCI 研究实验协议 v1

状态：代码候选方案，尚未训练。以下均为预先规划，不是实验结论。

## 问题与假设

研究问题：在 nano 检测器预算下，能否在未见过的数据来源上降低背景误报，同时保留烟雾召回？

- H1：P2 有助于远距离/小尺寸目标，但其收益需与计算量增加比较。
- H2：训练期 top-K 空图惩罚降低误报，其是否损伤烟雾召回必须检验。
- H3：源域光照扰动可能减小部分外观变化下的性能下降；不预设所有域有效。
- H4：源域校准在同域测试改善图像级概率质量；跨域是否保持有效是开放问题。

研究目标不能表述成“必然提高两点 mAP”或“达到某个期刊分区”。记录失败实验，
只在验证集开发方法。测试阶段冻结结构、增强、loss 参数、阈值选择规则和 checkpoint 选择方式。

## 数据协议

本版本只实现分别训练和双向 zero-shot 跨数据集评估。D-Fire 和 FASDD_CV 代表不同数据来源，
不能仅凭数据集名称推断它们是严格独立的场景域。

```text
D-Fire train -> D-Fire val 选 checkpoint/阈值 -> D-Fire test + FASDD_CV test
FASDD_CV train -> FASDD_CV val 选 checkpoint/阈值 -> FASDD_CV test + D-Fire test
```

不把训练与目标域无标签数据混用；如果未来加入 UDA，必须独立命名协议，单独说明使用目标
train 图像且不使用目标标签/测试图，不能与当前 source-only 结果混称 zero-shot。
混合训练、1:1 域平衡采样、第三外部数据集和 scene holdout 尚未实现，需要单独增加并验证。

原始 split 保留来源，标签异常整图排除定义一个 **annotation-valid subset**，不是官方完整测试集，
也不是已经去重的 clean subset。文章需同时列出原始数量、每 split 排除数量、原因与清单哈希。
禁止直接把本版本有效子集的数字与文献官方全量 split 数字作公平优劣比较。
如果补充官方协议复现实验，应单独标识，并说明上游框架实际过滤了哪些图像和框。

近重复审查：SHA256 表示字节完全一致；64-bit dHash 的汉明距离仅提供候选，不证明同场景或
相同内容。低纹理图像可能发生碰撞。当前审查输出跨 split/跨数据集候选，不自动删除。
人工确认来源/视频/场景后，应按组固定新划分，保留官方参考 split；记录敏感性分析。
当前不根据文件名猜测 scene ID，也不把无同名路径当作不存在数据泄漏。

## 训练与实验矩阵

主对照使用同一源码版本、数据版本、输入 640、100 epochs、batch 32、SGD、关闭 AMP、
同一初始化政策和相同基础增强。seed 为 0、1、2。
确定性选项不保证不同 CUDA/硬件上逐位一致，所以需要记录环境和重复实验。

2 个训练数据集 × 5 方法 × 3 seeds = 30 次正式训练。每个 best checkpoint 评估两个测试数据集。
完整同域和跨域检测主表为 60 次 test evaluation；另有各源域 val 的校准推理。
不将两个 seed0 目录当作两个独立重复。校准不新增训练模型。

best.pt 由固定版本上游验证 fitness 选择（该版本检测 fitness 为 mAP50-95）；
记录实际选择方式，不分别挑出最高 precision、最高 recall、最高 AP 所在轮次拼成一个模型结果。
固定 100 epochs 不代表已证明充分收敛；先用源域验证曲线审查，再统一登记新的训练预算。

若论文要主张独立“光照增强”贡献，还需追加 baseline+style、background+style 等对照，
目前五行消融只回答它在 p2_background 之上的条件增益，不能估计全部交互效应。
top-K/alpha 灵敏度建议 K∈{16,32,64}、alpha∈{0.1,0.25,0.5}，只用源域 val，控制搜索预算。
额外增强需与同预算普通颜色增强对比，防止把训练增强强度差异包装为方法收益。

## 评价口径

已实现：

| 类型 | 指标 | 口径 |
|---|---|---|
| 检测 | mAP50、mAP50-95、P、R、每类 AP/P/R | Ultralytics 验证器，标准匹配；P/R 是其工作点 |
| 图像报警 | 每类 recall、negative-image FPR | 源域验证集选阈值后冻结 |
| 背景报警 | background_alarm_fraction | 无 fire/smoke GT 的图像中至少一次报警的比例 |
| 背景框 | FP/image | 在报警对应阈值下的背景预测框数/背景图数 |
| 概率质量 | 图像级 ECE（15 bins）、Brier | 类别是否存在，不评价定位校准 |
| 稳健性 | seed 均值±样本标准差 | ddof=1；权重/seed 重复检查 |
| 采样不确定性 | bootstrap 95% 区间 | 默认图像级；完整 scene IDs 时按组重采样 |
| 复杂度 | params、GFLOPs | 显式记录 imgsz、上游 profiler 口径 |
| 性能 | forward 延迟 mean/std/median/p95、峰值 GPU allocated memory | batch1，FP32，fused；不含 NMS/IO |

图像报警 recall 与定位 recall 不能相互替代。空图上的 FP/image 不等于全部负类 anchor 的错误率。
每类负图含有“另一类别存在”的图像；纯背景图是更小子集。合并两类报警不保证仍满足单类 FPR 预算。

尚未实现，不能在论文中冒充已有结果：COCO 严格 APsmall/medium/large、检测 AP 的 paired scene
bootstrap、FROC 曲线、固定定位 recall 的误报指标、视频每小时误报、time-to-detection、
INT8/FP16/ONNX/TRT 导出评价、真实设备端到端延迟、功耗、校准风险形式化保证。
小目标假设若保留为最终贡献，必须先补齐统一坐标/面积口径的尺寸分层指标；
不能用训练缩放后的相对面积随意命名 COCO APsmall。

## 论文证据与工作顺序

1. 审查标注与内容近重复，固定数据版本；优先获取可靠 scene/video IDs。
2. 手动训练 baseline 的三个种子，分析背景、烟雾和跨域错误，而不是只看总体 mAP。
3. 按假设完成消融；同时测计算量与目标硬件延迟，决定是否保留 P2。
4. 模型选择完全冻结后，运行所有测试与双向迁移矩阵，报告负结果和置信区间。
5. 如要宣称早期预警，补充按事件/摄像头隔离的视频实验和非火背景小时数。
6. 发布代码版本、配置、split/排除清单哈希、环境快照和逐图预测；按数据许可处理原图。

论文图表建议：数据/排除/场景统计，方法图，同域主表，2×2 跨域矩阵，消融，图像校准曲线，
误报类别案例，精度-计算量/延迟图。注意校准只能改变分数含义，不能被描述为提升原始排序能力。

当前方法需要与普通 hard-negative/focal 类损失、普通光照增强以及同计算预算的检测器对照。
YOLO26n 可作为后续外部检测器对比，但尚未接入本损失适配器；不能把 YOLO11 特有的 loss 直接
套到端到端模型并假设行为相同。nano 后缀也不保证不同系列计算预算相同。

## 文献定位起点

以下作为阅读线索，不构成完成 novelty review 或证实当前方法优于已有方法：

- [Benchmarking Multi-Scene Fire and Smoke Detection, PRCV 2024](https://arxiv.org/abs/2410.16631)：场景与负样本基准。
- [MS-FSDB 作者仓库](https://github.com/XiaoyiHan6/MS-FSDB)：检查数据来源是否与现有数据重叠。
- [Cross-Dataset Evaluation of YOLOv8, Drones 2026](https://doi.org/10.3390/drones10080635)：跨域与污染审查。
- [Wildfire and smoke early detection for drone applications, 2024](https://doi.org/10.1016/j.engappai.2024.108977)：轻量检测/分割与部署。

SCI/JCR 与中科院分区不是同一制度，应按单位认定口径和投稿当年版本核实。
