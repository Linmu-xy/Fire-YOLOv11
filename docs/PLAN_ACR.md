# ACR 候选与论文实验方案

设计日期：2026-09-20。基于仓库 04c872aa874ee8735c1d13462b49dfc4cefb574c。
这是待验证的研究候选，未完成真实数据训练，不承诺精度提升或 SCI 二区录用。
JCR Q2 和中科院二区是不同口径，最终投稿前按目标年份、学科和单位要求核实。

## 研究判断

DCBR 相对同 seed P2 的最佳验证 AP50-95 差值为 seed0 -0.00224、seed2 +0.00468，
平均 +0.00122，即 +0.122 个百分点。两个种子不足以证明稳定增益。
历史 baseline 与 P2 的训练协议不同，不能据均值差证明 P2 贡献；先汇入另一工作站的
同协议 baseline，并核对 run.json、标签清单、初始化和预训练指纹。

工程目标设为：冻结测试集 AP50-95 相对同协议 P2 提升 2 个百分点，同时每类低误报
召回不降低。这是项目期望，绝不是预测结果，也不是二区发表门槛。
100 epoch 筛查的继续条件预先定为三 seed 配对平均至少 +0.005、至少两个同向；
若低于此值，只在已有预定的小目标/低误报终点有稳定收益时继续。保留全部负结果。

## 新模块：Aligned Context Residual (ACR)

研究假设：P2 的浅层细节与上采样 P3 语义之间存在局部错位；单一局部感受野无法同时
适应紧凑目标和弥散目标。它们是待检验的原因，现有整体 AP 未证明原因成立。
在现有 P2 Concat 后、DWConv 前插入，输入为 S、D，各 64 通道、stride=4。

1. C = SiLU(Conv1([S,D]))，压到 16 通道。
2. softmax(Conv1(C)) 产生九个空间权重，对 S 的 3×3 整数平移候选做凸组合得到 A。
   使用 replicate 边界；中心候选可还原原位置。只能修正一格以内的局部取样，
   不是光流、连续可变形卷积或目标框级对齐。
3. D 压到 16 通道，经 dilation=1/3/5 的三个深度卷积专家，感受野为 3/7/11 格。
   另一 softmax 产生每个空间位置的三路权重，得到上下文 E。
4. 输出 [S + Ws(A-S), D + Wd(E)]。Ws/Wd 零初始化，起点等价于父级 P2。

新层构建隔离全局 RNG；既有 backbone、neck、P2 后处理和 Detect 的对应初始化一致。
首步输出投影可获得梯度，内部路由在输出投影打开后获得梯度。不能声称全程等价。
不使用 grid_sample；仍必须实测目标 CUDA 环境的确定性、显存及导出支持。
九候选采用逐路累加，避免显式堆叠 B×C×9×H×W；autograd 仍有额外激活成本。
所有专家每次都执行，这是软特征融合，不是稀疏执行，不宣称节约计算。
不将任何专家命名为“火焰专家”或“烟雾专家”，因为没有类别监督约束。
新增输出投影为自由残差，不继承 DCBR 的 25% 修正界。

## 已实现与待做的边界

已实现：模块、四种模型 YAML、Ultralytics 注册、继承 v2 的配对训练入口、测试。
尚待实验：真实训练、机制可视化/分层统计、外部方法复现、AP-small、跨域结果、部署。
蒸馏、分割辅助任务、难例重采样和高分辨率训练均为后续实验选项，本版本没有实现。

| 方法 | 对齐 | 上下文 | 要检验的因素 |
|---|---|---|---|
| p2 | 无 | 原父级 | 主对照 |
| p2_acr | 九候选 | 空间自适应三路 | 完整候选 |
| p2_acr_no_align | 关闭 | 空间自适应三路 | 对齐贡献 |
| p2_acr_uniform | 九候选 | 固定平均三路 | 自适应路由贡献 |
| p2_acr_local | 九候选 | 仅 dilation=1 | 多尺度上下文贡献 |

消融保留参数形状，但关闭分支有不参与梯度的参数；不是有效容量完全相等的实验。
同种子比较并不消除训练噪声；论文需三种子以上，接近噪声时扩到五种子。

## 如何训练

把本次新增/修改文件同步至有数据和 CUDA 的仓库。保留 scripts/train_dcbr.py：
新入口复用它的父级/数据/权重/训练参数检查。使用已有 yolo11 环境，版本要求
ultralytics==8.4.42、torch==2.11.0、torchvision==0.26.0。
不要在 Windows 直接安装仓库的整份 Linux/CUDA requirements.txt。

```bash
conda activate yolo11
python scripts/train_acr.py --plan
python -m unittest discover -s tests -p test_acr.py -v
python scripts/train_acr.py --seed 0 --dry-run
python scripts/train_acr.py --seed 0 --execute
```

入口从 configs/protocol-v2.yaml 加入新方法。固定 100 epoch、640、batch32、SGD、
AMP=false，background_alpha=0、style=false；共享 0..22 层预训练，检测头从头初始化。
默认只预检；--plan 只打印计划，不检查数据/GPU。--execute 预检成功后才训练。
预检要求同 seed 的 experiments/yolo11n/D-Fire_p2_seedN/run.json；缺失或不匹配会报错。
新运行写入 outputs/module-acr-v1/dfire/p2_acr/seed0，含 best.pt、results.csv、run.json、
完整 protocol.yaml；不自动复制大体积归档。已存在的输出不会覆盖，重试使用新 --tag。

```bash
python scripts/train_acr.py --seed 1 --execute
python scripts/train_acr.py --seed 2 --execute
python scripts/train_acr.py --method p2_acr_no_align --seed 0 --execute
python scripts/train_acr.py --method p2_acr_uniform --seed 0 --execute
python scripts/train_acr.py --method p2_acr_local --seed 0 --execute
```

先看 seed0 能否稳定收敛，再补 seed1/2；完整消融按同样方法补齐 1/2。不能仅保留有利 seed。
若 GPU 显存不足，记录失败；统一降低所有配对实验 batch 并重跑父级，不能只改候选一侧。
现有预检的字体路径面向 Linux ~/.config/Ultralytics/Arial.ttf，训练端沿用已有字体准备方式。

## 评价与论文路线

阶段 A：同协议 baseline/P2/ACR 三 seed；报告最佳验证 AP50-95 同一行 P/R/AP50，
以及末轮值、每类 AP、学习曲线和时间。验证集用于筛选，测试集不参与模块调整。
阶段 B：三项内部消融；至少一个同预算普通卷积残差对照，及 DySample/FreqFusion
等近邻方法的公平对照。普通卷积对照尚待实现，不能用只加参数解释新机制。
阶段 C：冻结方案，在 D-Fire test 和 FASDD_CV zero-shot 评价，再做反向跨域。
若只完成一个方向就报告一个方向，不能称双向实验。使用训练/验证集检查近重复，
固定划分规则，训练前冻结测试清单；不能依结果挑选测试子集。
阶段 D：用源域 val 校准每类 1% 经验负例 FPR 阈值，再固定阈值测 test 的每类图像
召回、背景报警比例和 FP/image。两类各 1% 不等于联合报警率 1%。
AP-small 按原图 COCO 面积分层，额外报告 letterbox 后尺寸；必须正确处理不属于当前
面积段的 GT 忽略规则，不能删掉大框再把对应预测当 FP。
阶段 E：同硬件、batch1、同精度测 median/p95 latency、峰值显存、导出结果。
训练时间不能替代推理性能。按场景/视频分组 bootstrap，避免相邻帧伪独立。

如果 ACR 只带来零点几个百分点且没有稳定的低误报/跨域收益，应停止继续堆模块。
提高绝对指标的第二阶段可比较：640→960 输入、100→200 epoch、YOLO11n→s、
强教师蒸馏；每种设置都给 baseline/P2 相同预算，单独报告训练配方收益。
960 的像素量是 640 的 2.25 倍，需要实际显存测量，不能沿用当前 100 epoch 父级作对照。
新协议应冻结在独立文件/标签，并训练新父级；当前配对入口会拒绝不匹配的历史父级。

## 相关工作与新颖性

- DySample, ICCV 2023: https://arxiv.org/abs/2308.15085
- FreqFusion, TPAMI 2024: https://arxiv.org/abs/2408.12879
- SET, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html

这些工作分别已有可学习重采样、融合错位/频率处理、细目标特征分析，ACR 的对齐和
多尺度路由都不能单独宣称首次提出。此次是独立实现的机制候选，未复制上述源码。
论文价值取决于是否证明火烟弱目标的具体失败模式、机制为何有效、跨域/误报收益是否
稳定、以及是否优于最接近方法。2 个百分点目标不是任何期刊录用规则。

## 验证记录

实际执行情况见 ACR_VALIDATION.md。合成测试只能验证代码路径，不能证明数据集精度。
