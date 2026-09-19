# 新模块文献检索与取舍记录

检索日期：2026-09-19。面向 P2/SRDG 结果进行的定向检索，不是穷尽性系统综述或优先权证明。
实际阅读范围：论文摘要、公开方法段落、作者仓库说明及部分实现；未复现外部论文的数值。
优先 CVF/ECVA、arXiv 原文、期刊原站和作者官方仓库；第三方检索摘要不作为机制或精度证据。

## 检索线索

- `FreqFusion frequency aware feature fusion dense image prediction`
- `small object detection frequency feature fusion 2025 2026`
- `fire smoke detection frequency wavelet feature fusion YOLO 2025 2026`
- `semantic band residual feature fusion detection`
- `Frequency Selection Adaptive Dilation CVPR 2024`
- `LP-YOLO`, `PWM-net`, `SET Spectral Enhancement Tiny Object Detection`

部分 CVF 页面返回 403 或抓取错误，相关事实使用能访问的 CVF PDF 摘要和作者仓库交叉核对。
没有把所有搜索结果都当作已全文审阅，也没有用检索排名作为新颖性证据。

## 与本课题最相关的七项工作

| 工作 | 来源与已核实机制 | 对本项目的影响 |
|---|---|---|
| DySample，ICCV 2023 | 点采样式可学习上采样；官方实现基于 PyTorch | 提供上采样/对齐方向的外部对照；本轮保持上采样不变以隔离调制因素 |
| FADC，CVPR 2024 | AdaDR、AdaKern、FreqSelect；空间变化的频率重加权 | 最接近的频带选择先行工作，必须明确承认相似性 |
| FreqFusion，TPAMI 2024 | 自适应低通、高通与重采样，处理类内不一致和边界模糊 | “频率感知融合”本身不能作为本项目首次创新 |
| WTConv，ECCV 2024 | 通过小波变换扩展有效感受野 | 小波卷积已有成熟方法；直接替换 WTConv 的创新性有限 |
| SET，CVPR 2025 | 背景平滑 HBS 与训练期对抗扰动 API；论文研究特征频率对小目标的作用 | 不应机械假设高频增强总能帮助小目标 |
| LP-YOLO，CMC 2026 | 火烟场景结合 WTConv、BiFPN、FreqFusion | 已有应用重叠，频率模块+YOLO+火烟的组合不能作为独有贡献 |
| PWM-Net，Discover Computing，2026-09-10 | YOLO11n、PartialNet、WTConv、多谱多尺度注意力 | 与本项目模型家族及场景接近，进一步削弱通用模块堆叠的新颖性 |

上述论文的数据划分、训练预算、模型与设备不同，不能将论文 headline AP/FPS 与本仓库结果
直接相减。尤其 LP-YOLO 描述了 6:2:2 划分和 150 epochs，与本项目协议不同。

## 可核验链接

1. DySample：[论文](https://arxiv.org/abs/2308.15085)、
   [作者实现](https://github.com/tiny-smart/dysample)。
2. FADC：[CVF 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.pdf)、
   [作者实现](https://github.com/ying-fu/FADC)。
3. FreqFusion：[论文](https://arxiv.org/abs/2408.12879)、
   [作者实现](https://github.com/Linwei-Chen/FreqFusion)、
   [融合代码](https://github.com/Linwei-Chen/FreqFusion/blob/main/FreqFusion.py)。
   检索时实现有 MMCV CARAFE 导入及 PyTorch fallback，不能将其概括成“必须自定义 CUDA 才能用”。
4. WTConv：[论文](https://arxiv.org/abs/2407.05848)、
   [作者实现](https://github.com/BGU-CS-VIL/WTConv)。
5. SET：[CVF 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf)、
   [作者实现](https://github.com/HuixinSun/SET)。其训练期 HBS/API 并非本项目当前推理期模块。
6. LP-YOLO：[期刊全文](https://www.techscience.com/cmc/v86n3/65451/html)，
   DOI `10.32604/cmc.2025.072058`。DOI 中的 2025 不应代替刊载年份 2026。
7. PWM-Net：[期刊全文](https://link.springer.com/article/10.1007/s10791-026-10508-z)。

## 决策与未解决问题

DCBR 是独立编写的 PyTorch 候选实现，不复制上述仓库代码，也不声称复现它们。
它以最小结构变化测试“语义单独决定整特征调制是否不足”这一项目假设。

- 比 SRDG 新增的机制变量是联合局部条件与分残差调制；增益固定 .25，消融共用该增益。
  因此 SRDG→DCBR 差异也包含门控容量、组数和增益变化，不能把全部差异归因于频带分解。
  专门用 `whole` 和 `semantic` 内部对照分离这两点。
- 与 FreqSelect 的相似度高，这是论文新颖性最需要防守的近邻。即使 DCBR 有收益，也需要
  同协议 FreqSelect/常规注意力或融合对照以及机制解释，不能靠更名解决。
- 低频直通只是模块的代数结构；不是免疫漏检/误报的保证，也不是火烟物理频谱的先验真理。
- SRDG gate 是否真的抑制了弱目标尚未测量。当前数值只构成设计动机，不能写成已证实因果。
- 更大模型、频率方法、背景损失和跨域增强同时叠加会扩大归因难度，本轮先固定单模块。
- 若内部消融无效，应如实保留结果并调整研究问题，不能通过只报告有利 seed 或挑选 test
  样本来补足创新性。

实现、公式、训练入口和预先固定的实验判读规则见 [MODULE_DCBR.md](MODULE_DCBR.md)。
