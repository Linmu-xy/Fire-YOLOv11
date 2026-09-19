# 开发验证记录

日期：2026-09-17。未启动真实数据训练，未执行 optimizer.step，未评估历史权重的真实测试集性能。

## 环境和检查

- Python 3.10.20；Ultralytics 8.4.42；torch 2.11.0+cu130；torchvision 0.26.0。
- CPU：Intel Core i7-12700KF；验证过程使用 CPU。
- `python -m unittest discover -s tests -v`：20 项通过。
- 包含 canonical 标签与上游 loader 集成、随机权重 checkpoint 读写、3 张合成图像的完整
  AP/逐图预测评估、loss 有限性和梯度、所有消融配置预览、校准泄漏拒绝、种子重复拒绝。
- 梯度检查只在小随机张量上执行 backward，无优化器/参数更新；不能替代真实训练稳定性验证。
- `train` 无 `--execute` 时仅预览，测试确保不初始化训练 runtime。
- README/.gitignore 及新文件经过 whitespace/diff 检查。

## 模型结构测量

本机固定版本、nc=2、未融合模型；GFLOPs 用上游 profiler 按 640 输入计算。
随机前向实际使用 128 输入，检查输出形状、stride 和有限值。

| 结构 | 参数量 | GFLOPs@640 | stride | 输出形状@128 |
|---|---:|---:|---|---|
| baseline | 2,590,230 | 6.4416768 | 8/16/32 | 1×6×336 |
| P2 | 2,638,312 | 10.1378304 | 4/8/16/32 | 1×6×1360 |

P2 参数增加约 1.9%，计算量增加约 57.4%。不能据此宣称推理延迟增加小于 10%，
更不能仅因为参数少就宣称部署实时。若目标设备预算不满足，应把 P2 作为消融对照或重新设计。
这些是结构检查结果，不是精度、泛化或速度提升结果。

## 实际源数据标注审查

已读取两套数据的 split 清单和全部标签，未运行完整图像内容/近重复哈希扫描。

| 数据集/split | 原始图像数 | 规则排除图像数 | 准备后预期图像数 |
|---|---:|---:|---:|
| D-Fire train | 15,499 | 14 | 15,485 |
| D-Fire val | 1,722 | 0 | 1,722 |
| D-Fire test | 4,306 | 12 | 4,294 |
| FASDD_CV train | 47,660 | 2 | 47,658 |
| FASDD_CV val | 31,770 | 1 | 31,769 |
| FASDD_CV test | 15,884 | 0 | 15,884 |

排除依据：至少一个零面积框，或 xywh 分量不在 [0,1]。
D-Fire test 包含 4 个零面积框和 8 个归一化范围异常框；其他排除均因零面积框。
示例 `D-Fire/train/AoF05470.jpg` 对应一个 0×0 框；`D-Fire/test/WEB10769.jpg` 对应宽度 >1。
原始文件未修改，真实数据规范化 prepare 留给使用者按冻结协议执行。

另外，xywh 合法但边缘跨图像的框数量为 D-Fire train/val/test：292/25/54；
FASDD_CV train/val/test：84/46/8。本版本保留这些框并在审查中记录。
标注有效性子集不等于近重复 clean test，也不能直接与官方全量测试集成绩作公平对比。

开发时本地产物（不入 Git）：

- `artifacts/audit/metadata-v1/audit.json`
- `artifacts/audit/metadata-v1/label_issues.json`
- `artifacts/checks/baseline-forward.json`
- `artifacts/checks/p2-forward.json`

## 未验证内容

100 epochs 收敛、GPU 实际训练、全部图片解码、内容去重、真实 test 指标、真实跨域收益、
源域阈值在目标域的误报预算、真实边缘设备速度、量化、视频级预警、DDP、续训均未验证。
框架会保留运行元数据，正式实验必须由研究者启动并依据实际结果更新此记录。

## 2026-09-18：SRDG 候选模块

- `python -m unittest discover -s tests -v`：23 项通过；未执行 optimizer step。
- SRDG 在零初始化时与输入逐元素相等，投影层与输入梯度均有限且非零。
- 隔离模块初始化 RNG 后，同 seed 的 P2 与 P2+SRDG 下游 P2/Detect 参数逐张量一致。
- 随机权重 checkpoint 已完成保存、重新加载和有限值前向检查。
- `check-model`：2,642,472 参数，10.3475456 GFLOPs@640，stride 4/8/16/32，
  128 输入输出形状为 1×6×1360，所有值有限。
- 相对原 P2，SRDG 增加 4,160 参数和 0.2097152 GFLOPs；这些仅是结构测量，不是精度或
  误报改善证据。
- `scripts/train_srdg_seed0.py --dry-run` 完成数据、父级 P2 指纹和环境检查；当前自动化工具环境
  不暴露 CUDA，故 CUDA 项失败且训练被阻断。研究者终端需重新 dry-run，未由本次开发启动训练。

## 2026-09-19：DCBR 与两个内部消融

- 使用 `/home/cmj/miniconda3/envs/yolo11/bin/python`，torch 2.11.0+cu130，ultralytics 8.4.42。
- `python -m unittest discover -s tests -v`：34 项通过（约 2.5 秒测试主体），未构造训练循环，
  未执行 optimizer step。临时合成图片测试不属于真实数据集训练/测试。
- 三个新图都通过正目标/空目标损失及梯度检查；CPU 640×640 前向输出 `[1,6,34000]`，值有限。
- 同 seed 的父级 P2 及 DCBR 的对应参数和初始推理预测完全一致；输出投影能接收非零梯度。
- 常量边界、残差分解、逐点修正幅度界、内部消融、state_dict 与完整 checkpoint 加载、
  YOLO fuse、默认不训练和预检失败阻断均通过。
- P2/DCBR 都迁移 378 个张量、2,171,791 个元素；DCBR 参数 2,641,832，新增 3,520。
- Ultralytics `get_flops` 估计 DCBR 10.3221504 GFLOPs@640、P2 10.1378304。
  THOP 未完整覆盖所有逐点/非线性操作，不能作为完整运算量或实测延迟。
- `python scripts/train_dcbr.py --seed 0 --dry-run`：21501 张图的数据清单和标签检查通过，
  配对父级超参/数据/预训练/P2 图指纹通过；当前工具环境 CUDA 检查失败，exit=1 且未训练。
- 未测 GPU 显存、实际收敛或新模块 AP；新模块在 P2 高分辨率上保留多个中间张量，
  参数量增幅小不代表显存增幅小。若目标 GPU 显存不足，应先记录失败和内存数据，
  不悄悄改变候选一侧的 batch/AMP 来维持运行。
- 文献与设计边界见 `RESEARCH_DCBR.md`、`MODULE_DCBR.md`；未新增实验性能声明。
