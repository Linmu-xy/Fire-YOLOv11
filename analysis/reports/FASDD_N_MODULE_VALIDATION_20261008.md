# YOLO11n模块实现：工程验证记录

日期：2026-10-08，使用北京时间记录用户可见启动时间。目标环境cmj：Linux、RTX3090 24GB、Python3.10.20、torch2.11.0+cu130、torchvision0.26.0+cu130、Ultralytics8.4.42。

## 已完成检查

1. **cmj与lxy分别通过Linux全套59项测试**，0失败、0错误、0跳过。包含49项既有测试和10项新增测试；基线、P2、SRDG/DCBR/ACR相关路径回归正常。
2. 新测试检查参数数目、原head状态不变、RNG不变、起点预测与父级逐元素相同、方形/矩形前向、检测损失、空GT、梯度和非法结构拒绝。
3. 密集特征监督测试验证主要四种模式每个新增参数路径能学习；真实TAL稀疏分配不要求每层回归都有正样本，按有监督路径检查。
4. 四种主要模式设置非零残差权重后，保存→YOLO重载→BN融合的预测一致性通过。
5. 两台分别以严格`torch.use_deterministic_algorithms(True)`完成CUDA检查，主要四模式与baseline均完成640、batch32检测损失与反向，梯度有限、无OOM。640方形输出均为`[1,6,8400]`。
6. 主要四模式ONNX导出及CPU Runtime对齐通过，最大绝对误差约0.00043–0.00051，rtol/atol均1e-3。输入384×640，输出`[1,6,5040]`；新增输出权重故意设非零，覆盖实际模块计算。
7. **context与detail分别完成真实数据1轮pilot**：train16/val16、128分辨率，SGD、EMA、best/last保存、strip与最终验证正常，strip后的检查点重载前向有限。此pilot不用于精度判断。
8. 正式启动前预检89,405张原清单/标签/划分，内容SHA、冻结数据配置SHA和预训练SHA匹配；配方保持n640、100轮、batch32、SGD、AMP=false。

## 参数、计算与显存

| 模型 | 总参数 | GFLOPs@640 | 首次median/p95 ms | batch32训练峰值allocated |
|---|---:|---:|---:|---:|
| baseline | 2,590,230 | 6.4417 | 2.461 / 2.482 | 6.913 GiB |
| detail | 2,594,998 | 6.5418 | 2.519 / 2.570 | 7.210 GiB |
| context | 2,613,375 | 6.5196 | 2.954 / 2.993 | 7.279 GiB |
| full | 2,617,119 | 6.6067 | 3.078 / 3.116 | 7.476 GiB |
| local_control | 2,617,128 | 6.6129 | 2.718 / 2.748 | 7.221 GiB |

GFLOPs来自同版本`get_flops`/THOP工具估算，并非完整统计所有重排、融合、resize、激活和访存。显存是工程脚本的synthetic loss/backward allocated峰值，没有完整数据增强、SGD状态和全部trainer/EMA开销；正式训练初期日志context约8.35G，不能把表格当成最终训练显存上限。

延迟协议：同机、fused FP32、batch1、640，20次warmup、100次顺序前向，CUDA同步，包含Python模型调用，排除图像预处理/NMS。cmj第二次同协议中位数baseline/detail/context/full/local分别2.454/2.625/2.850/2.949/2.584ms。cmj两次测量的相对增幅：detail2.4–7.0%、context16.2–20.1%、full20.2–25.1%。lxy独立测量分别2.443/2.547/2.877/3.002/2.629ms，增幅落在上述范围内。范围不是统计置信区间。

因此，组合模块参数增长1.04%、卷积估算开销较小，但延迟增长明显超过优先≤10%的设计工作目标。小参数增幅不能代替实测效率结论，context/full作为研究候选需在准确度结果后进一步判断是否值得保留。

## 验证过程中处理的问题

- 首次CUDA训练检查从已strip的完成权重克隆，所有原参数处于冻结状态，导致baseline loss没有梯度。已在工程检查中恢复trainability并保持DFL固定基冻结；正式trainer从预训练构建新模型，无此问题。失败的v1产物保留，成功结果在v2。
- 初版测试要求稀疏TAL对每层回归输出都有非零梯度，微小合成目标不满足。拆分为实际检测损失的有效监督检查，以及全部路径的密集特征监督检查，后者要求每个参数获得梯度。
- Windows的既有prepare符号链接测试因WinError1314失败。10项新增CPU测试通过；目标Linux完整59项通过。没有为此修改数据准备实现或申请管理员权限。
- lxy在首次代码同步后再次SSH超时，恢复连接后补齐验证脚本修复、59项测试和CUDA检查。主要7份实现文件的SHA在两台核对一致。

## 保存位置

- 实现/使用说明：`docs/FASDD_N_REFINEMENT_IMPLEMENTATION_V1.md`。
- CUDA、ONNX、context pilot：`analysis/artifacts/fasdd_n_module_20261008/validation_cmj_v2/`。
- detail pilot及第二次测量：`analysis/artifacts/fasdd_n_module_20261008/validation_cmj_detail/`。
- lxy CUDA及独立延迟测量：`analysis/artifacts/fasdd_n_module_20261008/validation_lxy/`。
- 单元测试日志：仓库外`Fire-YOLOv11-runtime/n-refinement-validation-20261008/unit_tests.log`及JSON。
- 本地汇总副本：`analysis/artifacts/fasdd_n_module_20261008/validation_cmj.json`、`validation_cmj_detail.json`、`unit_tests_cmj.json`。
- lxy汇总副本：`validation_lxy.json`、`unit_tests_lxy.json`，位于同一汇总目录。
- 核心models.py修改前备份：仓库外`Fire-YOLOv11-runtime/code-update-20261008/`。

cmj正式context seed0于**2026-10-08 20:35**启动：

```text
results/fasdd_cv_clean_v1/fasdd-n-refinement-v1/context/seed0/
```

cmj监督PID38647，训练PID38698；脚本、配置、源码SHA、完整来源快照保存在该结果的`launch/`。实时日志在仓库外对应runtime目录。20:51检查时已完成3轮训练和验证，CSV与best/last权重均正常保存。早期AP不能作为模块精度结论。

lxy detail seed0于北京时间**2026-10-08 20:51**启动，监督PID17427、训练PID17437；输出`results/fasdd_cv_clean_v1/fasdd-n-refinement-v1/detail/seed0/`。20:52第1轮batch32运行正常，损失有限，日志显存约8.48G。两台均保持n640、100轮、batch32的固定筛选配方。

工程验证通过只说明代码和训练流程可运行；模块精度收益必须由完整固定验证协议的训练结果判断。
