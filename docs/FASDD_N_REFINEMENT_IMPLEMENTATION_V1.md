# YOLO11n细节与上下文模块：实现与使用

日期：2026-10-08。对应`FASDD_N_MODULE_DESIGN_V1.md`。工程验证记录见`analysis/reports/FASDD_N_MODULE_VALIDATION_20261008.md`。

## 实现文件

- `firesmoke/pyramid_refinement.py`：细节回归、三尺度上下文、普通卷积对照及Detect适配器。
- `firesmoke/models.py`：在已构建的原n模型上安装候选；现有方法路径保留。
- `configs/protocol-fasdd-n-refinement-v1.yaml`：固定clean-v1数据、n640、100轮的筛选配置。
- `scripts/train_fasdd_n_refinement.py`：预检、后台启动、源码/配置快照与运行归档。
- `tests/test_pyramid_refinement.py`、`tests/test_n_refinement_launcher.py`：模块与入口测试。
- `analysis/scripts/validate_fasdd_n_refinement_20261008.py`：CUDA、开销、ONNX与真实数据短训练验证。

原三层检测头保持输入通道64/128/256、stride8/16/32、分类宽度64、reg_max16，共8,400个640方形输入预测位置。细节模式额外缓存layer2，作为辅助特征，不新增检测层。分类/回归残差输出零初始化，新模块构建隔离RNG，原head参数保持一致。

## 方法与验证覆盖

| method | 含义 | 状态 |
|---|---|---|
| baseline | 原n，无模块 | 回归测试、CUDA基准 |
| detail | 仅浅层细节进入P3回归，r16 | CPU/CUDA/ONNX/真实训练pilot通过 |
| context | 仅三尺度上下文，r16 | CPU/CUDA/ONNX/真实训练pilot通过 |
| full | 两部分组合，r16 | CPU/CUDA/ONNX通过 |
| local_control | 普通局部瓶颈，隐藏宽度27/8/13 | CPU/CUDA/ONNX通过 |
| full_uniform | 上下文融合改固定均匀权重 | CPU参数/起点等价检查通过，尚无独立训练结果 |
| full_shared | 上下文的分类/回归共享输出投影 | CPU参数/起点等价检查通过，尚无独立训练结果 |
| full_r32 | 组合模块，r32 | CPU参数检查通过，尚无独立训练结果 |

普通卷积对照新增26,898参数，full新增26,889；新增卷积MAC相差约3.77%，在5%预算匹配范围内。普通卷积宽度分配不同，不能称表达能力完全相同。

## 启动方式

在目标Linux仓库根目录使用已有yolo11环境：

```bash
python scripts/train_fasdd_n_refinement.py --method detail --seed 0 --plan
python scripts/train_fasdd_n_refinement.py --method detail --seed 0 --dry-run
python scripts/train_fasdd_n_refinement.py --method detail --seed 0 --execute
```

默认只做预检；`--plan`只打印计划，不读数据、不训练。`--execute`后台启动监督进程，打印PID、日志、输出位置。

其余方法使用同一脚本改变`--method`。首轮已启动lxy=detail seed0、cmj=context seed0。已存在同名结果会拒绝覆盖；新的独立运行显式指定不同`--tag`。不要通过直接`YOLO(best.pt).train()`复建候选：标准trainer可能根据原baseline YAML重建普通head。训练候选使用本注册入口。

数据原配置`configs/protocol-fasdd-cv-clean-v1.yaml`的文件SHA仍与version.json严格绑定；方法配置另外登记协议/训练配方指纹。入口验证原数据政策、内容指纹、89,405张清单/标签/划分和预训练SHA，保持原100轮配方。没有修改prepared目录或原数据标签。

## 保存规范

结果：`results/fasdd_cv_clean_v1/fasdd-n-refinement-v1/<method>/seedN/`。

每次保存`run.json`、`protocol.yaml`、`weight_transfer.json`、`refinement.json`、CSV和best/last权重。`launch/`归档启动脚本、注册协议、源码tar快照、来源SHA及进程信息。实时日志与监督状态在仓库外`<仓库名>-runtime/`。

原初始化政策仍是共享层0..22迁移，所有原检测头重新初始化。兼容检测头预训练、分辨率和轮数变化要另设对照，不能混入本轮模块收益。

## 复现验证

```bash
python -m unittest discover -s tests -v
python analysis/scripts/validate_fasdd_n_refinement_20261008.py --checkpoint results/fasdd_cv_clean_v1/fasdd-cv-clean-v1/baseline/seed1/weights/best.pt --output analysis/artifacts/fasdd_n_module_20261008/new_validation_run --device 0 --export --pilot context
```

输出目录必须新建。验证脚本使用独立的工程fixture，从train/val各取16张原图，分别包含背景、纯火、纯烟、混合场景；标签复制、图像引用和缓存均在analysis/artifacts。pilot仅128分辨率、1轮，验证trainer/EMA/优化器/保存/strip/最终validator，不能用其AP说明精度收益。

ONNX验证限定本轮已测的静态FP32、batch1、384×640、opset17、无NMS；故意将新增输出权重设非零后与CPU ONNX Runtime比较。动态形状、TensorRT、端侧FP16和end2end头未验证。本适配器显式拒绝end2end、legacy及非原n三层输入结构。

## 实测开销与当前运行

cmj RTX3090，融合FP32、batch1、640、20次warmup/100次前向：

| 模型 | 参数 | THOP估算GFLOPs | 首次延迟中位数 |
|---|---:|---:|---:|
| baseline | 2,590,230 | 6.442 | 2.461 ms |
| detail | 2,594,998 | 6.542 | 2.519 ms |
| context | 2,613,375 | 6.520 | 2.954 ms |
| full | 2,617,119 | 6.607 | 3.078 ms |
| local_control | 2,617,128 | 6.613 | 2.718 ms |

重复测量中detail延迟增幅约2.4–7.0%，context约16.2–20.1%，full约20.2–25.1%。上下文候选未达到原设计优先≤10%延迟增幅的工作目标，应作为准确度研究候选；部署结论需在精度收益明确后进一步优化/测量。

两台机器均通过全套59项测试和主要四模式的CUDA batch32损失/反向检查；ONNX对齐和真实数据pilot在cmj完成。核心模块、配置、启动脚本、测试及验证脚本的字节SHA一致。

cmj context seed0于北京时间**2026-10-08 20:35**启动100轮正式训练，20:51已完成3轮训练和验证。lxy恢复连接并完成验证后，detail seed0于**20:51**启动，第1轮batch32训练正常。分别保存到`results/fasdd_cv_clean_v1/fasdd-n-refinement-v1/context/seed0/`和`results/fasdd_cv_clean_v1/fasdd-n-refinement-v1/detail/seed0/`。

以上为工程验证和开销，尚无模块的完整FASDD精度结果，不承诺增加5点。
