# 图像级告警指标的种子方差与 MDD

- seeds: [0, 1, 2]   权重模板: `experiments/yolo11n/D-Fire_p2_seed{seed}/weights/best.pt`
- 1% 负图 FPR 阈值在 **val** 上冻结, 指标在 **test** 上报告
- 参照: `mAP50-95` 三种子 σ = 0.4218pp → MDD = 1.0478pp

## 1. 图像级告警指标（test，阈值在 val 冻结）

| 指标 | 均值 | 种子σ | **MDD (α=.05)** | 相比 mAP50-95 的门槛 |
|---|---:|---:|---:|---:|
| `calibration.fire.after.image_ECE` | 0.08057 | 0.0439pp | **0.1089pp** | **低 9.6 倍** |
| `calibration.fire.after.image_Brier` | 0.04393 | 0.0461pp | **0.1146pp** | **低 9.1 倍** |
| `calibration.fire.before.image_Brier` | 0.04596 | 0.0522pp | **0.1298pp** | **低 8.1 倍** |
| `calibration.fire.before.image_ECE` | 0.08607 | 0.1008pp | **0.2504pp** | **低 4.2 倍** |
| `calibration.smoke.after.image_Brier` | 0.08248 | 0.1664pp | **0.4135pp** | **低 2.5 倍** |
| `calibration.smoke.before.image_Brier` | 0.08263 | 0.1722pp | **0.4279pp** | **低 2.4 倍** |
| `fire.negative_image_fpr` | 0.01025 | 0.2089pp | **0.5190pp** | **低 2.0 倍** |
| `calibration.smoke.before.image_ECE` | 0.14348 | 0.2927pp | **0.7273pp** | **低 1.4 倍** |
| `smoke.negative_image_fpr` | 0.01693 | 0.3742pp | **0.9297pp** | **低 1.1 倍** |
| `calibration.smoke.after.image_ECE` | 0.14260 | 0.4265pp | **1.0595pp** | 相当 |
| `background_alarm_fraction` | 0.00765 | 0.4358pp | **1.0826pp** | 相当 |
| `background_FP_per_image_at_alarm_thresholds` | 0.00798 | 0.4912pp | **1.2203pp** | 高 1.2 倍 |
| `fire.recall` | 0.93104 | 1.2786pp | **3.1764pp** | 高 3.0 倍 |
| `smoke.recall` | 0.72612 | 3.9421pp | **9.7935pp** | 高 9.3 倍 |

## 2. 检测指标（test，官方 evaluate 口径）

| 指标 | 均值 | 种子σ | **MDD (α=.05)** | 相比 mAP50-95 的门槛 |
|---|---:|---:|---:|---:|
| `fitness` | 0.44622 | 0.3693pp | **0.9176pp** | **低 1.1 倍** |
| `metrics/mAP50-95(B)` | 0.44622 | 0.3693pp | **0.9176pp** | **低 1.1 倍** |
| `metrics/mAP50(B)` | 0.77744 | 0.4160pp | **1.0336pp** | 相当 |
| `metrics/recall(B)` | 0.71664 | 0.6793pp | **1.6875pp** | 高 1.6 倍 |
| `metrics/precision(B)` | 0.76931 | 0.8447pp | **2.0986pp** | 高 2.0 倍 |

## 3. 校准诊断（val，in-sample）

> 注意: 阈值是在同一批 val 上选的, 属 in-sample, 数值乐观; 但跨种子程序一致, 方差仍可比。

| 指标 | 均值 | 种子σ | **MDD (α=.05)** | 相比 mAP50-95 的门槛 |
|---|---:|---:|---:|---:|
| `[0].after.image_Brier` | 0.04036 | 0.0648pp | **0.1610pp** | **低 6.5 倍** |
| `[0].before.image_Brier` | 0.04250 | 0.0756pp | **0.1877pp** | **低 5.6 倍** |
| `[1].after.image_Brier` | 0.08353 | 0.1068pp | **0.2653pp** | **低 3.9 倍** |
| `[1].before.image_Brier` | 0.08368 | 0.1103pp | **0.2740pp** | **低 3.8 倍** |
| `[0].after.image_ECE` | 0.07812 | 0.1305pp | **0.3243pp** | **低 3.2 倍** |
| `[0].before.image_ECE` | 0.08434 | 0.1859pp | **0.4617pp** | **低 2.3 倍** |
| `[1].before.image_ECE` | 0.14725 | 0.2997pp | **0.7446pp** | **低 1.4 倍** |
| `[1].after.image_ECE` | 0.14644 | 0.4492pp | **1.1158pp** | 高 1.1 倍 |
| `[0].validation_recall` | 0.95011 | 0.7514pp | **1.8668pp** | 高 1.8 倍 |
| `[1].validation_recall` | 0.73586 | 3.1660pp | **7.8655pp** | 高 7.5 倍 |
| `[1].threshold` | 0.56518 | 3.3558pp | **8.3369pp** | 高 8.0 倍 |
| `[0].threshold` | 0.24576 | 4.3438pp | **10.7916pp** | 高 10.3 倍 |

## 4. 结论怎么读

- 门槛最低的**真实指标**是 `calibration.fire.after.image_ECE`：MDD = **0.1089pp**（mAP50-95 实测 0.9176pp）。
- 换算：换终点可把可分辨效应从 0.92pp 降到 0.11pp，约 **8.4 倍**。
- **注意反直觉的地方**：带「阈值选择」的指标（固定 FPR 下的 recall）门槛**更大**，
  因为阈值本身是种子方差的主要来源。凡是要先选阈值的指标，都继承了这份方差。
- 判据：只有 MDD 明显低于 mAP50-95 的指标才值得当主终点。
- 之后应用低门槛指标**回头重新评估已有 checkpoint**（零训练成本），
  看 SRDG / DCBR / ACR 的效应是否变得可分辨。

> 本报告不含任何新训练；全部来自已有 checkpoint 的重新评估。