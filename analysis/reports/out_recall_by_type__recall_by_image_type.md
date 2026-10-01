# 按图像类别拆召回（域内 vs 跨域，同一套源域冻结阈值）

- 阈值 fire=0.3143 / smoke=0.6057（来自源域 D-Fire val 的 1% FPR）
- 图像类别由 GT 框推出：fire_only / smoke_only / both / none

| 预测配置 | 域 | 图类别 | 图数 | **fire 召回** | **smoke 召回** |
|---|---|---|---:|---:|---:|
| p2_s0 | 域内 D-Fire val | fire_only | 89 | 95.5% | — |
| p2_s0 | 域内 D-Fire val | smoke_only | 471 | — | 62.8% |
| p2_s0 | 域内 D-Fire val | both | 372 | 95.4% | 79.6% |
| p2_s0 | 跨域 FASDD test | fire_only | 2091 | 90.0% | — |
| p2_s0 | 跨域 FASDD test | smoke_only | 3902 | — | 33.9% |
| p2_s0 | 跨域 FASDD test | both | 3358 | 94.8% | 51.8% |
| baseline640_s0 | 域内 D-Fire val | fire_only | 89 | 94.4% | — |
| baseline640_s0 | 域内 D-Fire val | smoke_only | 471 | — | 62.6% |
| baseline640_s0 | 域内 D-Fire val | both | 372 | 93.3% | 77.4% |
| baseline640_s0 | 跨域 FASDD test | fire_only | 2091 | 90.9% | — |
| baseline640_s0 | 跨域 FASDD test | smoke_only | 3902 | — | 32.6% |
| baseline640_s0 | 跨域 FASDD test | both | 3358 | 95.3% | 53.2% |

## 关键对照：smoke_only 图的召回

| 配置 | 域内 | 跨域 | 落差 |
|---|---:|---:|---:|
| p2_s0 | **62.8%** | **33.9%** | -29.0pp |
| baseline640_s0 | **62.6%** | **32.6%** | -30.1pp |

## 判读

- **域内 smoke_only 召回也低** → 依赖火是模型内在特性；解法是数据构成/损失，换数据集无用。
- **域内高、跨域崩** → 是烟的域差；解法是增加烟的多样性与来源。
- 同时看 `both` 一列：若 both 图的 smoke 召回明显高于 smoke_only，
  则「有火时烟好检」这个模式在两个域内都成立，是稳健的结构性发现。

> 不含任何新训练或新推理；全部来自已有 predictions。