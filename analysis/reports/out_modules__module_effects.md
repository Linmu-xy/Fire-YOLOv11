# 三模块在低门槛终点下的配对效应

- 数据源：`analysis/out_modules` + P2 复用自 `analysis/out_reliability`
- 共享 FPR 阈值取自 p2/seed0 的 val：fire=0.3148, smoke=0.6060
- 效应 = 该模块 − 父级 P2（同 seed 配对）；负值表示比 P2 差

| 模块 | 终点 | seeds | 效应 | 配对σ | MDD | 参照门槛 | 判定 |
|---|---|---|---:|---:|---:|---:|---|
| p2_srdg | `p2_map50_95` | 1 | -0.142pp | — | — | 0.918pp | — |
| p2_srdg | `fixed_fire_recall` | 1 | +0.723pp | — | — | 0.343pp | — |
| p2_srdg | `fixed_smoke_recall` | 1 | +0.193pp | — | — | — | — |
| p2_srdg | `fixed_bg_alarm` | 1 | +0.000pp | — | — | 0.568pp | — |
| p2_srdg | `fixed_fire_fpr` | 1 | -0.094pp | — | — | — | — |
| p2_srdg | `ece_fire_after` | 1 | -0.157pp | — | — | 0.109pp | — |
| p2_srdg | `brier_fire_after` | 1 | -0.133pp | — | — | 0.115pp | — |
| p2_srdg | `ece_smoke_after` | 1 | -0.662pp | — | — | — | — |
| p2_srdg | `brier_smoke_after` | 1 | -0.018pp | — | — | — | — |
| p2_dcbr | `p2_map50_95` | 2 | -0.081pp | 0.387pp | 3.476pp | 0.918pp | 不可分辨 |
| p2_dcbr | `fixed_fire_recall` | 2 | +0.723pp | 0.383pp | 3.443pp | 0.343pp | 不可分辨 |
| p2_dcbr | `fixed_smoke_recall` | 2 | -0.097pp | 1.299pp | 11.668pp | — | 不可分辨 |
| p2_dcbr | `fixed_bg_alarm` | 2 | +0.100pp | 0.141pp | 1.267pp | 0.568pp | 不可分辨 |
| p2_dcbr | `fixed_fire_fpr` | 2 | -0.031pp | 0.177pp | 1.595pp | — | 不可分辨 |
| p2_dcbr | `ece_fire_after` | 2 | -0.094pp | 0.026pp | 0.235pp | 0.109pp | 不可分辨 |
| p2_dcbr | `brier_fire_after` | 2 | -0.062pp | 0.130pp | 1.166pp | 0.115pp | 不可分辨 |
| p2_dcbr | `ece_smoke_after` | 2 | -0.359pp | 0.113pp | 1.016pp | — | 不可分辨 |
| p2_dcbr | `brier_smoke_after` | 2 | +0.023pp | 0.087pp | 0.781pp | — | 不可分辨 |
| p2_acr | `p2_map50_95` | 1 | -0.463pp | — | — | 0.918pp | — |
| p2_acr | `fixed_fire_recall` | 1 | +0.361pp | — | — | 0.343pp | — |
| p2_acr | `fixed_smoke_recall` | 1 | -1.305pp | — | — | — | — |
| p2_acr | `fixed_bg_alarm` | 1 | +0.299pp | — | — | 0.568pp | — |
| p2_acr | `fixed_fire_fpr` | 1 | +0.000pp | — | — | — | — |
| p2_acr | `ece_fire_after` | 1 | -0.596pp | — | — | 0.109pp | — |
| p2_acr | `brier_fire_after` | 1 | -0.125pp | — | — | 0.115pp | — |
| p2_acr | `ece_smoke_after` | 1 | -0.223pp | — | — | — | — |
| p2_acr | `brier_smoke_after` | 1 | +0.233pp | — | — | — | — |

## 读法

- **配对 n=1**（srdg / acr）只有点估计，没有 σ，无法判定 —— 这是样本量问题，不是方法问题。
- **配对 n=2**（dcbr）可以给出 σ_d 和 MDD，但 df=1 的 t 临界值高达 12.706，门槛很宽。
- 参照门槛是该终点在**三种子 P2 上**实测的 MDD，用来判断“换终点值不值”。
- 真正要看的：同一个模块的效应，在不同终点下的**符号与量级是否一致**。
  若在低门槛终点上符号翻转或量级乱跳，说明它本来就在噪声里。