# FASDD 误报归因（目标域，源域冻结阈值）

- 冻结阈值 fire=0.3143, smoke=0.6057（来自源域 D-Fire val 的 1% FPR）
- 背景集 = 文件名前缀 `neitherFireNorSmoke` 的图像（全部无 GT）

## 1. 背景集告警分解

| 配置 | 背景图数 | 仅 fire 触发 | 仅 smoke 触发 | 两类都触发 | **任意触发** | 未触发 |
|---|---:|---:|---:|---:|---:|---:|
| baseline640_s0 | 6533 | 472 (7.2%) | 149 (2.3%) | 20 (0.3%) | **641 (9.81%)** | 5892 (90.2%) |
| p2_s0 | 6533 | 482 (7.4%) | 138 (2.1%) | 12 (0.2%) | **632 (9.67%)** | 5901 (90.3%) |
| p2_s1 | 6533 | 516 (7.9%) | 168 (2.6%) | 20 (0.3%) | **704 (10.78%)** | 5829 (89.2%) |
| p2_s2 | 6533 | 497 (7.6%) | 156 (2.4%) | 19 (0.3%) | **672 (10.29%)** | 5861 (89.7%) |
| p2_srdg_s0 | 6533 | 510 (7.8%) | 136 (2.1%) | 16 (0.2%) | **662 (10.13%)** | 5871 (89.9%) |
| p2_dcbr_s0 | 6533 | 646 (9.9%) | 143 (2.2%) | 16 (0.2%) | **805 (12.32%)** | 5728 (87.7%) |
| p2_dcbr_s2 | 6533 | 449 (6.9%) | 154 (2.4%) | 18 (0.3%) | **621 (9.51%)** | 5912 (90.5%) |
| p2_acr_s0 | 6533 | 538 (8.2%) | 144 (2.2%) | 16 (0.2%) | **698 (10.68%)** | 5835 (89.3%) |

## 2. 背景集触发率按分辨率

| 配置 | 分辨率档 | 图数 | 触发数 | 触发率 |
|---|---|---:|---:|---:|
| baseline640_s0 | <=480 | 963 | 115 | **11.9%** |
| baseline640_s0 | 481-720 | 2962 | 313 | **10.6%** |
| baseline640_s0 | 721-1080 | 1761 | 140 | **8.0%** |
| baseline640_s0 | 1081-1440 | 453 | 41 | **9.1%** |
| baseline640_s0 | 1441-1920 | 198 | 17 | **8.6%** |
| baseline640_s0 | >1920 | 196 | 15 | **7.7%** |
| p2_s0 | <=480 | 963 | 107 | **11.1%** |
| p2_s0 | 481-720 | 2962 | 318 | **10.7%** |
| p2_s0 | 721-1080 | 1761 | 136 | **7.7%** |
| p2_s0 | 1081-1440 | 453 | 41 | **9.1%** |
| p2_s0 | 1441-1920 | 198 | 17 | **8.6%** |
| p2_s0 | >1920 | 196 | 13 | **6.6%** |
| p2_s1 | <=480 | 963 | 117 | **12.1%** |
| p2_s1 | 481-720 | 2962 | 352 | **11.9%** |
| p2_s1 | 721-1080 | 1761 | 160 | **9.1%** |
| p2_s1 | 1081-1440 | 453 | 39 | **8.6%** |
| p2_s1 | 1441-1920 | 198 | 17 | **8.6%** |
| p2_s1 | >1920 | 196 | 19 | **9.7%** |
| p2_s2 | <=480 | 963 | 114 | **11.8%** |
| p2_s2 | 481-720 | 2962 | 348 | **11.7%** |
| p2_s2 | 721-1080 | 1761 | 136 | **7.7%** |
| p2_s2 | 1081-1440 | 453 | 37 | **8.2%** |
| p2_s2 | 1441-1920 | 198 | 19 | **9.6%** |
| p2_s2 | >1920 | 196 | 18 | **9.2%** |
| p2_srdg_s0 | <=480 | 963 | 112 | **11.6%** |
| p2_srdg_s0 | 481-720 | 2962 | 332 | **11.2%** |
| p2_srdg_s0 | 721-1080 | 1761 | 145 | **8.2%** |
| p2_srdg_s0 | 1081-1440 | 453 | 39 | **8.6%** |
| p2_srdg_s0 | 1441-1920 | 198 | 17 | **8.6%** |
| p2_srdg_s0 | >1920 | 196 | 17 | **8.7%** |
| p2_dcbr_s0 | <=480 | 963 | 136 | **14.1%** |
| p2_dcbr_s0 | 481-720 | 2962 | 398 | **13.4%** |
| p2_dcbr_s0 | 721-1080 | 1761 | 178 | **10.1%** |
| p2_dcbr_s0 | 1081-1440 | 453 | 49 | **10.8%** |
| p2_dcbr_s0 | 1441-1920 | 198 | 24 | **12.1%** |
| p2_dcbr_s0 | >1920 | 196 | 20 | **10.2%** |
| p2_dcbr_s2 | <=480 | 963 | 100 | **10.4%** |
| p2_dcbr_s2 | 481-720 | 2962 | 304 | **10.3%** |
| p2_dcbr_s2 | 721-1080 | 1761 | 136 | **7.7%** |
| p2_dcbr_s2 | 1081-1440 | 453 | 43 | **9.5%** |
| p2_dcbr_s2 | 1441-1920 | 198 | 23 | **11.6%** |
| p2_dcbr_s2 | >1920 | 196 | 15 | **7.7%** |
| p2_acr_s0 | <=480 | 963 | 115 | **11.9%** |
| p2_acr_s0 | 481-720 | 2962 | 345 | **11.6%** |
| p2_acr_s0 | 721-1080 | 1761 | 156 | **8.9%** |
| p2_acr_s0 | 1081-1440 | 453 | 44 | **9.7%** |
| p2_acr_s0 | 1441-1920 | 198 | 24 | **12.1%** |
| p2_acr_s0 | >1920 | 196 | 14 | **7.1%** |

## 3. 正样本召回按图像类别

| 配置 | 图类别 | 图数 | fire 召回 | smoke 召回 |
|---|---|---:|---:|---:|
| baseline640_s0 | fire | 2091 | 90.9% | — |
| baseline640_s0 | smoke | 3902 | — | 32.6% |
| baseline640_s0 | bothFireAndSmoke | 3358 | 95.3% | 53.2% |
| p2_s0 | fire | 2091 | 90.0% | — |
| p2_s0 | smoke | 3902 | — | 33.9% |
| p2_s0 | bothFireAndSmoke | 3358 | 94.8% | 51.8% |
| p2_s1 | fire | 2091 | 76.5% | — |
| p2_s1 | smoke | 3902 | — | 35.6% |
| p2_s1 | bothFireAndSmoke | 3358 | 95.1% | 52.1% |
| p2_s2 | fire | 2091 | 85.3% | — |
| p2_s2 | smoke | 3902 | — | 34.2% |
| p2_s2 | bothFireAndSmoke | 3358 | 95.4% | 53.1% |
| p2_srdg_s0 | fire | 2091 | 86.8% | — |
| p2_srdg_s0 | smoke | 3902 | — | 36.2% |
| p2_srdg_s0 | bothFireAndSmoke | 3358 | 95.0% | 51.1% |
| p2_dcbr_s0 | fire | 2091 | 85.5% | — |
| p2_dcbr_s0 | smoke | 3902 | — | 39.0% |
| p2_dcbr_s0 | bothFireAndSmoke | 3358 | 94.4% | 48.1% |
| p2_dcbr_s2 | fire | 2091 | 87.7% | — |
| p2_dcbr_s2 | smoke | 3902 | — | 29.9% |
| p2_dcbr_s2 | bothFireAndSmoke | 3358 | 96.2% | 51.6% |
| p2_acr_s0 | fire | 2091 | 89.6% | — |
| p2_acr_s0 | smoke | 3902 | — | 32.3% |
| p2_acr_s0 | bothFireAndSmoke | 3358 | 95.4% | 50.0% |

## 4. 背景集最高分的图像（最难负样本）

| 配置 | max(fire,smoke) | 文件名 | fire 最高分 | smoke 最高分 |
|---|---:|---|---:|---:|
| baseline640_s0 | 0.9320 | `neitherFireNorSmoke_CV001265.jpg` | 0.932 | 0.002 |
| baseline640_s0 | 0.8958 | `neitherFireNorSmoke_CV014005.jpg` | 0.061 | 0.896 |
| baseline640_s0 | 0.8623 | `neitherFireNorSmoke_CV012573.jpg` | 0.862 | 0.003 |
| baseline640_s0 | 0.8565 | `neitherFireNorSmoke_CV021881.jpg` | 0.857 | 0.378 |
| baseline640_s0 | 0.8560 | `neitherFireNorSmoke_CV015926.jpg` | 0.856 | 0.161 |
| baseline640_s0 | 0.8543 | `neitherFireNorSmoke_CV027856.jpg` | 0.000 | 0.854 |
| baseline640_s0 | 0.8535 | `neitherFireNorSmoke_CV007271.jpg` | 0.150 | 0.853 |
| baseline640_s0 | 0.8510 | `neitherFireNorSmoke_CV004271.jpg` | 0.020 | 0.851 |
| p2_s0 | 0.9004 | `neitherFireNorSmoke_CV025285.jpg` | 0.000 | 0.900 |
| p2_s0 | 0.8884 | `neitherFireNorSmoke_CV014005.jpg` | 0.023 | 0.888 |
| p2_s0 | 0.8842 | `neitherFireNorSmoke_CV012573.jpg` | 0.884 | 0.006 |
| p2_s0 | 0.8819 | `neitherFireNorSmoke_CV004271.jpg` | 0.002 | 0.882 |
| p2_s0 | 0.8801 | `neitherFireNorSmoke_CV007271.jpg` | 0.057 | 0.880 |
| p2_s0 | 0.8678 | `neitherFireNorSmoke_CV027856.jpg` | 0.007 | 0.868 |
| p2_s0 | 0.8609 | `neitherFireNorSmoke_CV002831.jpg` | 0.861 | 0.002 |
| p2_s0 | 0.8595 | `neitherFireNorSmoke_CV000904.jpg` | 0.002 | 0.860 |
| p2_s1 | 0.8974 | `neitherFireNorSmoke_CV030571.jpg` | 0.897 | 0.004 |
| p2_s1 | 0.8947 | `neitherFireNorSmoke_CV014005.jpg` | 0.008 | 0.895 |
| p2_s1 | 0.8605 | `neitherFireNorSmoke_CV015974.jpg` | 0.861 | 0.065 |
| p2_s1 | 0.8571 | `neitherFireNorSmoke_CV035214.jpg` | 0.054 | 0.857 |
| p2_s1 | 0.8533 | `neitherFireNorSmoke_CV002243.jpg` | 0.853 | 0.000 |
| p2_s1 | 0.8521 | `neitherFireNorSmoke_CV018165.jpg` | 0.852 | 0.018 |
| p2_s1 | 0.8517 | `neitherFireNorSmoke_CV012573.jpg` | 0.852 | 0.002 |
| p2_s1 | 0.8485 | `neitherFireNorSmoke_CV028936.jpg` | 0.123 | 0.849 |
| p2_s2 | 0.8876 | `neitherFireNorSmoke_CV012573.jpg` | 0.888 | 0.019 |
| p2_s2 | 0.8833 | `neitherFireNorSmoke_CV007194.jpg` | 0.883 | 0.058 |
| p2_s2 | 0.8788 | `neitherFireNorSmoke_CV028936.jpg` | 0.063 | 0.879 |
| p2_s2 | 0.8719 | `neitherFireNorSmoke_CV023023.jpg` | 0.089 | 0.872 |
| p2_s2 | 0.8693 | `neitherFireNorSmoke_CV018165.jpg` | 0.869 | 0.038 |
| p2_s2 | 0.8681 | `neitherFireNorSmoke_CV026121.jpg` | 0.447 | 0.868 |
| p2_s2 | 0.8669 | `neitherFireNorSmoke_CV011801.jpg` | 0.136 | 0.867 |
| p2_s2 | 0.8634 | `neitherFireNorSmoke_CV018060.jpg` | 0.093 | 0.863 |
| p2_srdg_s0 | 0.9093 | `neitherFireNorSmoke_CV012573.jpg` | 0.909 | 0.006 |
| p2_srdg_s0 | 0.8914 | `neitherFireNorSmoke_CV014005.jpg` | 0.016 | 0.891 |
| p2_srdg_s0 | 0.8679 | `neitherFireNorSmoke_CV020976.jpg` | 0.868 | 0.001 |
| p2_srdg_s0 | 0.8669 | `neitherFireNorSmoke_CV002243.jpg` | 0.867 | 0.000 |
| p2_srdg_s0 | 0.8663 | `neitherFireNorSmoke_CV025352.jpg` | 0.866 | 0.015 |
| p2_srdg_s0 | 0.8617 | `neitherFireNorSmoke_CV028936.jpg` | 0.201 | 0.862 |
| p2_srdg_s0 | 0.8548 | `neitherFireNorSmoke_CV004271.jpg` | 0.003 | 0.855 |
| p2_srdg_s0 | 0.8536 | `neitherFireNorSmoke_CV031301.jpg` | 0.854 | 0.003 |
| p2_dcbr_s0 | 0.9266 | `neitherFireNorSmoke_CV001265.jpg` | 0.927 | 0.000 |
| p2_dcbr_s0 | 0.8839 | `neitherFireNorSmoke_CV012573.jpg` | 0.884 | 0.010 |
| p2_dcbr_s0 | 0.8760 | `neitherFireNorSmoke_CV014005.jpg` | 0.036 | 0.876 |
| p2_dcbr_s0 | 0.8748 | `neitherFireNorSmoke_CV021881.jpg` | 0.875 | 0.645 |
| p2_dcbr_s0 | 0.8698 | `neitherFireNorSmoke_CV011036.jpg` | 0.000 | 0.870 |
| p2_dcbr_s0 | 0.8651 | `neitherFireNorSmoke_CV004271.jpg` | 0.007 | 0.865 |
| p2_dcbr_s0 | 0.8615 | `neitherFireNorSmoke_CV038141.jpg` | 0.018 | 0.861 |
| p2_dcbr_s0 | 0.8584 | `neitherFireNorSmoke_CV015926.jpg` | 0.858 | 0.139 |
| p2_dcbr_s2 | 0.8957 | `neitherFireNorSmoke_CV012573.jpg` | 0.896 | 0.007 |
| p2_dcbr_s2 | 0.8638 | `neitherFireNorSmoke_CV024208.jpg` | 0.864 | 0.008 |
| p2_dcbr_s2 | 0.8614 | `neitherFireNorSmoke_CV026121.jpg` | 0.543 | 0.861 |
| p2_dcbr_s2 | 0.8603 | `neitherFireNorSmoke_CV014005.jpg` | 0.023 | 0.860 |
| p2_dcbr_s2 | 0.8535 | `neitherFireNorSmoke_CV022868.jpg` | 0.109 | 0.854 |
| p2_dcbr_s2 | 0.8502 | `neitherFireNorSmoke_CV030571.jpg` | 0.850 | 0.005 |
| p2_dcbr_s2 | 0.8502 | `neitherFireNorSmoke_CV025285.jpg` | 0.001 | 0.850 |
| p2_dcbr_s2 | 0.8494 | `neitherFireNorSmoke_CV025147.jpg` | 0.000 | 0.849 |
| p2_acr_s0 | 0.9280 | `neitherFireNorSmoke_CV012573.jpg` | 0.928 | 0.016 |
| p2_acr_s0 | 0.9025 | `neitherFireNorSmoke_CV025285.jpg` | 0.000 | 0.903 |
| p2_acr_s0 | 0.8800 | `neitherFireNorSmoke_CV001265.jpg` | 0.880 | 0.008 |
| p2_acr_s0 | 0.8616 | `neitherFireNorSmoke_CV021881.jpg` | 0.862 | 0.390 |
| p2_acr_s0 | 0.8508 | `neitherFireNorSmoke_CV011007.jpg` | 0.003 | 0.851 |
| p2_acr_s0 | 0.8452 | `neitherFireNorSmoke_CV022868.jpg` | 0.436 | 0.845 |
| p2_acr_s0 | 0.8434 | `neitherFireNorSmoke_CV035214.jpg` | 0.043 | 0.843 |
| p2_acr_s0 | 0.8388 | `neitherFireNorSmoke_CV015913.jpg` | 0.216 | 0.839 |

## 5. 背景集分数分布

| 配置 | 类 | 均值 | p90 | p99 | 最大 |
|---|---|---:|---:|---:|---:|
| baseline640_s0 | fire | 0.0659 | 0.2436 | 0.6665 | 0.9320 |
| baseline640_s0 | smoke | 0.0843 | 0.2903 | 0.7235 | 0.8958 |
| p2_s0 | fire | 0.0641 | 0.2326 | 0.6747 | 0.8842 |
| p2_s0 | smoke | 0.0780 | 0.2637 | 0.7160 | 0.9004 |
| p2_s1 | fire | 0.0708 | 0.2678 | 0.6795 | 0.8974 |
| p2_s1 | smoke | 0.0927 | 0.3190 | 0.7239 | 0.8947 |
| p2_s2 | fire | 0.0677 | 0.2529 | 0.6741 | 0.8876 |
| p2_s2 | smoke | 0.0907 | 0.2995 | 0.7172 | 0.8788 |
| p2_srdg_s0 | fire | 0.0670 | 0.2492 | 0.6622 | 0.9093 |
| p2_srdg_s0 | smoke | 0.0891 | 0.3022 | 0.7126 | 0.8914 |
| p2_dcbr_s0 | fire | 0.0842 | 0.3158 | 0.6938 | 0.9266 |
| p2_dcbr_s0 | smoke | 0.0956 | 0.3107 | 0.7187 | 0.8760 |
| p2_dcbr_s2 | fire | 0.0647 | 0.2228 | 0.6576 | 0.8957 |
| p2_dcbr_s2 | smoke | 0.0905 | 0.3074 | 0.7160 | 0.8614 |
| p2_acr_s0 | fire | 0.0739 | 0.2748 | 0.6526 | 0.9280 |
| p2_acr_s0 | smoke | 0.0957 | 0.3157 | 0.7144 | 0.9025 |

## 6. 读法

- 看第 1 节的**触发类别构成**：若 smoke 侧占主导，说明误报主要来自「把非烟当烟」；
  若 fire 侧主导，则是「把暖色/高亮当火」。这决定 hard-negative 该采哪一类。
- 看第 2 节：若触发率随分辨率单调变化，说明是尺度/细节问题；若各档相近，
  说明是**语义混淆**（与分辨率无关），那才是真正需要负样本的地方。
- 看第 3 节：`smoke` 单独那一类的召回，能区分「烟本身难」与「火烟同框时才难」。
- 第 4 节给出具体文件名，可以直接打开看是什么东西被误报了。

> 本报告不含任何新训练或新推理；全部来自已有 predictions。