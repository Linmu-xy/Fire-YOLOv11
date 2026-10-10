# FASDD n refinement v2：seed0启动记录

日期：2026-10-09。版本说明见`FASDD_N_REFINEMENT_V2.md`，验证报告见`analysis/reports/FASDD_N_REFINEMENT_V2_VALIDATION_20261009.md`。

## 已启动任务

| 机器 | 方法 | 北京时间启动 | supervisor PID | training PID |
|---|---|---|---:|---:|
| cmj | context | 2026-10-09 18:39:09 | 10613 | 10618 |
| lxy | detail_norm | 2026-10-09 18:40:11 | 8932 | 8937 |

统一seed0、YOLO11n、640、batch32、100轮、SGD、AMP=false。登记tag为`fasdd-n-refinement-v2`，以已完成的同配方baseline seed0为对照。

## 实际结果位置

- cmj：`/home/cmj/Documents/FireSmoke/Fire-YOLOv11/results/fasdd_cv_clean_v1/fasdd-n-refinement-v2/context/seed0/`
- lxy：`/home/lxy/Documents/yolo11/Fire-YOLOv11/results/fasdd_cv_clean_v1/fasdd-n-refinement-v2/detail_norm/seed0/`

各目录包含run.json、protocol.yaml、refinement.json、refinement_learning.jsonl和launch源码快照。CSV和检查点由训练器按轮生成。

## 日志位置

- cmj：`/home/cmj/Documents/FireSmoke/Fire-YOLOv11-runtime/fasdd_cv_clean_v1/fasdd-n-refinement-v2/context/seed0/20261009T103909_412660Z/train.log`
- lxy：`/home/lxy/Documents/yolo11/Fire-YOLOv11-runtime/fasdd_cv_clean_v1/fasdd-n-refinement-v2/detail_norm/seed0/20261009T104011_715464Z/train.log`

启动核对时两边均已处理真实训练批次，GPU利用率约96–98%，显存占用约9.1–9.4 GB，损失有限。第一轮尚未完成时不报告验证AP或提升。

小夹具已验证实际恢复训练的模型、EMA、优化器动量与起始轮次；新detail_norm内部任务梯度非零。该验证不构成准确度提升证据。

正式配方的第一批次卷积学习率为0，零输出残差的最初统计可以为0；应看后续轮次的残差/梯度记录。新训练的收益以完整val同轮对比和最终best检查点为准。

后续暂停应安全中断训练进程，保留完整last.pt及其副本，并登记pause.json、状态、CSV轮次和SHA256；按对应tag使用恢复入口。执行新的训练或恢复前检查PID及输出目录，避免重复启动。

## 首轮与正式学习信号检查：18:45（北京时间）

两边都已完成第1轮并进入第2轮，首次last.pt已生成，训练进程仍正常。

| 方法 | 第1轮val mAP50–95 | 第1轮累计秒数 | 第2轮初始P3回归残差/原特征RMS |
|---|---:|---:|---:|
| context | 22.123% | 244.630 | 0.0047% |
| detail_norm | 21.570% | 246.917 | 0.6341% |

新detail_norm的mix/local卷积与BN参数都已出现非零任务梯度，抽样统计全部有限。这证明正式数据上的支路已开始参与学习，不能据此提前判断准确度提升。两种候选均继续100轮完整训练。
