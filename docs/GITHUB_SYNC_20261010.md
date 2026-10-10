# 两台机器 GitHub 同步记录（2026-10-10）

同步范围：代码、协议、文档、分析报告，以及已完成实验的结果、best/last 权重与运行来源记录。

## 已合并

- cmj：FASDD clean-v1 baseline seed1、V1 context seed0、V2 context seed0。
- lxy：FASDD clean-v1 baseline seed0、V1 detail seed0、V2 detail_norm seed0、stage3 frozen_quality_probe seed0。
- 两边共同的 FASDD 训练与分析实现、stage4 设计；lxy 独有的 Arm B 数据方案和零样本评估脚本更新。
- 本机保留的研究规划文档与诊断报告生成脚本。
- 质量头失败审计的紧凑统计和诊断图，放在 analysis/reports/evidence/；原始逐框中间文件保留在原机器。

## 待训练结束补同步

`results/fasdd_cv_clean_v1/fasdd-n-stage3/reg_transfer/seed0/`

首次检查仍在训练，已完成第 96/100 轮。整个目录排除，避免上传变化中的 CSV、best/last 权重和不完整结束状态。结束后确认进程退出、完成状态与 checkpoint 稳定，再单独补提交。

## 机器本地数据

按 PROJECT_LAYOUT.md 与 .gitignore，原始数据集、data/prepared、analysis/artifacts、日志、缓存及项目外 runtime 保留在各机器。数据构建脚本、清理规则、协议和内容指纹纳入同步；不改已有划分或标签。

来源文件的 SHA256、参与机器及基础提交见 [同步清单](GITHUB_SYNC_20261010.json)。清单记录同步前来源，报告的可移植图片链接和本记录是同步期间的新修改。没有重跑训练或实现测试。
