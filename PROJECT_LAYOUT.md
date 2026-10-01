# 项目规范（看这一份就够了）

> **两台电脑（cmj 的机器 / lxy 的机器）共用这一个 Git 仓库。**
> 这份文档只回答三个问题：**东西放哪？怎么改不出错？哪些事别做？**
>
> 最后更新：2026-10-02

---

## 0. 先记住一句话

> **能重新跑出来的，别提交；人的劳动成果，要提交。**

- ❌ **别提交**：数据集、`data/prepared/`、临时文件、缓存、日志
- ✅ **要提交**：代码、配置、文档、结论、**实验结果（含模型权重，方便换电脑直接测试）**

想不清时，用第 1 节的速查表对一下。

---

## 1. 我要放个东西，放哪？（速查表）

| 你要放的东西 | 放这里 | 进 Git 吗 |
|---|---|---|
| 训练脚本、启动器 | `scripts/` | ✅ |
| 核心代码（数据/训练/评估逻辑） | `firesmoke/` | ✅ |
| 实验配置（协议） | `configs/` | ✅ |
| 模型结构定义（`.yaml`） | `configs/models/` | ✅ |
| 方法说明、实验记录、结论 | `docs/` | ✅ |
| 单元测试 | `tests/` | ✅ |
| 预训练权重 | `weights/` | ✅ |
| 训练 / 评估结果 | `results/…`（见第 3 节） | ✅ |
| 分析脚本、分析报告 | `analysis/scripts/`、`analysis/reports/` | ✅ |
| 原始数据集 | `data/raw/<数据集名>/` | ❌ 太大 |
| 临时文件、备份、实验日志 | **项目文件夹外面** | ❌ 别放项目里 |

---

## 2. 目录一览（一级目录只有这 9 个，别再新增）

```
Fire-YOLOv11/
├── configs/     实验配置（协议、模型结构）
├── data/        数据（体积大，不进 Git）
│   ├── raw/         原始数据集，只读别改
│   └── prepared/    跑 prepare 命令自动生成，别手动改
├── docs/        方法 / 协议 / 验证文档
├── firesmoke/   核心 Python 包（真正干活的代码）
├── results/     所有实验结果，按数据集分类
├── scripts/     可执行脚本（训练启动器、数据整理）
├── tests/       单元测试
├── weights/     预训练权重（yolo11n.pt 等）
└── analysis/    离线分析（统计、画图、归因）
    ├── scripts/     分析脚本             ✅ 进 Git
    ├── reports/     分析报告（.md 结论）  ✅ 进 Git
    ├── artifacts/   分析中间产物          ❌ 不进（已在忽略名单）
    └── logs/ work/  运行缓存              ❌ 不进（已在忽略名单）

另外根目录还有：
  README.md           项目介绍、常用命令
  PROJECT_LAYOUT.md   就是本文档
  requirements.txt    环境依赖
  pyproject.toml      包信息
```

**为什么要这么分？**
一句话：**按"是不是能重新跑出来"分家**。代码和结论是人的劳动，要保管好；
数据、缓存、临时产物随时能重造，别塞进 Git 把仓库撑爆。

> 📌 历史包袱说明：这套结构是 2026-10-02 合并过的。
> 以前有 `outputs/`（训练原始输出）和 `experiments/`（归档副本）两个目录装同一批结果，
> `artifacts/` 和 `analysis/` 也功能重叠 —— 现在全部合并进 `results/` 和 `analysis/`，
> **所以看到旧文档里提到 `outputs/`、`experiments/`、`artifacts/`，都是过时的。**

---

## 3. 实验结果放哪？（记住这个公式）

训练结果统一放：

```
results/<数据集>/<配置名>/<方法名>/seed<编号>/
```

举例：

```
results/dfire/paper-v1/baseline/seed0/        ← D-Fire 上，paper-v1 配置，baseline 方法，第 0 个种子
results/dfire/paper-v1_960/p2/seed1/          ← 换了 960 分辨率配置
results/dfire/module-dcbr-v1/p2_dcbr/seed0/   ← DCBR 模块实验
results/dfire/legacy/D-Fire_seed0/            ← 很早期的实验，没有配置名，归到 legacy
results/fsyolo/fs-v1/baseline/seed0/          ← FireSmoke-YOLO 数据集（只有 lxy 那台有这份数据）
```

**训练跑完，结果就在那里，不用再复制一份**（以前那种"跑完再归档一份"的做法已经取消）。

---

## 4. 三个最容易踩的坑 ⚠️（重点看这段）

### 坑 1：`data/prepared/` 换台电脑就用不了

`data/prepared/` 里的文件记录的是**绝对路径**，比如 `/home/cmj/Documents/.../images/train/AoF00000.jpg`。
所以：

- 换电脑 ❌ 用不了
- 把项目文件夹挪个位置 ❌ 用不了
- 拷给同事 ❌ 用不了（他的路径和你不一样）

**解决办法：重新生成一次就行**（几分钟，划分结果不变，不会影响已有实验）：

```bash
cd <项目根目录>
rm -rf data/prepared/<数据集名>          # 先删掉旧的（这一步必须做）
python -m firesmoke prepare --dataset <数据集名>
```

> 报错 `FileNotFoundError: images/train/xxx.jpg` 十有八九就是这个原因。

### 坑 2：两个数据集的**类别顺序是反的**

| 数据集 | 类别 0 | 类别 1 |
|---|---|---|
| `D-Fire` / `dfire` | smoke 烟 | fire 火 |
| `FASDD_CV` / `fasdd` | fire 火 | smoke 烟 |

在 `configs/protocol.yaml` 里写 `source_names` 时**必须写对顺序**。
**写错了不会报错**，只会让模型的"火/烟"标签整体错位，白跑一场实验。

### 坑 3：改路径要同步改 6 个地方

如果你要改目录结构（比如把 `data/` 改名），这些地方**都要一起改**，漏一个就出错：

1. `configs/protocol.yaml` ← 主配置，路径都在这
2. `configs/protocol_800.yaml`
3. `configs/protocol_960.yaml`
4. `configs/protocol-v2.yaml`
5. `configs/protocol-dcbr.yaml`

   > 上面 4 个变体**仍然被脚本和测试引用，不能删**，路径要和主配置保持一致。

6. `firesmoke/experiment.py` 里的输出路径公式 + `scripts/` 里写死的路径

改完先自查：

```bash
python -m unittest discover -s tests                     # 测试全绿？
python scripts/train_p2_common.py --seed 0 --dry-run     # 输出路径对不对？
grep -rn "outputs/\|experiments/" firesmoke scripts docs # 还有旧路径吗？（应该是空的）
```

---

## 5. 做一次实验的标准流程

```bash
# ① 进项目、确认在正确的分支
cd /home/cmj/Documents/FireSmoke/Fire-YOLOv11     # lxy 那台改成 /home/lxy/Documents/yolo11/Fire-YOLOv11
git pull --ff-only

# ② 第一次跑、或换过电脑 / 挪过文件夹：生成数据（只需做一次）
python -m firesmoke prepare --dataset dfire

# ③ 先预检（不会真的训练，只检查环境、数据、输出路径）
python scripts/train_p2_common.py --seed 0 --dry-run

# ④ 确认没问题再开跑
python scripts/train_p2_common.py --seed 0 --execute

# ⑤ 结果就在 results/dfire/paper-v1/p2/seed0/，可以直接用了
```

**换配置 / 换分辨率**：改 `--tag`（比如 `--tag paper-v1_960`），结果会落到对应的新目录。

---

## 6. 两台电脑怎么同步

| | cmj 的机器 | lxy 的机器 |
|---|---|---|
| 项目路径 | `/home/cmj/Documents/FireSmoke/Fire-YOLOv11` | `/home/lxy/Documents/yolo11/Fire-YOLOv11` |
| 环境 | conda 环境 `yolo11` | conda 环境 `yolo11` |
| GPU | RTX 3090 | RTX 3090 |

**路径不一样，所以 `data/` 这台机器的东西不能直接拷给那台**（见坑 1）。
其余的代码、配置、文档都是可移植的。

日常同步就两步：

```bash
git pull --ff-only                              # ① 拉最新代码
python -m firesmoke prepare --dataset dfire     # ② 只在 data/ 或项目路径变过时才需要
```

**同步前先确认两边一致**（这三条命令在两边跑，结果应该一模一样）：

```bash
git rev-parse HEAD                      # 当前提交 ID
git status --short                      # 应该是空的（没有未提交改动）
git ls-files -s | md5sum                # 文件指纹，两边必须相同
```

---

## 7. 别做这些事（会坑到别人）

1. ❌ **别把 `data/` 提交上去** —— 16 G+，而且换台电脑本来就用不了
2. ❌ **别在 `configs/`、`scripts/` 里留备份文件**（`xxx.orig`、`xxx.bak`、`xxx.old`）—— 备份放**项目文件夹外面**
3. ❌ **别在代码里写死 `/home/xxx/...` 这种路径** —— 换台电脑就废。正确做法：

   ```python
   # ❌ 错
   SRC = Path("/home/lxy/Documents/yolo11/D-Fire")

   # ✅ 对：从项目根目录推导
   ROOT = Path(__file__).resolve().parent.parent
   SRC = ROOT / "data/raw/D-Fire"
   ```

4. ❌ **别手动改 `data/prepared/` 里的 `manifest.jsonl` / `train.txt` / `metadata.json`** ——
   程序会校验文件指纹（sha256），改了一定报 `Prepared manifest was modified`。要改就重新 `prepare`
5. ❌ **别覆盖 `results/` 里已有的实验** —— 它们是已发表结论的依据
   （比如 `paper-v1/baseline/seed0` 和 `paper-v1/p2/seed0` 是一对对照实验）。
   程序本身也会拒绝写进已存在的输出目录，想重跑请换 `--tag`
6. ❌ **别在项目文件夹里放备份或临时目录**（如旧的 `backups/`、`runs/`）—— 放外面
7. ❌ **别新增和现有 9 个目录职责重叠的一级目录**（比如再建一个 `out/`、`logs/`）
8. ❌ **别删 `configs/` 里的 4 个 `protocol_*.yaml` 变体** —— 脚本和测试还在用

---

## 8. 提交前自查（30 秒）

```bash
git status --short          # 看清楚自己动了什么，别把不该提交的带上
git ls-files | wc -l        # 文件总数，不应该因为跑实验而暴涨
du -sh .git                 # 仓库大小，历史里曾误入过 2.75M 行的 json，导致 .git 涨到 300M+
```

**提交信息写清楚**：做了什么、为什么。例：
`Fix data path after moving project folder`、`Add DCBR module ablation results`

---

## 9. 万一要改目录结构（流程）

改结构会影响所有历史实验的可读性，所以按顺序来：

1. **先备份到项目外面**：
   ```bash
   rsync -a <项目>/ <项目外面的备份目录>/
   ```
2. 改配置（第 4 节坑 3 的 6 个地方）+ 本文档 + `README.md` + `docs/`
3. 按第 4 节坑 3 的三条命令自查
4. 两台电脑都做一遍，各自重新 `prepare`
5. 提交推送，并**告诉另一台电脑上的人**拉取 + 重新 `prepare`

---

## 10. 遇到问题先看这里

| 现象 | 最可能的原因 | 怎么办 |
|---|---|---|
| `FileNotFoundError: images/train/xxx.jpg` | `data/prepared/` 路径失效 | 重新 `prepare`（坑 1） |
| `Prepared manifest was modified` | 手改了 `data/prepared/` 里的文件 | 删掉重新 `prepare` |
| 报错说找不到 `configs/xxx.yaml` | 在错误的目录下执行命令 | `cd` 到项目根目录再跑 |
| 训练报 CUDA 不可用 | 驱动 / 显卡占用问题 | 先 `nvidia-smi` 确认，或临时把 `protocol.yaml` 的 `device` 改成 `'cpu'` |
| 输出目录已存在，跑不了 | v1 协议不覆盖已有实验 | 换个 `--tag` |
| 火/烟标签感觉反了 | `source_names` 顺序写错（坑 2） | 对照第 4 节的表格检查 |
