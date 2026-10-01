# 项目结构说明与存放规范（PROJECT_LAYOUT）

> 本文档是仓库的**结构契约**。新增文件前请先读本页，按「存放规范」放置；
> 不确定时**先问**，不要随手在根目录或 `configs/` 里新建文件。
>
> 最后更新：2026-10-02（数据层由 `datasets/` + `artifacts/data/` 归并为 `data/`）

---

## 1. 顶层结构

```
Fire-YOLOv11/
├── README.md                 项目入口：环境、训练/评估命令、注意事项
├── PROJECT_LAYOUT.md         本文件：结构契约与存放规范
├── pyproject.toml            包元数据
├── requirements.txt          环境依赖（由 conda env yolo11 导出）
├── .gitignore                忽略规则（见第 5 节）
│
├── configs/                  【配置层】
│   ├── protocol.yaml             主实验协议（唯一配置源）
│   ├── protocol_*.yaml           历史/变体协议（如 protocol_800/960/1280）
│   └── models/                   模型结构定义（如 yolo11n-p2.yaml）
│
├── firesmoke/                【代码层】核心 Python 包
│   ├── data.py                   数据清点 / 规范化 / 校验
│   ├── experiment.py             训练编排
│   ├── evaluation.py             评估与校准
│   ├── cli.py, common.py, models.py, ...
│
├── scripts/                  【代码层】可执行脚本
│   ├── train_p2_common.py        参数化训练启动器（推荐入口）
│   ├── train_acr.py / train_dcbr.py / train_srdg_seed0.py
│   ├── prepare_dataset.py        D-Fire 初始划分（历史记录，需 --src）
│   └── train_*.py                各实验的薄封装
│
├── docs/                     【文档层】方法 / 协议 / 验证说明
├── tests/                    【测试层】单元测试
│
├── data/                     【数据层】—— 体积大，**永不入库**
│   ├── raw/                      原始数据集（只读，不要改写）
│   │   ├── D-Fire/               21527 张，0=smoke, 1=fire
│   │   └── FASDD_CV/             95314 张，0=fire, 1=smoke
│   └── prepared/                 由 `python -m firesmoke prepare` 生成
│       ├── dfire/  fasdd/  dfire_hn3/ ...
│       └── 每个含 images/ labels/ manifest.jsonl data.yaml metadata.json
│
├── weights/                  【权重层】预训练权重（yolo11n.pt / yolo26n.pt）
│
├── outputs/                  【产物层】协议原始输出（历史已入库）
│   └── <tag>/<dataset>/<method>/seed<N>/
│
├── experiments/              【产物层】训练成功后的归档副本（历史已入库）
│   └── <算法>/<数据集>/
│
└── artifacts/                【产物层】审计 / 校验 / 评估产物（不入库）
    ├── audit/  checks/  eval/
    └── analysis/
```

---

## 2. 七个层的职责（「这个文件该放哪」）

| 层 | 目录 | 放什么 | 入库 |
|---|---|---|---|
| 配置 | `configs/` | 协议 YAML、模型定义 | ✅ |
| 代码 | `firesmoke/` | 可复用的包代码 | ✅ |
| 代码 | `scripts/` | 一次性/参数化的运行脚本 | ✅ |
| 文档 | `docs/` | 方法、协议、验证记录 | ✅ |
| 测试 | `tests/` | 单元测试 | ✅ |
| 数据 | `data/raw/` | 原始数据集 | ❌ |
| 数据 | `data/prepared/` | prepare 产物（symlink） | ❌ |
| 权重 | `weights/` | `.pt` 预训练权重 | ✅（现状） |
| 产物 | `outputs/` | 协议原始输出 | ✅（历史遗留，见 §5） |
| 产物 | `experiments/` | 归档副本 | ✅（历史遗留，见 §5） |
| 产物 | `artifacts/` | 审计/评估中间产物 | ❌ |

**一句话**：**能重新生成的**都不该入库；**人的劳动成果**（代码/配置/文档/结论）才入库。

---

## 3. 路径规范（最容易出错的地方）

### 3.1 绝对禁止硬编码机器路径

```python
# ❌ 错：换台机器就废
SRC = Path("/home/lxy/Documents/yolo11/D-Fire(1)")
DST = Path("/home/lxy/Documents/yolo11/Fire-YOLOv11/datasets/D-Fire")

# ✅ 对：从仓库根推导
ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data/raw/D-Fire"
```

需要外部路径时用**命令行参数**或环境变量，并给相对仓库根的默认值。

### 3.2 配置里的路径一律相对仓库根

`configs/protocol.yaml` 顶部已注明：

> Paths are relative to the repository, not to your terminal's working directory.

`firesmoke/common.py:path()` 会把相对路径解析为 `<repo>/...`，所以**在任何目录下运行都成立**。

当前四项：

```yaml
datasets:
  dfire:  {root: data/raw/D-Fire}
  fasdd:  {root: data/raw/FASDD_CV}
prepared_root: data/prepared
results_root: outputs
weights: weights/yolo11n.pt
```

### 3.3 ⚠️ `data/prepared/` 里的 manifest 存的是**绝对路径**

`manifest.jsonl`、`train.txt`、`val.txt`、`test.txt`、`data.yaml` 都会写入**当时的绝对路径**。
这意味着：

- **移动本仓库目录、移动 `data/`、换机器 → 这些文件立即失效**（报 `FileNotFoundError`）
- 修复办法只有一个：**重新 prepare**

```bash
cd <repo>
rm -rf data/prepared/<dataset>          # fresh_dir 要求目录不存在
python -m firesmoke prepare --dataset <dataset>
```

- 划分由 `data/raw/<dataset>/images/{train,val,test}` 的目录结构决定，**结果确定**，重跑不会改变 train/val/test 成员
- 因此 `outputs/`、`experiments/` 里的历史实验**仍然可比**，无需重训
- ⚠️ 但**不要**把 `data/prepared/` 拷给别人：跨机器路径必然不同，对方必须自己重新 prepare

### 3.4 新增数据集

1. 把原始数据放到 `data/raw/<名字>/`（保持 `images/{train,val,test}` + `labels/{train,val,test}` 结构）
2. 在 `configs/protocol.yaml` 的 `datasets:` 下加一条：`root` / `layout` / `source_names` / `invalid_box_policy`
3. 跑 `python -m firesmoke prepare --dataset <名字>`
4. ⚠️ **`source_names` 必须写清类别顺序**：D-Fire 是 `[smoke, fire]`，FASDD_CV 是 `[fire, smoke]`，两者相反，写错会导致标签错位且**不会报错**

---

## 4. 命名规范

| 对象 | 规范 | 示例 |
|---|---|---|
| 协议文件 | `protocol.yaml` 为主，变体 `protocol_<参数>.yaml` | `protocol_960.yaml` |
| 输出 tag | `<阶段>-v<版本>`，变体接 `_<参数>` | `paper-v1`、`paper-v1_960` |
| 实验归档 | `<数据集>_<方法>_seed<N>[_<模块tag>]` | `D-Fire_p2_dcbr_seed0_module-dcbr-v1` |
| 脚本 | `train_<方法>[_<参数>].py` | `train_baseline_seed0_960.py` |
| 文档 | 全大写 + 下划线 | `MODULE_DCBR.md` |

**避免**（历史遗留，新文件不要这样）：

- 把参数塞进目录名后不再维护（`paper-v1_800` vs `paper-v1_960` 并存且无索引）
- 备份文件留在配置目录（`configs/protocol.yaml.orig_20261001` ❌ → 备份一律放项目外）
- 权重放在 `scripts/`（`scripts/yolo26n.pt` ❌ → 统一放 `weights/`）

---

## 5. .gitignore 与「历史产物仍入库」的现状

当前 `outputs/`、`experiments/`、`weights/` 下的 **645 个文件早已被 git 跟踪**（占全部跟踪文件的 91%）。
历史包袱已保留，**但请不要再新增**：

- `.gitignore` 里 `outputs/` 一行是**注释掉**的（`# outputs/`），因为注释它才能让历史产物继续被跟踪；
  新增产物会显示为 `??`
- **例外**：`weights/*.pt`（小且必需）建议保留入库

**提交前自查**：

```bash
git status --short
git ls-files | wc -l          # 应保持稳定，不应因训练而暴涨
du -sh .git                   # 历史里曾误入 275 万行的 predictions.json，.git 因此达 359M
```

---

## 6. 禁止事项（给后人）

1. ❌ **不要把 `data/` 拷进仓库或提交**：16 G，且 `prepared/` 是机器绑定的
2. ❌ **不要在 `configs/`、`scripts/` 里新建备份文件**（`*.orig*`、`*.bak`、`*.old`）——备份放项目外
3. ❌ **不要硬编码 `/home/<用户>/...`**（见 §3.1）
4. ❌ **不要把权重、可视化 jpg/png、`best.pt` 提交进新产物目录**
5. ❌ **不要手改 `data/prepared/` 里的 `manifest.jsonl` / `*.txt` / `metadata.json`**：
   `load_prepared()` 会校验 sha256，改了必然报 `Prepared manifest was modified`。要改就重新 prepare
6. ❌ **不要覆盖 `experiments/` 里的历史归档**：它们是已发表结论的依据（如 `D-Fire_seed0` 与 `D-Fire_p2_seed0` 是配对对照）
7. ❌ **不要在项目目录内放备份**（`backups/`）——放项目同级的外部目录

---

## 7. 本项目在两台机器上运行（重要）

| | 机器 A（cmj） | 机器 B（lxy） |
|---|---|---|
| 仓库路径 | `/home/cmj/Documents/FireSmoke/Fire-YOLOv11` | `/home/lxy/Documents/yolo11/Fire-YOLOv11` |
| conda env | `yolo11` | `yolo11` |
| GPU | RTX 3090 | RTX 3090 |

**根路径不同 → `data/prepared/` 绝对不能互相拷贝**，各自跑 `prepare`。
**其余（`configs/`、`firesmoke/`、`scripts/`、`docs/`）全部可移植**，因为路径都相对仓库根。

同步流程：

```bash
git pull --ff-only                 # 拉取
python -m firesmoke prepare --dataset dfire   # 仅在 data/ 或仓库路径变过时才需要
```

---

## 8. 常用命令（都在仓库根执行）

```bash
# 数据规范化（新机器/移动目录后必跑）
python -m firesmoke prepare --dataset dfire
python -m firesmoke prepare --dataset fasdd

# 数据审计
python -m firesmoke audit --datasets dfire fasdd --output artifacts/audit/metadata

# 训练（协议化入口）
python scripts/train_p2_common.py --seed 0 --dry-run     # 预检
python scripts/train_p2_common.py --seed 0 --execute     # 真正开跑

# 测试
python -m unittest discover -s tests -v
```

---

## 9. 变更本结构的流程

结构调整（移动/重命名目录、改 `protocol.yaml` 的路径项）会**影响所有历史实验的可读性**，必须：

1. 先在项目**外**做全量备份（`rsync -a <repo>/ <外部目录>/`）
2. 改 `configs/protocol.yaml` + 相关 `scripts/` 硬编码 + 本文件 + `README.md`
3. 重新 `prepare` 并验证：
   ```bash
   python -c "from firesmoke.data import load_prepared, load_protocol; \
   cfg=load_protocol('configs/protocol.yaml'); \
   [print(ds, len(load_prepared(cfg, ds)[2])) for ds in cfg['datasets']]"
   ```
4. 两台机器都要做，且 `git pull` 后各自重新 `prepare`
