# 华数杯 C 题：面向算电协同的多目标调度优化

多区域 AI 数据中心的 **算力—储能—电力** 协同优化：问题一做 GPU 需求预测与基础调度，问题二做碳感知多目标任务调度，问题三做区域储能多目标优化，问题四做算—储—电两阶段联合优化。

## 仓库结构

```text
huashubei/
├── README.md                 # 本说明
├── .gitignore
├── data/                     # 官方附件 Excel（各问共用权威数据源）
├── material/                 # 建模笔记、答疑口径、文献综述
├── overview/                 # 数据总览与流水线示意
├── paper/                    # 竞赛论文（主文 + 图）
│   ├── main.tex              # 正文（XeLaTeX）
│   ├── figures/              # 论文插图（由各问结果汇总）
│   └── archive/              # 历史草稿（如 2.tex）
├── q1/                       # 问题一：预测 + 基础算力调度
├── q2/                       # 问题二：NSGA-II 碳感知调度
├── q3/                       # 问题三：区域储能多目标 LP
├── q4/                       # 问题四：算—储—电两阶段协同
├── task_c/                   # 赛题 PDF 与附件原始材料
└── output/                   # 其它临时/评审输出
```

各问目录尽量 **自包含**（代码、结果、图、局部 README）；根目录 `data/` 为统一附件数据。`q3/data/` 是第三问自带的副本，便于单独复现。

## 环境依赖（概览）

| 模块 | 主要依赖 |
|------|----------|
| 共用 | Python ≥ 3.10，`pandas`，`numpy`，`openpyxl` |
| Q1 | `scipy` / HiGHS（MILP） |
| Q2 | `pymoo`，`matplotlib`，`scikit-learn` |
| Q3 | `scipy.optimize.linprog`（见 `q3/requirements.txt`） |
| Q4 | HiGHS / `scipy`（见 `q4/requirements.txt`） |
| 论文 | XeLaTeX + `ctex` |

建议为各问单独建虚拟环境（`q1/.venv`、`q4/.venv` 等），避免依赖冲突。

## 快速复现

以下命令均以仓库根目录为当前目录。

### 问题一

详见 [`q1/README.md`](q1/README.md)。

```bash
python3 -m venv q1/.venv
q1/.venv/bin/pip install -r q1/requirements.txt
q1/.venv/bin/python q1/run_q1.py
q1/.venv/bin/python q1/optimize_q1.py
q1/.venv/bin/python q1/strengthen_q1.py
```

### 问题二

详见 [`q2/python_files_README.md`](q2/python_files_README.md)。

```bash
cd q2
python3 q2_data_prep.py      # 默认读 ../data
python3 q2_nsga2_v2.py
python3 q2_nsga2_visualize.py
```

### 问题三

详见 [`q3/README.md`](q3/README.md)。

```bash
python3 -m venv q3/.venv
q3/.venv/bin/pip install -r q3/requirements.txt
q3/.venv/bin/python q3/src/q3_optimize.py
MPLCONFIGDIR=/tmp/mpl q3/.venv/bin/python q3/src/q3_visualize.py
```

可用 `--data-dir data` 指向根目录附件，而不用 `q3/data` 副本。

### 问题四

详见 [`q4/README.md`](q4/README.md)。

```bash
python3 -m venv q4/.venv
q4/.venv/bin/pip install -r q4/requirements.txt
q4/.venv/bin/python q4/run_q4.py --fast   # 冒烟
q4/.venv/bin/python q4/run_q4.py          # 全量
```

## 论文编译

```bash
cd paper
xelatex main.tex
xelatex main.tex
```

插图统一放在 `paper/figures/`，正文已设置 `\graphicspath{{figures/}}`。

当前仍缺两张图（需从本地补齐后放入 `paper/figures/`）：

- `task_type_resource_share.png`（问题一任务类型资源占比）
- `nsga.png`（问题二 NSGA-II 流程示意）

历史草稿见 `paper/archive/`。

## 数据说明

根目录 `data/` 应包含赛题附件 Excel，例如：

- `workload_trace.xlsx`
- `region_time_data.xlsx`
- `GPU_information.xlsx`
- `storage_information.xlsx`
- `network_latency.xlsx`
- `power_mapping.xlsx`

`task_c/` 中保留赛题 PDF 与附件原文；建模约定与答疑见 `material/`。

## 结果与图的位置

| 问 | 结果 | 图 |
|----|------|----|
| Q1 | `q1/outputs/` | `q1/outputs/figures/` → 已汇总到 `paper/figures/` |
| Q2 | `q2/output_nsga2/` 等 | `q2/output_nsga2/fig*.png` → 论文用到的已汇总 |
| Q3 | `q3/results/` | `q3/figures/` → 已汇总到 `paper/figures/` |
| Q4 | `q4/outputs/` | `q4/figures/`、`q4/outputs/figures/` → 已汇总 |

重新出图后，请把需要入文的 PNG 复制到 `paper/figures/` 并保持文件名与 `main.tex` 一致。

## 协作约定

- 各问代码改动优先落在对应 `q1/`–`q4/` 目录；论文改动落在 `paper/`。
- 不要提交 `__pycache__`、虚拟环境、本机绝对路径脚本。
- 大体积中间结果（如完整调度 CSV）已随问目录提交以便复现；若需精简仓库，可只保留 `*metrics*` / 汇总表并在 README 中说明重跑命令。
