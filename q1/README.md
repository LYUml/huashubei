# Q1 运行说明

## 一键复现

在项目根目录运行：

```powershell
& '.\q1\.venv\Scripts\python.exe' '.\q1\run_q1.py'
```

在已有完整求解结果上运行补强分析（无需重新运行 180 秒 MILP）：

```powershell
& '.\q1\.venv\Scripts\python.exe' '.\q1\strengthen_q1.py'
```

该脚本生成多层级预测评价、Poisson/负二项计数诊断、调度 Pareto 图，并重写无乱码的 `q1tex.md`。

完整优化顺序为：

```powershell
& '.\q1\.venv\Scripts\python.exe' '.\q1\run_q1.py'
& '.\q1\.venv\Scripts\python.exe' '.\q1\optimize_q1.py'
& '.\q1\.venv\Scripts\python.exe' '.\q1\strengthen_q1.py'
```

`optimize_q1.py` 使用原生 HiGHS 接口实现真实 MIP start、wait-first 字典序和 ε 约束实验。主推方案由 `recommended_schedule.json` 指定，原加权 MILP 仅保留为极限削峰对照。

ε 约束点如需完整收敛（论文 Pareto 前沿），用 `rerun_epsilon.py` 以链式 MIP start + 300 秒/点重跑 ε 系列（约 20 分钟，ε 递增时前一解作为后一场景热启动，保证峰值单调非增）；随后运行 `plot_pareto.py` 重绘 `09_full_pareto.png`，最后重跑 `strengthen_q1.py` 使正文数字与推荐方案自动同步。

如果需要重新创建环境：

```powershell
python -m venv .\q1\.venv
& '.\q1\.venv\Scripts\python.exe' -m pip install -r '.\q1\requirements.txt'
```

## 主要文件

- `run_q1.py`：完整的数据审计、统计、预测、carry-in、贪心、MILP、验证与绘图流水线；
- `q1flow.md`：整体技术路径；
- `q1tex.md`：根据实际运行结果生成的简版正文；
- `outputs/tables/`：预测、调度和独立校验结果；
- `outputs/figures/`：统计、预测、甘特图与GPU利用率图。

## 求解说明

- MILP求解器为开源HiGHS；
- 默认时间上限180秒，目标MIP Gap为1%；
- 若达到时间上限但已有可行解，脚本保留最好可行解并记录实际MIP Gap；
- 所有硬约束由独立Validator重新计算。
