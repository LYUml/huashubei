# Q4 多区域算—储—电协同优化

## 模型思路

两阶段协同（主目标：最小化运行成本；碳/峰值/利用率等用 ε-约束或情景体现）：

1. **阶段一（算力调度）**：在电价、碳强度、时延、等待、迁移、峰值代理代价下，对任务做动态贪心调度（策略：`joint` / `local_first` / `lowest_price` / `lowest_carbon`）。
2. **阶段二（储—电 LP）**：固定任务形成的 AI IT 负荷后，各区域独立求解储能充放电、购售电与新能源分配，目标为最小化  
   `购电成本 − 售电收益`，并可施加碳预算、峰值净购电上限、新能源利用率下限。  
   能量平衡采用附件口径：  
   `GridPurchase + AvailableRE + Discharge = TotalLoad + Charge + GridSell + Curtailment`。
3. **情景比较**：
   - 碳约束：基准碳的 100%/90%/80%/70%（**硬 ε-约束**）。阶段二按区域碳下界分配预算且不可静默丢弃；若当前排程下 \(E_{\min}>\epsilon\)，则提高阶段一碳权重重调度。若 ε 低于 NonAI 碳下界，则判定 **infeasible**（不再假装满足）。
   - 电价机制：峰谷拉大 / 平价 / 碳—电联动价
   - 新能源波动：可用出力 ×0.8 / ×1.2
   - 峰值约束：净购电峰值压至基准的 90%

### 新能源口径说明

附件中 `AvailableRenewable_MW` 在六个区域逐时完全相同，若按区域独立写入能量平衡会六倍重复计算系统新能源。本实现：

- **调度/LP 上界**：各区域基准可消纳上界 `UsedRenewable + RenewableCharge + GridSell`（且不超过附件 `AvailableRenewable`）。
- **利用率指标（headline）**：`消纳量 / 附件 AvailableRenewable`（六区加总）。消纳量 = 可消纳上界 − LP 弃光。  
  此前若用「可消纳上界」做分母，会在弃光≈0 时得到接近 **100%**，这是口径自洽而非真实利用率，已纠正。
- **诊断指标**：`renewable_utilization_of_deliverable` 表示 LP 对可消纳上界的填充率（可接近 100%）。

注意：附件 AvailableRenewable 六区重复，按六区加总作分母时利用率会系统性偏低（约三分之一量级）；这是数据口径选择，不是把弃光算成零。

### 碳约束口径说明

- 正式模型：\(\min C\ \mathrm{s.t.}\ E\le \epsilon\)，\(\epsilon\in\{1.0,0.9,0.8,0.7\}E_0\)。
- NonAI 负荷对应的系统碳下界约 \(1.45\times 10^6\) tCO2；因此 **carbon_80 / carbon_70 相对 \(E_0\) 通常物理不可行**，结果中会标 `carbon_feasible=false` 并给出原因，而不是输出一个假的“已满足”排放。
- carbon_90 接近该下界，需通过提高阶段一碳权重、把 AI 负荷迁向低碳区时才可能可行。

## 运行方式

在仓库根目录：

```powershell
python -m venv .\q4\.venv
.\q4\.venv\Scripts\python.exe -m pip install -r .\q4\requirements.txt
.\q4\.venv\Scripts\python.exe .\q4\run_q4.py --fast
.\q4\.venv\Scripts\python.exe .\q4\run_q4.py
```

Linux / macOS：

```bash
python3 -m venv q4/.venv
q4/.venv/bin/python -m pip install -r q4/requirements.txt
q4/.venv/bin/python q4/run_q4.py --fast    # 快速：仅 ArrivalHour>=2300 的任务
q4/.venv/bin/python q4/run_q4.py           # 全量 5 万任务（较慢）
```

常用参数：

- `--fast`：只用后期到达任务，便于冒烟测试
- `--start-hour 2000`：自定义任务子集
- `--skip-scenarios`：只跑四种基线策略

## 主要文件

| 文件 | 作用 |
|---|---|
| `run_q4.py` | 主入口 |
| `data_loader.py` | 读附件、电价机制/新能源缩放 |
| `schedule.py` | 阶段一任务调度 |
| `power_opt.py` | 阶段二区域储电 LP（HiGHS） |
| `scenarios.py` | 基线与情景编排 |
| `metrics.py` | 六类指标汇总与校验 |
| `plot_results.py` | 出图 |

## 输出

全部写入 `q4/outputs/`：

- `tables/scenario_summary.csv`：情景总表
- `tables/schedule_*.csv` / `power_*.csv` / `metrics_*.json`
- `tables/recommended_q4.json`：推荐联合方案
- `figures/01_scenario_metrics.png` 等
- `q4_report.md`：简要报告
