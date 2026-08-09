# Q4 多区域算—储—电协同优化

## 模型思路

两阶段协同（主目标：最小化运行成本；碳/峰值/利用率等用 ε-约束或情景体现）：

1. **阶段一（算力调度）**：在电价、碳强度、时延、等待、迁移、峰值代理代价下，对任务做动态贪心调度（策略：`joint` / `local_first` / `lowest_price` / `lowest_carbon`）。
2. **阶段二（储—电 LP）**：固定任务形成的 AI IT 负荷后，各区域独立求解储能充放电、购售电与新能源分配，目标为最小化  
   `购电成本 − 售电收益`，并可施加碳预算、峰值净购电上限、新能源利用率下限。  
   能量平衡采用附件口径：  
   `GridPurchase + AvailableRE + Discharge = TotalLoad + Charge + GridSell + Curtailment`。
3. **情景比较**：
   - 碳约束：基准碳的 100%/90%/80%/70%（收紧时重调度并偏向低碳策略）
   - 电价机制：峰谷拉大 / 平价 / 碳—电联动价
   - 新能源波动：可用出力 ×0.8 / ×1.2
   - 峰值约束：净购电峰值压至基准的 90%

### 新能源口径说明

附件中 `AvailableRenewable_MW` 在六个区域逐时完全相同，若按区域独立使用会六倍重复计算系统新能源。本实现采用各区域基准**可消纳上界**  
`UsedRenewable + RenewableCharge + GridSell`，且不超过附件 `AvailableRenewable`。

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
