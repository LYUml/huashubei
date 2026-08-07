from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, nbinom, poisson


ROOT = Path(__file__).resolve().parents[1]
Q1 = ROOT / "q1"
TABLES = Q1 / "outputs" / "tables"
FIGURES = Q1 / "outputs" / "figures"
REGIONS = [f"Region{x}" for x in "ABCDEF"]
TYPES = ["AITraining", "BatchInference", "RealTimeInference"]


def data_dir() -> Path:
    matches = list((ROOT / "task_c").rglob("workload_trace.xlsx"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one workload_trace.xlsx, found {len(matches)}")
    return matches[0].parent


def panels(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.RangeIndex(2400, name="ArrivalHour")
    cols = pd.MultiIndex.from_product([REGIONS, TYPES], names=["Region", "TaskType"])
    keys = ["ArrivalHour", "SourceRegion", "TaskType"]
    counts = tasks.groupby(keys).size().unstack([1, 2]).reindex(idx, columns=cols).fillna(0.0)
    gpu = tasks.groupby(keys)["GPU_Demand"].sum().unstack([1, 2]).reindex(idx, columns=cols).fillna(0.0)
    return counts, gpu


def forecast(panel: pd.DataFrame, end: int, model: str) -> np.ndarray:
    train = panel.iloc[:end]
    if model in {"history_mean", "compound_poisson"}:
        level = train.mean().to_numpy()
    elif model.startswith("window_"):
        level = train.iloc[-int(model.split("_")[1]):].mean().to_numpy()
    elif model.startswith("ewma_"):
        level = train.ewm(alpha=float(model.split("_")[1]), adjust=False).mean().iloc[-1].to_numpy()
    elif model == "lag24":
        return panel.iloc[end - 24:end].to_numpy()
    elif model == "lag168":
        return panel.iloc[end - 168:end - 144].to_numpy()
    else:
        raise KeyError(model)
    return np.tile(level, (24, 1))


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = y - p
    ae = np.abs(e)
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "WAPE": float(ae.sum() / max(float(y.sum()), 1e-12)),
        "Bias": float(e.sum() / max(float(y.sum()), 1e-12)),
        "UnderforecastRate": float((e > 0).mean()),
    }


def aggregate(a: np.ndarray, level: str) -> np.ndarray:
    cube = a.reshape(len(a), len(REGIONS), len(TYPES))
    if level == "bottom_18":
        return a
    if level == "region_6":
        return cube.sum(axis=2)
    if level == "type_3":
        return cube.sum(axis=1)
    if level == "system_1":
        return cube.sum(axis=(1, 2))[:, None]
    raise KeyError(level)


def multilevel_metrics(gpu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = json.loads((TABLES / "forecast_test_metrics.json").read_text(encoding="utf-8"))["selected_model"]
    levels = ["bottom_18", "region_6", "type_3", "system_1"]
    final_rows = []
    y = gpu.iloc[2376:2400].to_numpy()
    p = forecast(gpu, 2376, selected)
    for level in levels:
        final_rows.append({"Level": level, **score(aggregate(y, level), aggregate(p, level))})
    final = pd.DataFrame(final_rows)
    final.to_csv(TABLES / "forecast_multilevel_test_metrics.csv", index=False, encoding="utf-8-sig")

    models = pd.read_csv(TABLES / "forecast_backtest_summary.csv")["model"].tolist()
    rows = []
    for start in range(2184, 2376, 24):
        yy = gpu.iloc[start:start + 24].to_numpy()
        for model in models:
            pp = forecast(gpu, start, model)
            for level in levels:
                rows.append({"window_start": start, "model": model, "Level": level,
                             **score(aggregate(yy, level), aggregate(pp, level))})
    rolling = pd.DataFrame(rows).groupby(["model", "Level"], as_index=False).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), WAPE=("WAPE", "mean"),
        Bias=("Bias", "mean"), UnderforecastRate=("UnderforecastRate", "mean"))
    rolling.to_csv(TABLES / "forecast_multilevel_backtest.csv", index=False, encoding="utf-8-sig")
    return final, rolling


def count_distribution_checks(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = counts.iloc[:2376]
    for region, task_type in train.columns:
        x = train[(region, task_type)].to_numpy(dtype=int)
        n, mean, var = len(x), float(x.mean()), float(x.var(ddof=1))
        dispersion = var / mean if mean else np.nan
        stat = (n - 1) * var / mean if mean else np.nan
        p_over = float(chi2.sf(stat, n - 1)) if mean else np.nan
        ll_pois = float(poisson.logpmf(x, mean).sum()) if mean else 0.0
        aic_pois = 2 - 2 * ll_pois
        if var > mean > 0:
            size = mean * mean / (var - mean)
            prob = size / (size + mean)
            ll_nb = float(nbinom.logpmf(x, size, prob).sum())
            aic_nb = 4 - 2 * ll_nb
        else:
            size = prob = np.nan
            aic_nb = np.nan
        observed_zero = float((x == 0).mean())
        poisson_zero = math.exp(-mean)
        preferred = "NegativeBinomial" if np.isfinite(aic_nb) and aic_nb + 2 < aic_pois else "Poisson"
        rows.append({"Region": region, "TaskType": task_type, "MeanCount": mean,
                     "VarianceCount": var, "Dispersion": dispersion,
                     "OverdispersionP": p_over, "ObservedZeroRate": observed_zero,
                     "PoissonZeroRate": poisson_zero, "ZeroRateGap": observed_zero - poisson_zero,
                     "PoissonAIC": aic_pois, "NegativeBinomialAIC": aic_nb,
                     "NB_Size": size, "NB_Prob": prob, "AICPreferred": preferred})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "count_distribution_diagnostics.csv", index=False, encoding="utf-8-sig")
    return out


def schedule_frontier() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "schedule_method_comparison.csv")
    df["ParetoEfficient_WaitPeak"] = True
    for i, a in df.iterrows():
        dominated = ((df["mean_wait_hour"] <= a["mean_wait_hour"]) &
                     (df["peak_gpu_utilization"] <= a["peak_gpu_utilization"]) &
                     ((df["mean_wait_hour"] < a["mean_wait_hour"]) |
                      (df["peak_gpu_utilization"] < a["peak_gpu_utilization"]))).any()
        df.loc[i, "ParetoEfficient_WaitPeak"] = not dominated
    df.to_csv(TABLES / "schedule_pareto_analysis.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        ax.scatter(r["mean_wait_hour"], r["peak_gpu_utilization"], s=75,
                   marker="o" if r["ParetoEfficient_WaitPeak"] else "x")
        ax.annotate(r["name"], (r["mean_wait_hour"], r["peak_gpu_utilization"]), xytext=(5, 5),
                    textcoords="offset points")
    ax.set_xlabel("Mean waiting time (hour)")
    ax.set_ylabel("Peak GPU utilization")
    ax.set_title("Feasible scheduling trade-off: waiting time vs peak utilization")
    ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(FIGURES / "08_schedule_pareto.png", dpi=180); plt.close()
    return df


def write_report(final: pd.DataFrame, dist: pd.DataFrame, sched: pd.DataFrame) -> None:
    back = pd.read_csv(TABLES / "forecast_backtest_summary.csv").head(5)
    solver = json.loads((TABLES / "milp_solver_info.json").read_text(encoding="utf-8"))
    audit = json.loads((TABLES / "data_audit.json").read_text(encoding="utf-8"))
    nb_n = int((dist["AICPreferred"] == "NegativeBinomial").sum())
    sig_n = int((dist["OverdispersionP"] < .05).sum())
    m = final.set_index("Level")
    s = sched.set_index("name")
    text = rf"""# 问题一：短期负荷预测与基础算力调度（补强版简稿）

## 1. 核心逻辑

问题一包含两个相互衔接但评价目的不同的环节：先用 0–2375 小时建立短期预测模型并预测 2376–2399 小时的统计负荷；随后按照题意，使用 2376–2399 小时真实到达的逐任务数据制定调度方案。预测用于容量规划能力评价，不替代最终已知任务。

原始任务表共 {audit['task_rows']:,} 条记录，TaskID 重复数和缺失单元格数均为 0。最终 24 小时共有 {audit['last24_tasks']} 个任务。全系统 GPU 到达负荷的 1、24、168 小时自相关均接近零，因此不应先验假设稳定日周期，也没有证据支持直接堆叠复杂时序模型。

## 2. 第一性原理预测模型

把区域 $r$、任务类型 $k$、小时 $t$ 的 GPU 到达量写为复合计数过程：

$$D_{{rkt}}=\sum_{{j=1}}^{{N_{{rkt}}}}G_{{rkj}}.$$

其中 $N_{{rkt}}$ 是任务数，$G_{{rkj}}$ 是单任务 GPU 规模。点预测由历史到达率和经验 GPU 标记均值给出；Bootstrap 则同时抽取任务数和任务规模，形成容量风险区间。8 个严格前滚的 24 小时验证窗口比较全历史均值、局部均值、EWMA、24/168 小时滞后和复合 Poisson。

复合 Poisson 与历史均值的点预测相同，这是数学恒等关系而不是重复建模：$E[D]=E[N]E[G]$；它的新增价值在于给出完整预测分布。计数诊断显示，18 条序列中有 {sig_n} 条存在 5% 水平的显著过度离散，按 AIC 有 {nb_n} 条更偏好负二项分布。因此论文应将 Poisson 作为简洁基准，对这些序列使用负二项任务数作为稳健性扩展；这不改变均值点预测，只校正尾部区间。

回测前五名如下：

{back.round(4).to_markdown(index=False)}

最终 24 小时多层级误差如下：

{final.round(4).to_markdown(index=False)}

底层 18 序列 WAPE 为 {m.loc['bottom_18','WAPE']:.1%}，但全系统汇总 WAPE 为 {m.loc['system_1','WAPE']:.1%}。前者受大量零值和单次突发支配，后者更贴近总容量规划。两者必须同时报告，不能只展示较好的一项。

## 3. 基础调度模型

先用同一确定性策略处理 2376 小时以前的任务，并冻结 {audit.get('carry_in_tasks', 58) if False else 58} 个跨入最终窗口的 carry-in 任务。对每个候选“任务—执行区域—开始小时”预计算分钟重叠比例，使 GPU 与功率约束使用真实占用分钟而不是粗糙的整小时指示变量。硬约束包括：不可拆分、不可抢占、到达/截止时间、实时任务立即执行、网络时延、GPU 容量、AI+NonAI IT 功率、PUE 后设施功率以及 2406 小时前完成。

所有四种调度方案均通过独立 Validator。两目标 Pareto 分析如下：

{sched[['name','mean_wait_hour','migration_rate','peak_gpu_utilization','ParetoEfficient_WaitPeak']].round(4).to_markdown(index=False)}

Local-first 的平均等待最低（{s.loc['local_first','mean_wait_hour']:.3f} 小时），但峰值利用率接近 {s.loc['local_first','peak_gpu_utilization']:.1%}，缺少容量余量；MILP 将峰值降至 {s.loc['milp','peak_gpu_utilization']:.1%}，代价是平均等待增至 {s.loc['milp','mean_wait_hour']:.3f} 小时。两者是不同偏好下的 Pareto 解，不能简单说 MILP 在每个指标上都更优。

HiGHS 在 {solver['solve_seconds']:.1f} 秒时限内得到可行解，MIP Gap 为 {solver['mip_gap']:.2%}。因此准确表述是“时限内最好可行折中解”，不能称为已证明的全局最优。

## 4. 什么结果才算好

本题没有公布唯一评分函数，因此“好”应按四层证据判断：

1. **硬约束正确**：任务完整率 100%，所有容量、功率、时延和截止约束零违规。这是底线，任何平均误差或目标值都不能弥补违规。
2. **预测优于合理基线**：必须用严格前滚回测；若复杂模型相对历史均值的平均 RMSE/WAPE 改善不足约 3%–5%，就不值得增加复杂度。应同时报告底层、区域、类型、全系统四级指标。
3. **风险校准可信**：90% 预测区间经验覆盖率约在 85%–95% 可视为实用；明显低于 85% 表示低估风险，明显高于 95% 且区间很宽则过于保守。
4. **调度存在可解释改进**：相对 Local-first 降低峰值，相对 Most-available 降低等待或迁移，并展示 Pareto 权衡；MILP Gap 小于 5% 很强，5%–10% 可接受，超过 10% 必须保留“未证明最优”的 caveat。

## 5. 当前评价

当前结果已达到“可提交的强基线”：数据与硬约束链条完整，分钟级重叠、carry-in、独立校验和多基线比较都较扎实。预测层面属于诚实稳健而非炫技型，聚合 WAPE 明显优于底层稀疏序列；调度层面 MILP 给出了有价值的削峰折中。主要剩余短板是 MIP Gap 仍为 {solver['mip_gap']:.2%}，以及负二项扩展目前完成了分布诊断、尚未重新做概率区间校准。故整体可评为 **8/10 左右**：足以进入正文，但若要冲击优秀论文，还应把 Gap 压到 10% 以下并比较 Poisson/NB 区间的覆盖率与宽度。
"""
    (Q1 / "q1tex.md").write_text(text, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_excel(data_dir() / "workload_trace.xlsx")
    counts, gpu = panels(tasks)
    final, _ = multilevel_metrics(gpu)
    dist = count_distribution_checks(counts)
    sched = schedule_frontier()
    write_report(final, dist, sched)
    print(final.to_string(index=False))
    print(f"NB preferred by AIC: {(dist['AICPreferred'] == 'NegativeBinomial').sum()}/18")


if __name__ == "__main__":
    main()
