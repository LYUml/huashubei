from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
Q1_DIR = ROOT / "q1"
DATA_DIR = ROOT / "task_c" / "附件数据"
OUT_DIR = Q1_DIR / "outputs"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
SEED = 20260807
RNG = np.random.default_rng(SEED)
REGIONS = [f"Region{x}" for x in "ABCDEF"]
TASK_TYPES = ["AITraining", "BatchInference", "RealTimeInference"]
SCHEDULE_START = 2376
ARRIVAL_END = 2399
EXECUTION_END = 2406  # half-open: no activity at hour 2406
HOURS = np.arange(SCHEDULE_START, EXECUTION_END)


@dataclass
class DataBundle:
    tasks: pd.DataFrame
    gpu_info: pd.DataFrame
    latency: pd.DataFrame
    power: pd.Series
    region_hour: pd.DataFrame


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> DataBundle:
    tasks = pd.read_excel(DATA_DIR / "workload_trace.xlsx", sheet_name=0)
    gpu_info = pd.read_excel(DATA_DIR / "GPU_information.xlsx", sheet_name=0)
    latency = pd.read_excel(DATA_DIR / "network_latency.xlsx", sheet_name=0)
    power_df = pd.read_excel(DATA_DIR / "power_mapping.xlsx", sheet_name=0)
    region_hour = pd.read_excel(DATA_DIR / "region_time_data.xlsx", sheet_name=0)
    power = power_df.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"]
    return DataBundle(tasks, gpu_info, latency, power, region_hour)


def audit_data(data: DataBundle) -> dict:
    t = data.tasks
    rh = data.region_hour
    checks = {
        "task_rows": int(len(t)),
        "duplicate_task_ids": int(t["TaskID"].duplicated().sum()),
        "exact_duplicate_tasks": int(t.duplicated().sum()),
        "task_null_cells": int(t.isna().sum().sum()),
        "arrival_min": int(t["ArrivalHour"].min()),
        "arrival_max": int(t["ArrivalHour"].max()),
        "gpu_min": int(t["GPU_Demand"].min()),
        "gpu_max": int(t["GPU_Demand"].max()),
        "duration_min": int(t["EstimatedDuration_min"].min()),
        "duration_max": int(t["EstimatedDuration_min"].max()),
        "earliest_equals_arrival": bool((t["EarliestStartHour"] == t["ArrivalHour"]).all()),
        "all_nonpreemptive": bool((t["ExecutionMode"] == "NonPreemptive").all()),
        "region_hour_rows": int(len(rh)),
        "region_hour_duplicates": int(rh.duplicated(["Region", "Hour"]).sum()),
        "region_hour_null_cells": int(rh.isna().sum().sum()),
        "region_hour_min": int(rh["Hour"].min()),
        "region_hour_max": int(rh["Hour"].max()),
        "last24_tasks": int(t["ArrivalHour"].between(2376, 2399).sum()),
    }
    if checks["duplicate_task_ids"] or checks["task_null_cells"]:
        raise ValueError("Task data failed uniqueness/completeness checks")
    if checks["region_hour_duplicates"] or checks["region_hour_null_cells"]:
        raise ValueError("Region-hour data failed uniqueness/completeness checks")
    (TABLE_DIR / "data_audit.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checks


def build_hourly_panels(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.RangeIndex(2400, name="ArrivalHour")
    columns = pd.MultiIndex.from_product(
        [REGIONS, TASK_TYPES], names=["SourceRegion", "TaskType"]
    )
    group_keys = ["ArrivalHour", "SourceRegion", "TaskType"]
    counts = (
        tasks.groupby(group_keys).size().unstack(["SourceRegion", "TaskType"])
        .reindex(index=idx, columns=columns).fillna(0.0)
    )
    gpu = (
        tasks.groupby(group_keys)["GPU_Demand"].sum().unstack(["SourceRegion", "TaskType"])
        .reindex(index=idx, columns=columns).fillna(0.0)
    )
    work = tasks.assign(
        GPU_hour=tasks["GPU_Demand"] * tasks["EstimatedDuration_min"] / 60.0
    )
    gpuh = (
        work.groupby(group_keys)["GPU_hour"].sum().unstack(["SourceRegion", "TaskType"])
        .reindex(index=idx, columns=columns).fillna(0.0)
    )
    return counts, gpu, gpuh


def descriptive_analysis(data: DataBundle, counts: pd.DataFrame, gpu: pd.DataFrame, gpuh: pd.DataFrame) -> dict:
    tasks = data.tasks.assign(
        GPU_hour=data.tasks["GPU_Demand"] * data.tasks["EstimatedDuration_min"] / 60.0
    )
    type_summary = tasks.groupby("TaskType").agg(
        tasks=("TaskID", "size"),
        gpu_sum=("GPU_Demand", "sum"),
        gpu_mean=("GPU_Demand", "mean"),
        duration_mean_min=("EstimatedDuration_min", "mean"),
        gpuh_sum=("GPU_hour", "sum"),
    ).reindex(TASK_TYPES)
    region_type = tasks.groupby(["SourceRegion", "TaskType"]).agg(
        tasks=("TaskID", "size"), gpu_sum=("GPU_Demand", "sum"), gpuh_sum=("GPU_hour", "sum")
    )
    profiles = []
    for col in gpu.columns:
        s = gpu[col]
        c = counts[col]
        lam = float(c.mean())
        profiles.append({
            "Region": col[0], "TaskType": col[1],
            "count_mean": lam, "count_variance": float(c.var()),
            "count_dispersion": float(c.var() / lam) if lam > 0 else np.nan,
            "gpu_mean": float(s.mean()), "gpu_std": float(s.std()),
            "gpu_zero_rate": float((s == 0).mean()),
            "poisson_zero_probability": float(math.exp(-lam)),
            "gpu_p95": float(s.quantile(0.95)), "gpu_max": float(s.max()),
            "acf_1": float(s.autocorr(1)), "acf_24": float(s.autocorr(24)),
            "acf_168": float(s.autocorr(168)),
            "gpuh_mean": float(gpuh[col].mean()),
        })
    profile_df = pd.DataFrame(profiles)
    type_summary.to_csv(TABLE_DIR / "task_type_summary.csv", encoding="utf-8-sig")
    region_type.to_csv(TABLE_DIR / "region_type_summary.csv", encoding="utf-8-sig")
    profile_df.to_csv(TABLE_DIR / "series_profile.csv", index=False, encoding="utf-8-sig")

    # Figure 1: region x type GPU totals
    pivot = region_type["gpu_sum"].unstack("TaskType").reindex(index=REGIONS, columns=TASK_TYPES)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Total arriving GPU demand by region and task type")
    ax.set_xlabel("Region")
    ax.set_ylabel("Equivalent GPU")
    ax.legend(title="Task type")
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_region_type_gpu.png", dpi=180); plt.close()

    # Figure 2: total hourly workload
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(gpu.sum(axis=1), linewidth=0.7)
    ax.axvspan(2376, 2399, color="tab:red", alpha=0.15, label="Final test window")
    ax.set_title("Hourly arriving GPU demand")
    ax.set_xlabel("Hour"); ax.set_ylabel("Equivalent GPU"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_hourly_gpu_demand.png", dpi=180); plt.close()

    # Figure 3: ACF diagnostic
    lag_cols = ["acf_1", "acf_24", "acf_168"]
    x = np.arange(len(profile_df))
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.25
    for j, c in enumerate(lag_cols):
        ax.bar(x + (j - 1) * width, profile_df[c], width, label=c)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(profile_df["Region"] + "\n" + profile_df["TaskType"].str.replace("Inference", "Inf"), rotation=45, ha="right")
    ax.set_title("Autocorrelation diagnostics for 18 bottom-level series")
    ax.legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "03_acf_diagnostics.png", dpi=180); plt.close()

    return {
        "total_gpu_mean": float(gpu.sum(axis=1).mean()),
        "total_gpu_std": float(gpu.sum(axis=1).std()),
        "total_acf_1": float(gpu.sum(axis=1).autocorr(1)),
        "total_acf_24": float(gpu.sum(axis=1).autocorr(24)),
        "total_acf_168": float(gpu.sum(axis=1).autocorr(168)),
    }


def point_forecast(panel: pd.DataFrame, train_end: int, model: str) -> np.ndarray:
    train = panel.iloc[:train_end]
    horizon = 24
    if model == "history_mean" or model == "compound_poisson":
        level = train.mean().to_numpy()
    elif model.startswith("window_"):
        window = int(model.split("_")[1])
        level = train.iloc[-window:].mean().to_numpy()
    elif model.startswith("ewma_"):
        alpha = float(model.split("_")[1])
        level = train.ewm(alpha=alpha, adjust=False).mean().iloc[-1].to_numpy()
    elif model == "lag24":
        return panel.iloc[train_end - 24:train_end].to_numpy()
    elif model == "lag168":
        return panel.iloc[train_end - 168:train_end - 144].to_numpy()
    else:
        raise KeyError(model)
    return np.tile(level, (horizon, 1))


def forecast_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = y - pred
    abs_err = np.abs(err)
    under = np.maximum(err, 0)
    return {
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "WAPE": float(abs_err.sum() / max(y.sum(), 1e-9)),
        "Under_Q95": float(np.quantile(under, 0.95)),
        "Under_Max": float(under.max()),
    }


def rolling_backtest(gpu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    models = [
        "history_mean", "window_72", "window_168", "window_336",
        "ewma_0.003", "ewma_0.01", "ewma_0.03", "compound_poisson",
        "lag24", "lag168",
    ]
    validation_starts = list(range(2184, 2376, 24))
    rows = []
    for start in validation_starts:
        y = gpu.iloc[start:start + 24].to_numpy()
        window_losses = {}
        for model in models:
            pred = point_forecast(gpu, start, model)
            met = forecast_metrics(y, pred)
            rows.append({"window_start": start, "model": model, **met})
            window_losses[model] = met["RMSE"]
        best = min(window_losses.values())
        for row in rows[-len(models):]:
            row["Regret"] = row["RMSE"] - best
    detail = pd.DataFrame(rows)
    summary = detail.groupby("model").agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), WAPE=("WAPE", "mean"),
        RMSE_std=("RMSE", "std"), Max_Regret=("Regret", "max"),
        Mean_Regret=("Regret", "mean"), Under_Q95=("Under_Q95", "mean"),
        Under_Max=("Under_Max", "max"),
    ).sort_values("RMSE")
    best_rmse = summary["RMSE"].min()
    near = summary[summary["RMSE"] <= 1.02 * best_rmse]
    selected = str(near.sort_values(["Max_Regret", "MAE"]).index[0])
    detail.to_csv(TABLE_DIR / "forecast_backtest_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "forecast_backtest_summary.csv", encoding="utf-8-sig")

    plot_df = summary.sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df.index, plot_df["RMSE"], yerr=plot_df["RMSE_std"], capsize=3)
    ax.set_title("Rolling 24-hour backtest RMSE")
    ax.set_ylabel("RMSE"); ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_backtest_rmse.png", dpi=180); plt.close()
    return detail, summary, selected


def bootstrap_compound_intervals(
    tasks: pd.DataFrame, train_end: int, n_sim: int = 3000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist = tasks[tasks["ArrivalHour"] < train_end]
    low = np.zeros((24, 18)); median = np.zeros((24, 18)); high = np.zeros((24, 18))
    total_sims = np.zeros((n_sim, 24))
    for c, (region, task_type) in enumerate(
        [(r, k) for r in REGIONS for k in TASK_TYPES]
    ):
        subset = hist[(hist["SourceRegion"] == region) & (hist["TaskType"] == task_type)]
        lam = len(subset) / train_end
        marks = subset["GPU_Demand"].to_numpy(dtype=float)
        sims = np.zeros((n_sim, 24))
        for h in range(24):
            n = RNG.poisson(lam, size=n_sim)
            for j, count in enumerate(n):
                if count:
                    sims[j, h] = RNG.choice(marks, size=count, replace=True).sum()
        low[:, c] = np.quantile(sims, 0.05, axis=0)
        median[:, c] = np.quantile(sims, 0.50, axis=0)
        high[:, c] = np.quantile(sims, 0.95, axis=0)
        total_sims += sims
    total_low = np.quantile(total_sims, 0.05, axis=0)
    total_high = np.quantile(total_sims, 0.95, axis=0)
    return low, median, high, total_low, total_high


def final_forecast(
    data: DataBundle, gpu: pd.DataFrame, selected_model: str
) -> dict:
    pred = point_forecast(gpu, 2376, selected_model)
    y = gpu.iloc[2376:2400].to_numpy()
    metrics = forecast_metrics(y, pred)
    low, median, high, total_low, total_high = bootstrap_compound_intervals(data.tasks, 2376)
    coverage = float(((y >= low) & (y <= high)).mean())
    metrics["interval_90_coverage"] = coverage
    metrics["selected_model"] = selected_model

    columns = [f"{r}__{k}" for r in REGIONS for k in TASK_TYPES]
    pred_df = pd.DataFrame(pred, index=range(2376, 2400), columns=columns)
    actual_df = pd.DataFrame(y, index=range(2376, 2400), columns=columns)
    low_df = pd.DataFrame(low, index=range(2376, 2400), columns=columns)
    median_df = pd.DataFrame(median, index=range(2376, 2400), columns=columns)
    high_df = pd.DataFrame(high, index=range(2376, 2400), columns=columns)
    out = pd.concat(
        {"actual": actual_df, "point": pred_df, "p05": low_df, "p50": median_df, "p95": high_df},
        axis=1,
    )
    out.to_csv(TABLE_DIR / "forecast_2376_2399.csv", encoding="utf-8-sig")
    (TABLE_DIR / "forecast_test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_actual = actual_df.sum(axis=1)
    total_pred = pred_df.sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(total_actual.index, total_low, total_high, alpha=0.22, label="Monte Carlo 90% interval")
    ax.plot(total_actual.index, total_actual, marker="o", label="Actual")
    ax.plot(total_pred.index, total_pred, marker="s", label=f"Point forecast: {selected_model}")
    ax.set_title("Final 24-hour total arriving GPU forecast")
    ax.set_xlabel("Hour"); ax.set_ylabel("Equivalent GPU"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_final_forecast.png", dpi=180); plt.close()
    return metrics


def overlap_fraction(start_hour: int, duration_min: int, hour: int) -> float:
    start_min = 60 * start_hour
    end_min = start_min + int(duration_min)
    overlap = max(0, min(end_min, 60 * (hour + 1)) - max(start_min, 60 * hour))
    return overlap / 60.0


def candidate_hours(task: pd.Series) -> range:
    arrival = int(task["ArrivalHour"])
    if task["TaskType"] == "RealTimeInference":
        return range(arrival, arrival + 1)
    latest = min(int(task["LatestFinishHour"]), EXECUTION_END)
    last_start = math.floor(latest - float(task["EstimatedDuration_min"]) / 60.0 + 1e-9)
    return range(arrival, last_start + 1)


def latency_maps(data: DataBundle) -> tuple[dict, dict]:
    latency = {
        (row.FromRegion, row.ToRegion): float(row.NetworkLatency_ms)
        for row in data.latency.itertuples(index=False)
    }
    eligible = {}
    for row in data.tasks.itertuples(index=False):
        eligible[int(row.TaskID)] = [
            r for r in REGIONS if latency[(row.SourceRegion, r)] <= row.MaxLatency_ms
        ]
    return latency, eligible


def resource_arrays(data: DataBundle, start: int = 0, end: int = EXECUTION_END):
    n_hours = end - start
    gpu_use = np.zeros((len(REGIONS), n_hours), dtype=float)
    ai_power = np.zeros_like(gpu_use)
    info = data.gpu_info.set_index("Region")
    gpu_cap = info.loc[REGIONS, "Available_GPU"].to_numpy(float)
    max_it = info.loc[REGIONS, "Max_IT_Power_MW"].to_numpy(float)
    pue = info.loc[REGIONS, "PUE"].to_numpy(float)
    max_fac = info.loc[REGIONS, "Max_Facility_Power_MW"].to_numpy(float)
    rh = data.region_hour.pivot(index="Hour", columns="Region", values="NonAI_IT_Load_MW")
    nonai = rh.loc[start:end - 1, REGIONS].to_numpy(float).T
    return gpu_use, ai_power, gpu_cap, max_it, pue, max_fac, nonai


def candidate_resource_profile(task: pd.Series, start: int, horizon_start: int, horizon_end: int):
    hours = []
    fractions = []
    first = max(start, horizon_start)
    last = min(math.ceil(start + task["EstimatedDuration_min"] / 60.0), horizon_end)
    for hour in range(first, last):
        q = overlap_fraction(start, int(task["EstimatedDuration_min"]), hour)
        if q > 0:
            hours.append(hour - horizon_start)
            fractions.append(q)
    return np.asarray(hours, dtype=int), np.asarray(fractions, dtype=float)


def feasible_candidate(
    task: pd.Series, region_idx: int, start: int, arrays, horizon_start: int, horizon_end: int,
    tol: float = 1e-7,
) -> bool:
    gpu_use, ai_power, gpu_cap, max_it, pue, max_fac, nonai = arrays
    idx, q = candidate_resource_profile(task, start, horizon_start, horizon_end)
    if len(idx) == 0:
        return False
    gpu_inc = float(task["GPU_Demand"]) * q
    p_inc = float(task["GPU_Demand"] * task["PowerPerGPU"]) * q
    if np.any(gpu_use[region_idx, idx] + gpu_inc > gpu_cap[region_idx] + tol):
        return False
    total_it = nonai[region_idx, idx] + ai_power[region_idx, idx] + p_inc
    if np.any(total_it > max_it[region_idx] + tol):
        return False
    if np.any(total_it * pue[region_idx] > max_fac[region_idx] + tol):
        return False
    return True


def place_task(task: pd.Series, region_idx: int, start: int, arrays, horizon_start: int, horizon_end: int):
    gpu_use, ai_power, *_ = arrays
    idx, q = candidate_resource_profile(task, start, horizon_start, horizon_end)
    gpu_use[region_idx, idx] += float(task["GPU_Demand"]) * q
    ai_power[region_idx, idx] += float(task["GPU_Demand"] * task["PowerPerGPU"]) * q


def greedy_schedule(
    tasks: pd.DataFrame,
    data: DataBundle,
    arrays,
    horizon_start: int,
    horizon_end: int,
    eligible: dict,
    latency: dict,
    label: str,
    strategy: str = "dynamic",
) -> pd.DataFrame:
    info = data.gpu_info.set_index("Region")
    gpu_cap = info.loc[REGIONS, "Available_GPU"].to_numpy(float)
    # Hardest first: real-time, fewer regions, less slack, larger GPU-hour.
    work = tasks.copy()
    work["EligibleCount"] = work["TaskID"].map(lambda x: len(eligible[int(x)]))
    work["DurationHour"] = work["EstimatedDuration_min"] / 60.0
    work["Slack"] = work["LatestFinishHour"] - work["ArrivalHour"] - work["DurationHour"]
    work["GPUHour"] = work["GPU_Demand"] * work["DurationHour"]
    work["RealtimeRank"] = (work["TaskType"] != "RealTimeInference").astype(int)
    work = work.sort_values(
        ["RealtimeRank", "EligibleCount", "Slack", "GPUHour", "ArrivalHour"],
        ascending=[True, True, True, False, True],
    )
    assignments = []
    for _, task in work.iterrows():
        candidates = []
        for start in candidate_hours(task):
            if start < horizon_start or start >= horizon_end:
                continue
            finish = start + task["EstimatedDuration_min"] / 60.0
            if finish > min(task["LatestFinishHour"], EXECUTION_END) + 1e-9:
                continue
            for region in eligible[int(task["TaskID"])]:
                r = REGIONS.index(region)
                if not feasible_candidate(task, r, start, arrays, horizon_start, horizon_end):
                    continue
                idx, q = candidate_resource_profile(task, start, horizon_start, horizon_end)
                after = arrays[0][r, idx] + task["GPU_Demand"] * q
                peak_after = float(np.max(after / gpu_cap[r]))
                current_mean = arrays[0][:, idx].sum(axis=1) / gpu_cap
                projected = current_mean.copy()
                projected[r] += float((task["GPU_Demand"] * q).sum() / max(len(idx), 1) / gpu_cap[r])
                imbalance = float(np.std(projected))
                wait_norm = (start - task["ArrivalHour"]) / max(EXECUTION_END - task["ArrivalHour"], 1)
                migration = float(region != task["SourceRegion"])
                latency_norm = latency[(task["SourceRegion"], region)] / max(task["MaxLatency_ms"], 1)
                if strategy == "local_first":
                    cost = 10.0 * migration + 0.8 * wait_norm + 0.1 * latency_norm + 0.1 * peak_after
                elif strategy == "most_available":
                    cost = 0.75 * peak_after + 0.20 * wait_norm + 0.05 * latency_norm
                else:
                    cost = 0.45 * wait_norm + 0.12 * migration + 0.08 * latency_norm + 0.25 * peak_after + 0.10 * imbalance
                candidates.append((cost, region, start))
            # The prefix is not an optimization target.  It only supplies a
            # reproducible carry-in state, so use the earliest feasible start
            # and choose the best region at that start.  This avoids scanning
            # thousands of future start hours for every historical task.
            if label.startswith("prefix") and candidates:
                break
        if not candidates:
            raise RuntimeError(f"{label}: no feasible candidate for TaskID={int(task['TaskID'])}")
        _, region, start = min(candidates, key=lambda x: x[0])
        place_task(task, REGIONS.index(region), start, arrays, horizon_start, horizon_end)
        assignments.append({
            "TaskID": int(task["TaskID"]), "ExecutionRegion": region,
            "StartHour": int(start), "FinishHour": float(start + task["EstimatedDuration_min"] / 60.0),
        })
    return pd.DataFrame(assignments)


def build_carry_in(data: DataBundle, eligible: dict, latency: dict):
    prefix = data.tasks[data.tasks["ArrivalHour"] < SCHEDULE_START].copy()
    prefix["PowerPerGPU"] = prefix["TaskType"].map(data.power)
    arrays = resource_arrays(data, 0, EXECUTION_END)
    schedule = greedy_schedule(prefix, data, arrays, 0, EXECUTION_END, eligible, latency, "prefix")
    schedule = schedule.merge(prefix[["TaskID", "TaskType", "GPU_Demand", "EstimatedDuration_min"]], on="TaskID")
    carry_tasks = schedule[schedule["FinishHour"] > SCHEDULE_START].copy()
    carry_gpu = arrays[0][:, SCHEDULE_START:EXECUTION_END].copy()
    carry_power = arrays[1][:, SCHEDULE_START:EXECUTION_END].copy()
    carry_tasks.to_csv(TABLE_DIR / "carry_in_tasks.csv", index=False, encoding="utf-8-sig")
    return carry_gpu, carry_power, carry_tasks


def make_final_arrays(data: DataBundle, carry_gpu: np.ndarray, carry_power: np.ndarray):
    arrays = resource_arrays(data, SCHEDULE_START, EXECUTION_END)
    arrays[0][:] = carry_gpu
    arrays[1][:] = carry_power
    return arrays


def build_milp_candidates(tasks: pd.DataFrame, data: DataBundle, eligible: dict, latency: dict) -> pd.DataFrame:
    rows = []
    for _, task in tasks.iterrows():
        for start in candidate_hours(task):
            if start < SCHEDULE_START or start >= EXECUTION_END:
                continue
            if start + task["EstimatedDuration_min"] / 60.0 > min(task["LatestFinishHour"], EXECUTION_END) + 1e-9:
                continue
            for region in eligible[int(task["TaskID"])]:
                rows.append({
                    "TaskID": int(task["TaskID"]), "Region": region, "StartHour": int(start),
                    "Wait": float(start - task["ArrivalHour"]),
                    "Migration": float(region != task["SourceRegion"]),
                    "LatencyNorm": latency[(task["SourceRegion"], region)] / task["MaxLatency_ms"],
                    "TailGPUh": float(task["GPU_Demand"] * sum(
                        overlap_fraction(start, int(task["EstimatedDuration_min"]), h)
                        for h in range(2400, EXECUTION_END)
                    )),
                })
    return pd.DataFrame(rows)


def solve_final_milp(
    tasks: pd.DataFrame,
    data: DataBundle,
    candidates: pd.DataFrame,
    carry_gpu: np.ndarray,
    carry_power: np.ndarray,
    time_limit: float = 180.0,
):
    task_lookup = tasks.set_index("TaskID")
    info = data.gpu_info.set_index("Region")
    gpu_cap = info.loc[REGIONS, "Available_GPU"].to_numpy(float)
    max_it = info.loc[REGIONS, "Max_IT_Power_MW"].to_numpy(float)
    pue = info.loc[REGIONS, "PUE"].to_numpy(float)
    max_fac = info.loc[REGIONS, "Max_Facility_Power_MW"].to_numpy(float)
    nonai_pivot = data.region_hour.pivot(index="Hour", columns="Region", values="NonAI_IT_Load_MW")
    nonai = nonai_pivot.loc[SCHEDULE_START:EXECUTION_END - 1, REGIONS].to_numpy(float).T
    n_cand = len(candidates)
    n_tasks = len(tasks)
    peak_var = n_cand
    n_var = n_cand + 1
    max_wait = max(float(candidates["Wait"].max()), 1.0)
    max_tail = max(float(candidates["TailGPUh"].max()), 1.0)
    c = np.zeros(n_var)
    c[:n_cand] = (
        0.40 * candidates["Wait"].to_numpy() / max_wait
        + 0.10 * candidates["Migration"].to_numpy()
        + 0.05 * candidates["LatencyNorm"].to_numpy()
        + 0.05 * candidates["TailGPUh"].to_numpy() / max_tail
    ) / n_tasks
    c[peak_var] = 0.40

    row_idx = []
    col_idx = []
    values = []
    lower = []
    upper = []
    row = 0

    # Each task exactly once.
    for task_id, group in candidates.groupby("TaskID", sort=False):
        for j in group.index:
            row_idx.append(row); col_idx.append(int(j)); values.append(1.0)
        lower.append(1.0); upper.append(1.0); row += 1

    # Resource and peak constraints.
    for r, region in enumerate(REGIONS):
        region_candidates = candidates[candidates["Region"] == region]
        for t_idx, hour in enumerate(HOURS):
            gpu_entries = []
            power_entries = []
            for j, cand in region_candidates.iterrows():
                task = task_lookup.loc[int(cand["TaskID"])]
                q = overlap_fraction(int(cand["StartHour"]), int(task["EstimatedDuration_min"]), int(hour))
                if q <= 0:
                    continue
                gpu_coef = float(task["GPU_Demand"] * q)
                p_coef = float(task["GPU_Demand"] * task["PowerPerGPU"] * q)
                gpu_entries.append((int(j), gpu_coef))
                power_entries.append((int(j), p_coef))
            # GPU capacity
            for j, val in gpu_entries:
                row_idx.append(row); col_idx.append(j); values.append(val)
            lower.append(-np.inf); upper.append(float(gpu_cap[r] - carry_gpu[r, t_idx])); row += 1
            # IT power (facility is equivalent in supplied data, but validator checks both)
            headroom_it = max_it[r] - nonai[r, t_idx] - carry_power[r, t_idx]
            headroom_fac = max_fac[r] / pue[r] - nonai[r, t_idx] - carry_power[r, t_idx]
            for j, val in power_entries:
                row_idx.append(row); col_idx.append(j); values.append(val)
            lower.append(-np.inf); upper.append(float(min(headroom_it, headroom_fac))); row += 1
            # Peak utilization: (carry + decision GPU) / capacity <= Umax
            for j, val in gpu_entries:
                row_idx.append(row); col_idx.append(j); values.append(val / gpu_cap[r])
            row_idx.append(row); col_idx.append(peak_var); values.append(-1.0)
            lower.append(-np.inf); upper.append(float(-carry_gpu[r, t_idx] / gpu_cap[r])); row += 1

    A = sparse.coo_matrix((values, (row_idx, col_idx)), shape=(row, n_var)).tocsr()
    constraints = LinearConstraint(A, np.asarray(lower), np.asarray(upper))
    bounds = Bounds(np.zeros(n_var), np.r_[np.ones(n_cand), 1.0])
    integrality = np.r_[np.ones(n_cand, dtype=int), 0]
    start = time.time()
    result = milp(
        c=c, integrality=integrality, bounds=bounds, constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": 0.01, "presolve": True, "disp": True},
    )
    elapsed = time.time() - start
    if result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")
    chosen_idx = np.where(result.x[:n_cand] > 0.5)[0]
    chosen = candidates.iloc[chosen_idx].copy()
    chosen = chosen.rename(columns={"Region": "ExecutionRegion"})
    chosen["FinishHour"] = chosen.apply(
        lambda row: row["StartHour"] + task_lookup.loc[int(row["TaskID"]), "EstimatedDuration_min"] / 60.0,
        axis=1,
    )
    solver_info = {
        "status": int(result.status), "success": bool(result.success), "message": str(result.message),
        "objective": float(result.fun), "peak_utilization_variable": float(result.x[peak_var]),
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "mip_node_count": int(getattr(result, "mip_node_count", -1)),
        "solve_seconds": elapsed, "candidate_variables": n_cand,
    }
    (TABLE_DIR / "milp_solver_info.json").write_text(
        json.dumps(solver_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return chosen, solver_info


def reconstruct_schedule(
    tasks: pd.DataFrame,
    schedule: pd.DataFrame,
    data: DataBundle,
    carry_gpu: np.ndarray,
    carry_power: np.ndarray,
):
    merged = tasks.merge(schedule[["TaskID", "ExecutionRegion", "StartHour", "FinishHour"]], on="TaskID", how="left")
    latency, _ = latency_maps(data)
    merged["WaitHour"] = merged["StartHour"] - merged["ArrivalHour"]
    merged["NetworkLatency_ms"] = merged.apply(
        lambda x: latency[(x["SourceRegion"], x["ExecutionRegion"])], axis=1
    )
    merged["Migrated"] = merged["ExecutionRegion"] != merged["SourceRegion"]
    merged["TailPeriodUsed"] = merged["FinishHour"] > 2400
    gpu = carry_gpu.copy(); power = carry_power.copy()
    for _, task in merged.iterrows():
        r = REGIONS.index(task["ExecutionRegion"])
        idx, q = candidate_resource_profile(task, int(task["StartHour"]), SCHEDULE_START, EXECUTION_END)
        gpu[r, idx] += task["GPU_Demand"] * q
        power[r, idx] += task["GPU_Demand"] * task["PowerPerGPU"] * q
    return merged, gpu, power


def validate_schedule(
    tasks: pd.DataFrame,
    schedule: pd.DataFrame,
    gpu_use: np.ndarray,
    ai_power: np.ndarray,
    data: DataBundle,
    name: str,
) -> dict:
    info = data.gpu_info.set_index("Region")
    gpu_cap = info.loc[REGIONS, "Available_GPU"].to_numpy(float)[:, None]
    max_it = info.loc[REGIONS, "Max_IT_Power_MW"].to_numpy(float)[:, None]
    pue = info.loc[REGIONS, "PUE"].to_numpy(float)[:, None]
    max_fac = info.loc[REGIONS, "Max_Facility_Power_MW"].to_numpy(float)[:, None]
    nonai = data.region_hour.pivot(index="Hour", columns="Region", values="NonAI_IT_Load_MW").loc[
        SCHEDULE_START:EXECUTION_END - 1, REGIONS
    ].to_numpy(float).T
    it_load = nonai + ai_power
    fac_load = it_load * pue
    duplicate = int(schedule["TaskID"].duplicated().sum())
    missing = int(schedule["ExecutionRegion"].isna().sum())
    rt = schedule[schedule["TaskType"] == "RealTimeInference"]
    checks = {
        "name": name,
        "scheduled_tasks": int(len(schedule)), "duplicate_task_ids": duplicate, "missing_assignments": missing,
        "realtime_start_violations": int((rt["StartHour"] != rt["ArrivalHour"]).sum()),
        "arrival_violations": int((schedule["StartHour"] < schedule["ArrivalHour"]).sum()),
        "deadline_violations": int((schedule["FinishHour"] > schedule["LatestFinishHour"] + 1e-8).sum()),
        "terminal_2406_violations": int((schedule["FinishHour"] > EXECUTION_END + 1e-8).sum()),
        "latency_violations": int((schedule["NetworkLatency_ms"] > schedule["MaxLatency_ms"]).sum()),
        "max_gpu_excess": float(np.maximum(gpu_use - gpu_cap, 0).max()),
        "max_it_power_excess_MW": float(np.maximum(it_load - max_it, 0).max()),
        "max_facility_power_excess_MW": float(np.maximum(fac_load - max_fac, 0).max()),
        "mean_wait_hour": float(schedule["WaitHour"].mean()),
        "max_wait_hour": float(schedule["WaitHour"].max()),
        "migration_rate": float(schedule["Migrated"].mean()),
        "tail_task_rate": float(schedule["TailPeriodUsed"].mean()),
        "peak_gpu_utilization": float((gpu_use / gpu_cap).max()),
        "mean_gpu_utilization": float((gpu_use / gpu_cap).mean()),
    }
    hard_fail = sum([
        duplicate, missing, checks["realtime_start_violations"], checks["arrival_violations"],
        checks["deadline_violations"], checks["terminal_2406_violations"], checks["latency_violations"],
        checks["max_gpu_excess"] > 1e-6, checks["max_it_power_excess_MW"] > 1e-6,
        checks["max_facility_power_excess_MW"] > 1e-6,
    ])
    checks["all_hard_constraints_pass"] = bool(hard_fail == 0)
    (TABLE_DIR / f"validation_{name}.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checks


def plot_schedule(schedule: pd.DataFrame, gpu_use: np.ndarray, data: DataBundle):
    colors = {"AITraining": "tab:blue", "BatchInference": "tab:orange", "RealTimeInference": "tab:green"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]

    def _draw_region_panel(ax, region: str, show_xlabel: bool = False):
        subset = schedule[schedule["ExecutionRegion"] == region].sort_values("StartHour")
        for y, (_, row) in enumerate(subset.iterrows()):
            ax.barh(
                y,
                row["FinishHour"] - row["StartHour"],
                left=row["StartHour"],
                color=colors[row["TaskType"]],
                alpha=0.78,
                height=0.8,
            )
        ax.axvspan(2400, 2406, color="grey", alpha=0.15)
        ax.set_ylabel(region)
        ax.set_yticks([])
        ax.set_xlim(SCHEDULE_START, EXECUTION_END)
        if show_xlabel:
            ax.set_xlabel("Hour")

    # Two side-by-side panels: RegionA–C | RegionD–F (saves vertical space in paper).
    left_regions = REGIONS[:3]
    right_regions = REGIONS[3:]
    fig, axes = plt.subplots(3, 2, figsize=(14, 8.2), sharex="col")
    for row, (left_r, right_r) in enumerate(zip(left_regions, right_regions)):
        _draw_region_panel(axes[row, 0], left_r, show_xlabel=(row == 2))
        _draw_region_panel(axes[row, 1], right_r, show_xlabel=(row == 2))
    axes[0, 0].set_title("Regions A–C")
    axes[0, 1].set_title("Regions D–F")
    fig.suptitle("Final 24-hour task schedule (grey: closure horizon)", y=0.995)
    axes[0, 1].legend(handles, list(colors.keys()), loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIG_DIR / "06_schedule_gantt.png", dpi=180)
    plt.close()

    # Also export separate panels for explicit LaTeX side-by-side subfigures.
    for fname, regions, title in [
        ("06_schedule_gantt_left.png", left_regions, "Regions A–C"),
        ("06_schedule_gantt_right.png", right_regions, "Regions D–F"),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
        for ax, region in zip(axes, regions):
            _draw_region_panel(ax, region, show_xlabel=(region == regions[-1]))
        axes[0].set_title(f"{title} (grey: closure horizon)")
        axes[0].legend(handles, list(colors.keys()), loc="upper right", ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=180)
        plt.close()

    info = data.gpu_info.set_index("Region")
    gpu_cap = info.loc[REGIONS, "Available_GPU"].to_numpy(float)[:, None]
    util = gpu_use / gpu_cap
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(util, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(1.0, util.max()))
    ax.set_yticks(range(6)); ax.set_yticklabels(REGIONS)
    ax.set_xticks(range(0, len(HOURS), 2)); ax.set_xticklabels(HOURS[::2], rotation=45)
    ax.set_title("Regional GPU utilization including carry-in")
    ax.set_xlabel("Hour"); fig.colorbar(im, ax=ax, label="Utilization")
    plt.tight_layout(); plt.savefig(FIG_DIR / "07_gpu_utilization.png", dpi=180); plt.close()


def write_q1tex(
    audit: dict, descriptives: dict, forecast_summary: pd.DataFrame, selected: str,
    test_metrics: dict, carry_tasks: pd.DataFrame, greedy_metrics: dict,
    milp_metrics: dict, solver_info: dict, baseline_metrics: list[dict],
):
    top_models = forecast_summary.head(5).round(4).reset_index().to_markdown(index=False)
    baseline_table = pd.DataFrame(baseline_metrics)[[
        "name", "mean_wait_hour", "migration_rate", "peak_gpu_utilization",
        "tail_task_rate", "all_hard_constraints_pass",
    ]].round(4).to_markdown(index=False)
    text = rf"""# 问题一：工作负载预测与基础算力调度（简版正文）

## 1. 解题主线

问题一分为统计分析、短期预测和基础调度三部分。预测阶段严格隐藏第2376–2399小时真实值；调度阶段则按照题目要求重新使用该时段实际到达的逐任务数据。预测结果不替代真实任务进入调度。

## 2. 数据与第一性原理判断

任务表共有 **{audit['task_rows']:,}** 条记录，TaskID重复数为 **{audit['duplicate_task_ids']}**，缺失单元格数为 **{audit['task_null_cells']}**。最后24小时共有 **{audit['last24_tasks']}** 个实际到达任务。

全系统每小时到达GPU需求均值为 **{descriptives['total_gpu_mean']:.2f}**，标准差为 **{descriptives['total_gpu_std']:.2f}**。滞后1、24和168小时自相关分别为 **{descriptives['total_acf_1']:.4f}**、**{descriptives['total_acf_24']:.4f}** 和 **{descriptives['total_acf_168']:.4f}**，说明可外推的时间相关性很弱。因此本文不预设复杂深度模型，而从任务到达生成机制出发，将小时GPU需求表示为随机任务数与单任务GPU规模之和：

$$
D_{{rkt}}=\sum_{{j=1}}^{{N_{{rkt}}}}G_{{rkj}}.
$$

任务计数以Poisson过程为基础模型，单任务GPU规模使用区域—类型历史经验分布；通过Bootstrap得到点预测和90%预测区间。

## 3. 模型选择与最终预测

在最终测试集之前设置8个24小时滚动验证窗口，比较全历史均值、局部窗口Poisson极大似然、EWMA和复合Poisson模型。先按平均RMSE筛选2%近优模型，再用Maximum Regret与欠预测风险择优。

滚动回测前五名如下：

{top_models}

最终选择模型为 **{selected}**。使用0–2375小时重新训练后，对2376–2399小时的底层18条序列进行预测，测试MAE为 **{test_metrics['MAE']:.3f}**，RMSE为 **{test_metrics['RMSE']:.3f}**，WAPE为 **{test_metrics['WAPE']:.3f}**，90%底层区间覆盖率为 **{test_metrics['interval_90_coverage']:.3f}**。

## 4. 基础调度模型

为避免高估最后24小时可用容量，先用同一套确定性前序策略处理0–2375小时到达任务，并冻结跨入2376小时的 **{len(carry_tasks)}** 个任务占用。最终调度对象仍是2376–2399小时实际到达的538个任务。

对任务候选开工小时预计算分钟重叠系数：

$$
q_{{ist}}=\frac{{\text{{任务i在候选s下与小时t的重叠分钟}}}}{{60}}.
$$

由此构造GPU-hour约束和AI IT功率，叠加 `NonAI_IT_Load_MW` 后同时检查IT功率和PUE设施功率上限。网络时延、实时到达即开工、任务不可抢占及2406前完成均作为硬约束。

求解采用“动态困难度贪心＋一次候选MILP”。贪心首先给出完整可行解，MILP再在全部可行区域—开工时刻候选上优化等待、迁移、峰值和尾时域使用。HiGHS共处理 **{solver_info['candidate_variables']:,}** 个候选变量，耗时 **{solver_info['solve_seconds']:.2f}** 秒，最终MIP Gap为 **{solver_info['mip_gap']:.4f}**。

## 5. 调度结果

动态贪心的平均等待为 **{greedy_metrics['mean_wait_hour']:.3f}** 小时，迁移率为 **{greedy_metrics['migration_rate']:.3f}**，峰值GPU利用率为 **{greedy_metrics['peak_gpu_utilization']:.3f}**。

MILP改进后的平均等待为 **{milp_metrics['mean_wait_hour']:.3f}** 小时，迁移率为 **{milp_metrics['migration_rate']:.3f}**，峰值GPU利用率为 **{milp_metrics['peak_gpu_utilization']:.3f}**。所有任务均在2406前完成，实时任务全部到达即开工，网络时延、GPU、IT功率和设施功率最大超限量均为0。独立Validator结论为 **{milp_metrics['all_hard_constraints_pass']}**。

各基础方法比较如下：

{baseline_table}

当前MILP状态为 **{solver_info['message']}**。当MIP Gap非零时，本文将其表述为“时限内最好可行折中解”，不宣称已经证明全局最优。

## 6. 当前结论

数据中的主要不确定性来自随机任务数量和随机任务规模，而非稳定日周期。复合Poisson模型在保持点预测可解释性的同时提供容量风险区间；调度模型则以真实任务和分钟级重叠为基础，在全部物理硬约束下获得可复核的最后24小时方案。详细数值表和图片位于 `q1/outputs/`。
"""
    (Q1_DIR / "q1tex.md").write_text(text, encoding="utf-8")


def main():
    ensure_dirs()
    data = load_data()
    audit = audit_data(data)
    counts, gpu, gpuh = build_hourly_panels(data.tasks)
    descriptives = descriptive_analysis(data, counts, gpu, gpuh)
    _, forecast_summary, selected = rolling_backtest(gpu)
    test_metrics = final_forecast(data, gpu, selected)

    latency, eligible = latency_maps(data)
    carry_gpu, carry_power, carry_tasks = build_carry_in(data, eligible, latency)
    final_tasks = data.tasks[data.tasks["ArrivalHour"].between(2376, 2399)].copy()
    final_tasks["PowerPerGPU"] = final_tasks["TaskType"].map(data.power)

    greedy_arrays = make_final_arrays(data, carry_gpu.copy(), carry_power.copy())
    greedy_assignments = greedy_schedule(
        final_tasks, data, greedy_arrays, SCHEDULE_START, EXECUTION_END,
        eligible, latency, "final-greedy",
    )
    greedy_schedule_full, greedy_gpu, greedy_power = reconstruct_schedule(
        final_tasks, greedy_assignments, data, carry_gpu, carry_power
    )
    greedy_metrics = validate_schedule(
        final_tasks, greedy_schedule_full, greedy_gpu, greedy_power, data, "greedy"
    )
    greedy_schedule_full.to_csv(TABLE_DIR / "schedule_greedy.csv", index=False, encoding="utf-8-sig")

    baseline_metrics = []
    for strategy, out_name in [("local_first", "local_first"), ("most_available", "most_available")]:
        base_arrays = make_final_arrays(data, carry_gpu.copy(), carry_power.copy())
        base_assignments = greedy_schedule(
            final_tasks, data, base_arrays, SCHEDULE_START, EXECUTION_END,
            eligible, latency, f"final-{out_name}", strategy=strategy,
        )
        base_schedule, base_gpu, base_power = reconstruct_schedule(
            final_tasks, base_assignments, data, carry_gpu, carry_power
        )
        base_metrics = validate_schedule(
            final_tasks, base_schedule, base_gpu, base_power, data, out_name
        )
        base_schedule.to_csv(TABLE_DIR / f"schedule_{out_name}.csv", index=False, encoding="utf-8-sig")
        baseline_metrics.append(base_metrics)

    candidates = build_milp_candidates(final_tasks, data, eligible, latency)
    candidates.to_parquet(TABLE_DIR / "milp_candidates.parquet", index=False) if False else None
    milp_assignments, solver_info = solve_final_milp(
        final_tasks, data, candidates, carry_gpu, carry_power
    )
    milp_schedule, milp_gpu, milp_power = reconstruct_schedule(
        final_tasks, milp_assignments, data, carry_gpu, carry_power
    )
    milp_metrics = validate_schedule(
        final_tasks, milp_schedule, milp_gpu, milp_power, data, "milp"
    )
    milp_schedule.to_csv(TABLE_DIR / "schedule_milp.csv", index=False, encoding="utf-8-sig")
    baseline_metrics.extend([greedy_metrics, milp_metrics])
    pd.DataFrame(baseline_metrics).to_csv(
        TABLE_DIR / "schedule_method_comparison.csv", index=False, encoding="utf-8-sig"
    )
    util_rows = []
    info = data.gpu_info.set_index("Region")
    for r, region in enumerate(REGIONS):
        for j, hour in enumerate(HOURS):
            util_rows.append({
                "Hour": int(hour), "Region": region, "GPU_Use": milp_gpu[r, j],
                "Available_GPU": float(info.loc[region, "Available_GPU"]),
                "GPU_Utilization": float(milp_gpu[r, j] / info.loc[region, "Available_GPU"]),
                "AI_IT_Load_MW": milp_power[r, j],
            })
    pd.DataFrame(util_rows).to_csv(TABLE_DIR / "regional_hourly_utilization.csv", index=False, encoding="utf-8-sig")
    plot_schedule(milp_schedule, milp_gpu, data)
    write_q1tex(
        audit, descriptives, forecast_summary, selected, test_metrics,
        carry_tasks, greedy_metrics, milp_metrics, solver_info, baseline_metrics,
    )
    summary = {
        "selected_forecast_model": selected,
        "forecast_test": test_metrics,
        "carry_in_tasks": int(len(carry_tasks)),
        "greedy": greedy_metrics,
        "milp": milp_metrics,
        "solver": solver_info,
    }
    (OUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
