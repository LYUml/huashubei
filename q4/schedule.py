from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from config import EXECUTION_END, REGIONS
from data_loader import Q4Data

Strategy = Literal["joint", "local_first", "lowest_price", "lowest_carbon"]


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


def candidate_profile(task: pd.Series, start: int) -> tuple[np.ndarray, np.ndarray]:
    hours, fracs = [], []
    last = min(math.ceil(start + task["EstimatedDuration_min"] / 60.0), EXECUTION_END)
    for hour in range(max(start, 0), last):
        q = overlap_fraction(start, int(task["EstimatedDuration_min"]), hour)
        if q > 0:
            hours.append(hour)
            fracs.append(q)
    return np.asarray(hours, dtype=int), np.asarray(fracs, dtype=float)


def feasible(
    task: pd.Series,
    r: int,
    start: int,
    gpu_use: np.ndarray,
    ai_power: np.ndarray,
    data: Q4Data,
    tol: float = 1e-7,
) -> bool:
    idx, q = candidate_profile(task, start)
    if len(idx) == 0:
        return False
    g_inc = float(task["GPU_Demand"]) * q
    p_inc = float(task["GPU_Demand"] * task["PowerPerGPU"]) * q
    if np.any(gpu_use[r, idx] + g_inc > data.gpu_cap[r] + tol):
        return False
    total_it = data.non_ai[r, idx] + ai_power[r, idx] + p_inc
    if np.any(total_it > data.max_it[r] + tol):
        return False
    if np.any(total_it * data.pue[r] > data.max_fac[r] + tol):
        return False
    return True


def place(task: pd.Series, r: int, start: int, gpu_use: np.ndarray, ai_power: np.ndarray) -> None:
    idx, q = candidate_profile(task, start)
    gpu_use[r, idx] += float(task["GPU_Demand"]) * q
    ai_power[r, idx] += float(task["GPU_Demand"] * task["PowerPerGPU"]) * q


def score_candidate(
    task: pd.Series,
    region: str,
    start: int,
    data: Q4Data,
    strategy: Strategy,
    gpu_use: np.ndarray,
    ai_power: np.ndarray,
) -> float:
    r = REGIONS.index(region)
    idx, q = candidate_profile(task, start)
    p_inc = float(task["GPU_Demand"] * task["PowerPerGPU"]) * q
    facility = p_inc * data.pue[r]
    # Proxy: treat incremental facility load as grid-facing energy for cost/carbon ranking.
    energy_cost = float(np.dot(facility, data.price[r, idx]))
    carbon_cost = float(np.dot(facility, data.carbon[r, idx]))
    wait = float(start - task["ArrivalHour"])
    latency = data.latency_map[(task["SourceRegion"], region)]
    migration = float(region != task["SourceRegion"])
    peak_after = float(np.max((gpu_use[r, idx] + float(task["GPU_Demand"]) * q) / data.gpu_cap[r]))

    if strategy == "local_first":
        return 10.0 * migration + 0.8 * wait + 0.1 * latency / max(task["MaxLatency_ms"], 1) + 0.1 * peak_after
    if strategy == "lowest_price":
        return energy_cost + 1e-3 * wait + 1e-4 * latency
    if strategy == "lowest_carbon":
        return 1e3 * carbon_cost + 1e-3 * wait + 1e-4 * latency
    # joint: normalized mix of cost, carbon, wait, latency, migration, peak
    return (
        1.0 * energy_cost / 1000.0
        + 80.0 * carbon_cost
        + 15.0 * wait
        + 0.05 * latency
        + 8.0 * migration
        + 20.0 * peak_after
    )


def schedule_tasks(
    data: Q4Data,
    strategy: Strategy = "joint",
    task_subset: pd.DataFrame | None = None,
    max_delay_scan: int | None = 72,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Phase-1 dynamic greedy scheduling over the full (or subset) task set.

    max_delay_scan: for flexible tasks, only evaluate the first K feasible start
    hours plus a few price/carbon-attractive offsets, to keep 50k-scale tractable.
    """
    tasks = data.tasks if task_subset is None else task_subset.copy()
    if "PowerPerGPU" not in tasks.columns:
        tasks["PowerPerGPU"] = tasks["TaskType"].map(data.power)

    gpu_use = np.zeros((len(REGIONS), EXECUTION_END), dtype=float)
    ai_power = np.zeros_like(gpu_use)

    work = tasks.copy()
    work["EligibleCount"] = work["TaskID"].map(lambda x: len(data.eligible[int(x)]))
    work["DurationHour"] = work["EstimatedDuration_min"] / 60.0
    work["Slack"] = work["LatestFinishHour"] - work["ArrivalHour"] - work["DurationHour"]
    work["GPUHour"] = work["GPU_Demand"] * work["DurationHour"]
    work["RealtimeRank"] = (work["TaskType"] != "RealTimeInference").astype(int)
    work = work.sort_values(
        ["RealtimeRank", "EligibleCount", "Slack", "GPUHour", "ArrivalHour"],
        ascending=[True, True, True, False, True],
    )

    assignments = []
    n_tasks = len(work)
    for i, (_, task) in enumerate(work.iterrows(), start=1):
        if i == 1 or i % 5000 == 0 or i == n_tasks:
            print(f"  [{strategy}] scheduling {i}/{n_tasks}", flush=True)
        starts = list(candidate_hours(task))
        if task["TaskType"] != "RealTimeInference" and max_delay_scan is not None and len(starts) > max_delay_scan:
            # Keep early window + sparse later samples for valley chasing.
            head = starts[: max(24, max_delay_scan // 2)]
            step = max(1, len(starts) // max_delay_scan)
            sampled = starts[::step][:max_delay_scan]
            starts = sorted(set(head) | set(sampled))

        best = None
        for start in starts:
            finish = start + task["EstimatedDuration_min"] / 60.0
            if finish > min(task["LatestFinishHour"], EXECUTION_END) + 1e-9:
                continue
            for region in data.eligible[int(task["TaskID"])]:
                r = REGIONS.index(region)
                if not feasible(task, r, start, gpu_use, ai_power, data):
                    continue
                cost = score_candidate(task, region, start, data, strategy, gpu_use, ai_power)
                if best is None or cost < best[0]:
                    best = (cost, region, start)
        if best is None:
            raise RuntimeError(f"No feasible placement for TaskID={int(task['TaskID'])} under {strategy}")
        _, region, start = best
        place(task, REGIONS.index(region), start, gpu_use, ai_power)
        latency = data.latency_map[(task["SourceRegion"], region)]
        assignments.append(
            {
                "TaskID": int(task["TaskID"]),
                "TaskType": task["TaskType"],
                "SourceRegion": task["SourceRegion"],
                "ExecutionRegion": region,
                "ArrivalHour": int(task["ArrivalHour"]),
                "StartHour": int(start),
                "FinishHour": float(start + task["EstimatedDuration_min"] / 60.0),
                "WaitHour": float(start - task["ArrivalHour"]),
                "GPU_Demand": float(task["GPU_Demand"]),
                "EstimatedDuration_min": float(task["EstimatedDuration_min"]),
                "PowerPerGPU": float(task["PowerPerGPU"]),
                "NetworkLatency_ms": float(latency),
                "MaxLatency_ms": float(task["MaxLatency_ms"]),
                "LatestFinishHour": float(task["LatestFinishHour"]),
                "Migrated": bool(region != task["SourceRegion"]),
            }
        )

    schedule = pd.DataFrame(assignments)
    # Extend AI power array to POWER_HOURS with zeros at 2406 for power stage.
    ai_full = np.zeros((len(REGIONS), EXECUTION_END + 1), dtype=float)
    ai_full[:, :EXECUTION_END] = ai_power
    return schedule, gpu_use, ai_full
