from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from config import EXECUTION_END, REGIONS
from data_loader import Q4Data

Strategy = Literal["joint", "local_first", "lowest_price", "lowest_carbon"]
REGION_INDEX = {r: i for i, r in enumerate(REGIONS)}


def overlap_fraction(start_hour: int, duration_min: int, hour: int) -> float:
    start_min = 60 * start_hour
    end_min = start_min + int(duration_min)
    overlap = max(0, min(end_min, 60 * (hour + 1)) - max(start_min, 60 * hour))
    return overlap / 60.0


def candidate_hours(task_type: str, arrival: int, latest_finish: int, duration_min: float) -> range:
    if task_type == "RealTimeInference":
        return range(arrival, arrival + 1)
    latest = min(int(latest_finish), EXECUTION_END)
    last_start = math.floor(latest - float(duration_min) / 60.0 + 1e-9)
    return range(arrival, last_start + 1)


def candidate_profile(duration_min: int, start: int) -> tuple[np.ndarray, np.ndarray]:
    hours, fracs = [], []
    last = min(math.ceil(start + duration_min / 60.0), EXECUTION_END)
    for hour in range(max(start, 0), last):
        q = overlap_fraction(start, duration_min, hour)
        if q > 0:
            hours.append(hour)
            fracs.append(q)
    return np.asarray(hours, dtype=int), np.asarray(fracs, dtype=float)


def feasible(
    gpu_demand: float,
    power_per_gpu: float,
    r: int,
    start: int,
    duration_min: int,
    gpu_use: np.ndarray,
    ai_power: np.ndarray,
    data: Q4Data,
    tol: float = 1e-7,
) -> bool:
    idx, q = candidate_profile(duration_min, start)
    if len(idx) == 0:
        return False
    g_inc = gpu_demand * q
    p_inc = gpu_demand * power_per_gpu * q
    if np.any(gpu_use[r, idx] + g_inc > data.gpu_cap[r] + tol):
        return False
    total_it = data.non_ai[r, idx] + ai_power[r, idx] + p_inc
    if np.any(total_it > data.max_it[r] + tol):
        return False
    if np.any(total_it * data.pue[r] > data.max_fac[r] + tol):
        return False
    return True


def place(
    gpu_demand: float,
    power_per_gpu: float,
    r: int,
    start: int,
    duration_min: int,
    gpu_use: np.ndarray,
    ai_power: np.ndarray,
) -> None:
    idx, q = candidate_profile(duration_min, start)
    gpu_use[r, idx] += gpu_demand * q
    ai_power[r, idx] += gpu_demand * power_per_gpu * q


def sample_starts(starts: list[int], max_delay_scan: int | None, realtime: bool) -> list[int]:
    if realtime or max_delay_scan is None or len(starts) <= max_delay_scan:
        return starts
    head = starts[: max(16, max_delay_scan // 2)]
    step = max(1, len(starts) // max_delay_scan)
    sampled = starts[::step][:max_delay_scan]
    return sorted(set(head) | set(sampled))


def score_candidate(
    *,
    strategy: Strategy,
    source: str,
    region: str,
    start: int,
    arrival: int,
    duration_min: int,
    gpu_demand: float,
    power_per_gpu: float,
    max_latency: float,
    data: Q4Data,
    gpu_use: np.ndarray,
    carbon_weight: float | None = None,
) -> float:
    r = REGION_INDEX[region]
    idx, q = candidate_profile(duration_min, start)
    facility = gpu_demand * power_per_gpu * q * data.pue[r]
    energy_cost = float(np.dot(facility, data.price[r, idx]))
    carbon_cost = float(np.dot(facility, data.carbon[r, idx]))
    wait = float(start - arrival)
    latency = data.latency_map[(source, region)]
    migration = float(region != source)
    peak_after = float(np.max((gpu_use[r, idx] + gpu_demand * q) / data.gpu_cap[r]))

    if strategy == "local_first":
        return 10.0 * migration + 0.8 * wait + 0.1 * latency / max(max_latency, 1) + 0.1 * peak_after
    if strategy == "lowest_price":
        return energy_cost + 1e-3 * wait + 1e-4 * latency
    if strategy == "lowest_carbon":
        cw = 1e3 if carbon_weight is None else float(carbon_weight)
        return cw * carbon_cost + 1e-3 * wait + 1e-4 * latency
    # joint (default) or carbon-weight override used by ε-constraint search
    cw = 80.0 if carbon_weight is None else float(carbon_weight)
    return (
        energy_cost / 1000.0
        + cw * carbon_cost
        + 15.0 * wait
        + 0.05 * latency
        + 8.0 * migration
        + 20.0 * peak_after
    )


def schedule_tasks(
    data: Q4Data,
    strategy: Strategy = "joint",
    task_subset: pd.DataFrame | None = None,
    max_delay_scan: int | None = 48,
    carbon_weight: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Phase-1 dynamic greedy scheduling.

    max_delay_scan limits flexible-task start enumeration for tractability.
    carbon_weight overrides the joint/lowest_carbon carbon coefficient when set.
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
    for i, row in enumerate(work.itertuples(index=False), start=1):
        if i == 1 or i % 5000 == 0 or i == n_tasks:
            tag = strategy if carbon_weight is None else f"{strategy}/cw={carbon_weight:g}"
            print(f"  [{tag}] scheduling {i}/{n_tasks}", flush=True)

        task_id = int(row.TaskID)
        task_type = row.TaskType
        source = row.SourceRegion
        arrival = int(row.ArrivalHour)
        duration_min = int(row.EstimatedDuration_min)
        gpu_demand = float(row.GPU_Demand)
        power_per_gpu = float(row.PowerPerGPU)
        max_latency = float(row.MaxLatency_ms)
        latest_finish = float(row.LatestFinishHour)
        realtime = task_type == "RealTimeInference"

        starts = sample_starts(
            list(candidate_hours(task_type, arrival, int(latest_finish), duration_min)),
            max_delay_scan,
            realtime,
        )
        regions = data.eligible[task_id]
        # local_first: try source region earliest first
        if strategy == "local_first" and source in regions:
            regions = [source] + [r for r in regions if r != source]

        best = None
        for start in starts:
            finish = start + duration_min / 60.0
            if finish > min(latest_finish, EXECUTION_END) + 1e-9:
                continue
            for region in regions:
                r = REGION_INDEX[region]
                if not feasible(gpu_demand, power_per_gpu, r, start, duration_min, gpu_use, ai_power, data):
                    continue
                if strategy == "local_first":
                    # earliest feasible with local preference already ordered
                    best = (0.0, region, start)
                    break
                cost = score_candidate(
                    strategy=strategy,
                    source=source,
                    region=region,
                    start=start,
                    arrival=arrival,
                    duration_min=duration_min,
                    gpu_demand=gpu_demand,
                    power_per_gpu=power_per_gpu,
                    max_latency=max_latency,
                    data=data,
                    gpu_use=gpu_use,
                    carbon_weight=carbon_weight,
                )
                if best is None or cost < best[0]:
                    best = (cost, region, start)
            if strategy == "local_first" and best is not None:
                break

        if best is None:
            raise RuntimeError(f"No feasible placement for TaskID={task_id} under {strategy}")
        _, region, start = best
        place(gpu_demand, power_per_gpu, REGION_INDEX[region], start, duration_min, gpu_use, ai_power)
        latency = data.latency_map[(source, region)]
        assignments.append(
            {
                "TaskID": task_id,
                "TaskType": task_type,
                "SourceRegion": source,
                "ExecutionRegion": region,
                "ArrivalHour": arrival,
                "StartHour": int(start),
                "FinishHour": float(start + duration_min / 60.0),
                "WaitHour": float(start - arrival),
                "GPU_Demand": gpu_demand,
                "EstimatedDuration_min": float(duration_min),
                "PowerPerGPU": power_per_gpu,
                "NetworkLatency_ms": float(latency),
                "MaxLatency_ms": max_latency,
                "LatestFinishHour": latest_finish,
                "Migrated": bool(region != source),
            }
        )

    schedule = pd.DataFrame(assignments)
    ai_full = np.zeros((len(REGIONS), EXECUTION_END + 1), dtype=float)
    ai_full[:, :EXECUTION_END] = ai_power
    return schedule, gpu_use, ai_full
