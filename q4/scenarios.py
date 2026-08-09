from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import Q4Data, apply_price_mechanism, scale_renewables
from metrics import aggregate_power_metrics, dump_json, save_region_timeseries, schedule_qos_metrics, validate_schedule_resources
from power_opt import optimize_all_regions
from schedule import schedule_tasks


def run_pipeline(
    data: Q4Data,
    *,
    name: str,
    strategy: str = "joint",
    task_subset: pd.DataFrame | None = None,
    carbon_budget_total: float | None = None,
    peak_scale: float | None = None,
    min_re_utilization: float | None = None,
    reschedule: bool = True,
    base_schedule: pd.DataFrame | None = None,
    base_ai: np.ndarray | None = None,
    out_tables=None,
) -> dict:
    """
    Two-stage Q4 solve:
      1) task scheduling (optional reuse)
      2) storage-power LP with optional carbon/peak/RE constraints
    """
    if reschedule or base_schedule is None or base_ai is None:
        schedule, gpu_use, ai_power = schedule_tasks(data, strategy=strategy, task_subset=task_subset)
    else:
        schedule, gpu_use, ai_power = base_schedule, None, base_ai

    sched_m = schedule_qos_metrics(schedule)
    if gpu_use is not None:
        hard = validate_schedule_resources(schedule, data, gpu_use, ai_power[:, : gpu_use.shape[1]])
    else:
        hard = {"hard_pass": True}

    # Optional peak caps from a reference unconstrained solve
    peak_caps = None
    if peak_scale is not None:
        ref = optimize_all_regions(data, ai_power, time_limit=15.0)
        peak_caps = np.array([r["peak_net_import"] * peak_scale for r in ref], dtype=float)

    region_results = optimize_all_regions(
        data,
        ai_power,
        carbon_budget_total=carbon_budget_total,
        peak_caps=peak_caps,
        min_re_utilization=min_re_utilization,
        time_limit=25.0,
    )
    metrics = aggregate_power_metrics(region_results, sched_m)
    metrics["scenario"] = name
    metrics["strategy"] = strategy
    metrics["hard_pass"] = bool(hard.get("hard_pass", True)) and metrics["deadline_violation"] == 0 and metrics["latency_violation"] == 0
    metrics["resource_checks"] = hard

    if out_tables is not None:
        schedule.to_csv(out_tables / f"schedule_{name}.csv", index=False, encoding="utf-8-sig")
        save_region_timeseries(region_results, out_tables / f"power_{name}.csv")
        dump_json(metrics, out_tables / f"metrics_{name}.json")

    return {
        "name": name,
        "metrics": metrics,
        "schedule": schedule,
        "ai_power": ai_power,
        "region_results": region_results,
    }


def build_scenario_plan(baseline_carbon: float) -> list[dict]:
    """Scenario definitions for carbon / price / renewable stress tests."""
    plan = []
    # Carbon budgets relative to baseline carbon.
    # Tight budgets reschedule with a carbon-heavy proxy because Stage-2 alone
    # cannot cut carbon once renewable utilization is already ~100%.
    for frac, tag in [(1.0, "carbon_100"), (0.9, "carbon_90"), (0.8, "carbon_80"), (0.7, "carbon_70")]:
        plan.append(
            {
                "name": tag,
                "kind": "carbon",
                "strategy": "joint" if frac >= 0.95 else "lowest_carbon",
                "carbon_budget_total": baseline_carbon * frac,
                "price_mechanism": "baseline",
                "re_scale": 1.0,
                "reschedule": frac < 0.95,
                "peak_scale": None,
                "min_re_utilization": None,
            }
        )
    # Price mechanisms (reschedule with modified prices)
    for mech in ["peak_valley_amplify", "flat", "carbon_linked"]:
        plan.append(
            {
                "name": f"price_{mech}",
                "kind": "price",
                "strategy": "joint",
                "carbon_budget_total": None,
                "price_mechanism": mech,
                "re_scale": 1.0,
                "reschedule": True,
                "peak_scale": None,
                "min_re_utilization": None,
            }
        )
    # Renewable volatility
    for scale, tag in [(0.8, "re_minus20"), (1.2, "re_plus20")]:
        plan.append(
            {
                "name": tag,
                "kind": "renewable",
                "strategy": "joint",
                "carbon_budget_total": None,
                "price_mechanism": "baseline",
                "re_scale": scale,
                "reschedule": False,
                "peak_scale": None,
                "min_re_utilization": 0.90 if scale >= 1.0 else None,
            }
        )
    # Peak shaving stress on the joint schedule
    plan.append(
        {
            "name": "peak_cap_90",
            "kind": "peak",
            "strategy": "joint",
            "carbon_budget_total": None,
            "price_mechanism": "baseline",
            "re_scale": 1.0,
            "reschedule": False,
            "peak_scale": 0.90,
            "min_re_utilization": None,
        }
    )
    return plan


def run_baselines(data: Q4Data, task_subset: pd.DataFrame | None, out_tables) -> list[dict]:
    results = []
    for strategy in ["local_first", "lowest_price", "lowest_carbon", "joint"]:
        results.append(
            run_pipeline(
                data,
                name=f"baseline_{strategy}",
                strategy=strategy,
                task_subset=task_subset,
                out_tables=out_tables,
            )
        )
    return results


def run_scenario_suite(
    data: Q4Data,
    joint_result: dict,
    task_subset: pd.DataFrame | None,
    out_tables,
) -> list[dict]:
    base_carbon = float(joint_result["metrics"]["carbon_tCO2"])
    plan = build_scenario_plan(base_carbon)
    outs = []
    for sc in plan:
        d = data
        if sc["price_mechanism"] != "baseline":
            d = apply_price_mechanism(data, sc["price_mechanism"])
        if sc["re_scale"] != 1.0:
            d = scale_renewables(d, sc["re_scale"])
        outs.append(
            run_pipeline(
                d,
                name=sc["name"],
                strategy=sc.get("strategy", "joint"),
                task_subset=task_subset,
                carbon_budget_total=sc["carbon_budget_total"],
                peak_scale=sc["peak_scale"],
                min_re_utilization=sc["min_re_utilization"],
                reschedule=sc["reschedule"],
                base_schedule=joint_result["schedule"],
                base_ai=joint_result["ai_power"],
                out_tables=out_tables,
            )
        )
    return outs
