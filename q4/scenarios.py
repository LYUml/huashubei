from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import Q4Data, apply_price_mechanism, scale_renewables
from metrics import aggregate_power_metrics, dump_json, save_region_timeseries, schedule_qos_metrics, validate_schedule_resources
from power_opt import non_ai_carbon_floor, optimize_all_regions
from schedule import schedule_tasks


def _finalize(
    name: str,
    strategy: str,
    schedule: pd.DataFrame,
    ai_power: np.ndarray,
    region_results: list[dict],
    hard: dict,
    out_tables,
    extra_metrics: dict | None = None,
) -> dict:
    sched_m = schedule_qos_metrics(schedule)
    metrics = aggregate_power_metrics(region_results, sched_m)
    metrics["scenario"] = name
    metrics["strategy"] = strategy
    carbon_ok = bool(metrics.get("carbon_feasible", True))
    metrics["hard_pass"] = (
        bool(hard.get("hard_pass", True))
        and metrics["deadline_violation"] == 0
        and metrics["latency_violation"] == 0
        and carbon_ok
    )
    metrics["resource_checks"] = hard
    if extra_metrics:
        metrics.update(extra_metrics)

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
    max_delay_scan: int | None = 24,
    carbon_weight: float | None = None,
    enforce_carbon: bool | None = None,
) -> dict:
    """
    Two-stage Q4 solve:
      1) task scheduling (optional reuse)
      2) storage-power LP with optional carbon/peak/RE constraints
    """
    if reschedule or base_schedule is None or base_ai is None:
        schedule, gpu_use, ai_power = schedule_tasks(
            data,
            strategy=strategy,
            task_subset=task_subset,
            max_delay_scan=max_delay_scan,
            carbon_weight=carbon_weight,
        )
    else:
        schedule, gpu_use, ai_power = base_schedule, None, base_ai

    if gpu_use is not None:
        hard = validate_schedule_resources(schedule, data, gpu_use, ai_power[:, : gpu_use.shape[1]])
    else:
        hard = {"hard_pass": True}

    peak_caps = None
    if peak_scale is not None:
        ref = optimize_all_regions(data, ai_power, time_limit=15.0)
        peak_caps = np.array([r["peak_net_import"] * peak_scale for r in ref], dtype=float)

    enforce = bool(enforce_carbon) if enforce_carbon is not None else (carbon_budget_total is not None)
    region_results = optimize_all_regions(
        data,
        ai_power,
        carbon_budget_total=carbon_budget_total,
        peak_caps=peak_caps,
        min_re_utilization=min_re_utilization,
        time_limit=25.0,
        enforce_carbon=enforce,
    )
    return _finalize(
        name,
        strategy if carbon_weight is None else f"{strategy}|cw={carbon_weight:g}",
        schedule,
        ai_power,
        region_results,
        hard,
        out_tables,
        extra_metrics={"carbon_weight": carbon_weight},
    )


def run_carbon_epsilon(
    data: Q4Data,
    *,
    name: str,
    carbon_budget_total: float,
    task_subset: pd.DataFrame | None,
    base_schedule: pd.DataFrame,
    base_ai: np.ndarray,
    out_tables=None,
    max_delay_scan: int | None = 24,
    non_ai_floor: float | None = None,
    carbon_weights: list[float] | None = None,
) -> dict:
    """
    True ε-constraint on carbon:
      min cost  s.t.  E <= carbon_budget_total
    Stage-1 carbon weight is increased until Stage-2 is feasible, or the
    NonAI / schedule carbon floor proves the budget unreachable.
    """
    budget = float(carbon_budget_total)
    floor = float(non_ai_floor) if non_ai_floor is not None else non_ai_carbon_floor(data)
    weights = carbon_weights or [80.0, 200.0, 500.0, 1500.0, 5000.0, 2.0e4, 1.0e5]

    print(
        f"[carbon] {name}: budget={budget:.1f}, nonAI_floor={floor:.1f}",
        flush=True,
    )

    # Absolute physical lower bound (AI cannot reduce NonAI carbon).
    if floor > budget + 1.0:
        print(
            f"[carbon] {name}: INFEASIBLE — NonAI carbon floor {floor:.1f} > budget {budget:.1f}",
            flush=True,
        )
        # Best-effort: strongest carbon-aware schedule + min-carbon power dispatch.
        schedule, gpu_use, ai_power = schedule_tasks(
            data,
            strategy="joint",
            task_subset=task_subset,
            max_delay_scan=max_delay_scan,
            carbon_weight=weights[-1],
        )
        hard = validate_schedule_resources(schedule, data, gpu_use, ai_power[:, : gpu_use.shape[1]])
        region_results = optimize_all_regions(
            data,
            ai_power,
            carbon_budget_total=budget,
            time_limit=25.0,
            enforce_carbon=True,
        )
        return _finalize(
            name,
            f"joint|cw={weights[-1]:g}",
            schedule,
            ai_power,
            region_results,
            hard,
            out_tables,
            extra_metrics={
                "carbon_weight": weights[-1],
                "carbon_budget_tCO2": budget,
                "carbon_feasible": False,
                "carbon_infeasible_reason": f"nonAI_floor {floor:.3f} > budget {budget:.3f}",
                "nonAI_carbon_floor_tCO2": floor,
                "carbon_search_iterations": 0,
            },
        )

    # First try the baseline joint schedule without rescheduling.
    candidates: list[tuple[float | None, pd.DataFrame, np.ndarray, list[dict], dict]] = []
    print(f"[carbon] {name}: try baseline joint schedule", flush=True)
    region_results = optimize_all_regions(
        data,
        base_ai,
        carbon_budget_total=budget,
        time_limit=25.0,
        enforce_carbon=True,
    )
    meta = region_results[0].get("_meta") or {}
    hard0 = {"hard_pass": True}
    candidates.append((None, base_schedule, base_ai, region_results, hard0))
    if meta.get("carbon_feasible", False):
        print(f"[carbon] {name}: met with baseline schedule, E={meta.get('carbon_min_given_schedule')}", flush=True)
        return _finalize(
            name,
            "joint|baseline_schedule",
            base_schedule,
            base_ai,
            region_results,
            hard0,
            out_tables,
            extra_metrics={
                "carbon_weight": None,
                "carbon_budget_tCO2": budget,
                "nonAI_carbon_floor_tCO2": floor,
                "carbon_search_iterations": 0,
            },
        )

    best = None  # lowest carbon among attempts
    for i, cw in enumerate(weights, start=1):
        print(f"[carbon] {name}: iteration {i}/{len(weights)} carbon_weight={cw:g}", flush=True)
        schedule, gpu_use, ai_power = schedule_tasks(
            data,
            strategy="joint",
            task_subset=task_subset,
            max_delay_scan=max_delay_scan,
            carbon_weight=cw,
        )
        hard = validate_schedule_resources(schedule, data, gpu_use, ai_power[:, : gpu_use.shape[1]])
        region_results = optimize_all_regions(
            data,
            ai_power,
            carbon_budget_total=budget,
            time_limit=25.0,
            enforce_carbon=True,
        )
        meta = region_results[0].get("_meta") or {}
        actual = sum(float(r["carbon"]) for r in region_results if np.isfinite(r["carbon"]))
        emin = meta.get("carbon_min_given_schedule")
        print(
            f"[carbon] {name}: cw={cw:g} actual={actual:.1f} emin={emin} feasible={meta.get('carbon_feasible')}",
            flush=True,
        )
        cand = (cw, schedule, ai_power, region_results, hard)
        candidates.append(cand)
        if best is None or actual < best[0]:
            best = (actual, cand)
        if meta.get("carbon_feasible", False) and actual <= budget + 1e-2:
            return _finalize(
                name,
                f"joint|cw={cw:g}",
                schedule,
                ai_power,
                region_results,
                hard,
                out_tables,
                extra_metrics={
                    "carbon_weight": cw,
                    "carbon_budget_tCO2": budget,
                    "nonAI_carbon_floor_tCO2": floor,
                    "carbon_search_iterations": i,
                },
            )

    # Budget unreachable for explored schedules — keep lowest-carbon attempt.
    assert best is not None
    cw, schedule, ai_power, region_results, hard = best[1]
    print(
        f"[carbon] {name}: INFEASIBLE after search; best actual={best[0]:.1f} > budget={budget:.1f}",
        flush=True,
    )
    return _finalize(
        name,
        f"joint|cw={cw:g}" if cw is not None else "joint|baseline_schedule",
        schedule,
        ai_power,
        region_results,
        hard,
        out_tables,
        extra_metrics={
            "carbon_weight": cw,
            "carbon_budget_tCO2": budget,
            "carbon_feasible": False,
            "carbon_infeasible_reason": (
                region_results[0].get("_meta", {}) or {}
            ).get("carbon_infeasible_reason")
            or f"best_actual {best[0]:.3f} > budget {budget:.3f}",
            "nonAI_carbon_floor_tCO2": floor,
            "carbon_search_iterations": len(weights),
        },
    )


def build_scenario_plan(baseline_carbon: float) -> list[dict]:
    """Scenario definitions for carbon / price / renewable stress tests."""
    plan = []
    for frac, tag in [(1.0, "carbon_100"), (0.9, "carbon_90"), (0.8, "carbon_80"), (0.7, "carbon_70")]:
        plan.append(
            {
                "name": tag,
                "kind": "carbon",
                "carbon_budget_total": baseline_carbon * frac,
                "price_mechanism": "baseline",
                "re_scale": 1.0,
                "peak_scale": None,
                "min_re_utilization": None,
            }
        )
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
    floor = non_ai_carbon_floor(data)
    print(f"[carbon] NonAI carbon floor = {floor:.1f} tCO2; baseline E0 = {base_carbon:.1f}", flush=True)
    outs = []
    for sc in plan:
        d = data
        if sc["price_mechanism"] != "baseline":
            d = apply_price_mechanism(data, sc["price_mechanism"])
        if sc["re_scale"] != 1.0:
            d = scale_renewables(d, sc["re_scale"])

        if sc["kind"] == "carbon":
            outs.append(
                run_carbon_epsilon(
                    d,
                    name=sc["name"],
                    carbon_budget_total=float(sc["carbon_budget_total"]),
                    task_subset=task_subset,
                    base_schedule=joint_result["schedule"],
                    base_ai=joint_result["ai_power"],
                    out_tables=out_tables,
                    non_ai_floor=floor,
                )
            )
            continue

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
                enforce_carbon=False,
            )
        )
    return outs
