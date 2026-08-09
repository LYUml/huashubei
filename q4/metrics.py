from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import REGIONS
from data_loader import Q4Data


def schedule_qos_metrics(schedule: pd.DataFrame) -> dict:
    rt = schedule[schedule["TaskType"] == "RealTimeInference"]
    return {
        "n_tasks": int(len(schedule)),
        "mean_wait_hour": float(schedule["WaitHour"].mean()),
        "max_wait_hour": float(schedule["WaitHour"].max()),
        "migration_rate": float(schedule["Migrated"].mean()),
        "mean_latency_ms": float(schedule["NetworkLatency_ms"].mean()),
        "realtime_on_time_rate": float((rt["StartHour"] == rt["ArrivalHour"]).mean()) if len(rt) else 1.0,
        "deadline_violation": int((schedule["FinishHour"] > schedule["LatestFinishHour"] + 1e-8).sum()),
        "latency_violation": int((schedule["NetworkLatency_ms"] > schedule["MaxLatency_ms"] + 1e-9).sum()),
    }


def aggregate_power_metrics(region_results: list[dict], schedule_metrics: dict) -> dict:
    cost = sum(float(r["cost"]) for r in region_results if np.isfinite(r["cost"]))
    carbon = sum(float(r["carbon"]) for r in region_results if np.isfinite(r["carbon"]))
    # Absorbed RE = operational available − LP curtailment
    # (attachment: utilized = AvailableRE − Curtailment).
    absorbed = sum(float((r["available_re"] - r["curtailment"]).sum()) for r in region_results if np.isfinite(r["curtailment"]).all())
    deliverable = sum(float(r["available_re"].sum()) for r in region_results)
    raw = sum(float(r.get("available_re_raw", r["available_re"]).sum()) for r in region_results)
    # Headline utilization vs attachment AvailableRenewable (not the LP ceiling).
    util_raw = absorbed / raw if raw > 1e-9 else 0.0
    util_deliv = absorbed / deliverable if deliverable > 1e-9 else 0.0
    peaks = {
        r["region"]: float(r["peak_net_import"])
        for r in region_results
        if np.isfinite(r["peak_net_import"])
    }
    peak_sum = float(sum(peaks.values())) if peaks else float("nan")
    # QoS loss: mean wait + migration penalty (smaller better). Realtime must stay perfect.
    q_loss = float(schedule_metrics["mean_wait_hour"]) + 0.5 * float(schedule_metrics["migration_rate"])
    meta = region_results[0].get("_meta") or {}
    return {
        "n_tasks": schedule_metrics["n_tasks"],
        "operating_cost_CNY": cost,
        "carbon_tCO2": carbon,
        "mean_network_latency_ms": schedule_metrics["mean_latency_ms"],
        "qos_loss": q_loss,
        "mean_wait_hour": schedule_metrics["mean_wait_hour"],
        "migration_rate": schedule_metrics["migration_rate"],
        "realtime_on_time_rate": schedule_metrics["realtime_on_time_rate"],
        "renewable_utilization": util_raw,
        "renewable_utilization_of_deliverable": util_deliv,
        "absorbed_re_mwh": absorbed,
        "available_re_raw_mwh": raw,
        "available_re_deliverable_mwh": deliverable,
        "curtail_deliverable_mwh": deliverable - absorbed,
        "peak_net_import_sum_MW": peak_sum,
        "peak_net_import_by_region_MW": peaks,
        "deadline_violation": schedule_metrics["deadline_violation"],
        "latency_violation": schedule_metrics["latency_violation"],
        "carbon_budget_tCO2": meta.get("carbon_budget_total"),
        "carbon_min_given_schedule_tCO2": meta.get("carbon_min_given_schedule"),
        "carbon_feasible": meta.get("carbon_feasible", True),
        "carbon_infeasible_reason": meta.get("carbon_infeasible_reason"),
    }


def validate_schedule_resources(schedule: pd.DataFrame, data: Q4Data, gpu_use: np.ndarray, ai_power: np.ndarray) -> dict:
    T = gpu_use.shape[1]
    excess_gpu = 0.0
    excess_it = 0.0
    excess_fac = 0.0
    for r, region in enumerate(REGIONS):
        for t in range(T):
            excess_gpu = max(excess_gpu, gpu_use[r, t] - data.gpu_cap[r])
            it = data.non_ai[r, t] + ai_power[r, t]
            excess_it = max(excess_it, it - data.max_it[r])
            excess_fac = max(excess_fac, it * data.pue[r] - data.max_fac[r])
    checks = {
        "max_gpu_excess": float(max(0.0, excess_gpu)),
        "max_it_excess_MW": float(max(0.0, excess_it)),
        "max_facility_excess_MW": float(max(0.0, excess_fac)),
        "finish_after_2406": int((schedule["FinishHour"] > 2406 + 1e-8).sum()),
    }
    checks["hard_pass"] = (
        checks["max_gpu_excess"] <= 1e-6
        and checks["max_it_excess_MW"] <= 1e-6
        and checks["max_facility_excess_MW"] <= 1e-6
        and checks["finish_after_2406"] == 0
    )
    return checks


def save_region_timeseries(region_results: list[dict], path: Path) -> pd.DataFrame:
    rows = []
    for r in region_results:
        T = len(r["grid_purchase"])
        for t in range(T):
            rows.append(
                {
                    "Hour": t,
                    "Region": r["region"],
                    "TotalLoad_MW": float(r["total_load"][t]),
                    "AvailableRE_MW": float(r["available_re"][t]),
                    "AvailableRE_Raw_MW": float(
                        r.get("available_re_raw", r["available_re"])[t]
                    ),
                    "GridPurchase_MW": float(r["grid_purchase"][t]),
                    "GridSell_MW": float(r["grid_sell"][t]),
                    "UsedRE_MW": float(r["used_re"][t]),
                    "RE_Charge_MW": float(r["re_charge"][t]),
                    "GridCharge_MW": float(r["grid_charge"][t]),
                    "Discharge_MW": float(r["discharge"][t]),
                    "Curtailment_MW": float(r["curtailment"][t]),
                    "NetImport_MW": float(r["net_import"][t]),
                    "SOC_MWh": float(r["soc"][t]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def dump_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
