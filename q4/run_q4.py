from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
if str(Q4_DIR) not in sys.path:
    sys.path.insert(0, str(Q4_DIR))

from config import FIGURES, OUT, TABLES
from data_loader import load_data
from plot_results import plot_carbon_tradeoff, plot_region_net_import, plot_scenario_bars, plot_soc
from scenarios import run_baselines, run_scenario_suite


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def select_tasks(tasks: pd.DataFrame, fast: bool, start_hour: int | None) -> pd.DataFrame | None:
    if not fast and start_hour is None:
        return None
    lo = 2300 if start_hour is None else int(start_hour)
    subset = tasks[tasks["ArrivalHour"] >= lo].copy()
    print(f"[fast] using {len(subset)} / {len(tasks)} tasks with ArrivalHour >= {lo}")
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4 multi-region compute–storage–power co-optimization")
    parser.add_argument("--fast", action="store_true", help="Use late-horizon task subset for quick runs")
    parser.add_argument("--start-hour", type=int, default=None, help="Only schedule tasks arriving at/after this hour")
    parser.add_argument("--skip-scenarios", action="store_true", help="Only run baseline strategies")
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.time()
    print("Loading data...")
    data = load_data()
    task_subset = select_tasks(data.tasks, args.fast, args.start_hour)

    print("Running baseline strategies (local_first / lowest_price / lowest_carbon / joint)...")
    baselines = run_baselines(data, task_subset, TABLES)
    joint = next(r for r in baselines if r["name"] == "baseline_joint")

    scenario_results = []
    if not args.skip_scenarios:
        print("Running carbon / price / renewable scenarios...")
        scenario_results = run_scenario_suite(data, joint, task_subset, TABLES)

    all_results = baselines + scenario_results
    rows = []
    for r in all_results:
        m = dict(r["metrics"])
        peaks = m.pop("peak_net_import_by_region_MW", {})
        checks = m.pop("resource_checks", {})
        row = m
        for region, val in peaks.items():
            row[f"peak_{region}"] = val
        row["resource_hard_pass"] = checks.get("hard_pass", True)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "scenario_summary.csv", index=False, encoding="utf-8-sig")

    # Recommended: joint baseline (or tightest feasible carbon scenario with wait<=threshold)
    feasible = summary[summary["hard_pass"] == True].copy()  # noqa: E712
    preferred = feasible[feasible["scenario"] == "baseline_joint"]
    if preferred.empty:
        preferred = feasible.sort_values("operating_cost_CNY").head(1)
    rec = preferred.iloc[0].to_dict()
    recommendation = {
        "preferred_scenario": rec["scenario"],
        "selection_rule": "joint two-stage co-optimization minimizing operating cost under physical constraints; scenarios compare carbon budgets, price mechanisms, and renewable volatility",
        "operating_cost_CNY": float(rec["operating_cost_CNY"]),
        "carbon_tCO2": float(rec["carbon_tCO2"]),
        "renewable_utilization": float(rec["renewable_utilization"]),
        "peak_net_import_sum_MW": float(rec["peak_net_import_sum_MW"]),
        "mean_wait_hour": float(rec["mean_wait_hour"]),
        "mean_network_latency_ms": float(rec["mean_network_latency_ms"]),
        "qos_loss": float(rec["qos_loss"]),
        "hard_pass": bool(rec["hard_pass"]),
        "elapsed_seconds": time.time() - t0,
        "fast_mode": bool(args.fast or args.start_hour is not None),
        "n_tasks_scheduled": int(joint["metrics"]["n_tasks"]) if "n_tasks" in joint["metrics"] else int(len(joint["schedule"])),
    }
    # n_tasks may not be in aggregate metrics — add from schedule
    recommendation["n_tasks_scheduled"] = int(len(joint["schedule"]))
    (TABLES / "recommended_q4.json").write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Plotting...")
    plot_scenario_bars(summary, FIGURES / "01_scenario_metrics.png")
    plot_carbon_tradeoff(summary, FIGURES / "02_cost_carbon_tradeoff.png")
    plot_region_net_import(TABLES / "power_baseline_joint.csv", FIGURES / "03_net_import_joint.png", "Net grid import — joint baseline")
    plot_soc(TABLES / "power_baseline_joint.csv", FIGURES / "04_soc_joint.png", "Storage SOC trajectories — joint baseline")

    # Compact markdown report
    md = []
    md.append("# Q4 Run Summary\n")
    md.append("## Recommended joint solution\n")
    md.append("```json\n" + json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n```\n")
    md.append("## Scenario comparison\n")
    show_cols = [
        "scenario",
        "operating_cost_CNY",
        "carbon_tCO2",
        "renewable_utilization",
        "peak_net_import_sum_MW",
        "mean_wait_hour",
        "mean_network_latency_ms",
        "hard_pass",
    ]
    md.append(summary[show_cols].to_markdown(index=False))
    md.append("\n\nMethod: two-stage co-optimization. Stage 1 schedules tasks with a multi-factor greedy proxy; Stage 2 solves regional storage–power LPs minimizing operating cost subject to optional carbon / peak / RE-utilization constraints.\n")
    (OUT / "q4_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"Done in {time.time() - t0:.1f}s. Outputs written to {OUT}")


if __name__ == "__main__":
    main()
