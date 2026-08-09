"""Recompute renewable utilization metrics from existing power_*.csv outputs.

Does not re-run scheduling/LP. Fixes the headline utilization denominator:
  renewable_utilization = absorbed / attachment AvailableRenewable
instead of absorbed / deliverable ceiling (~100% by construction).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q4_DIR = Path(__file__).resolve().parent
if str(Q4_DIR) not in sys.path:
    sys.path.insert(0, str(Q4_DIR))

from config import FIGURES, OUT, TABLES
from data_loader import load_data
from plot_results import plot_carbon_tradeoff, plot_scenario_bars


def re_scale_for_scenario(name: str) -> float:
    if name == "re_minus20":
        return 0.8
    if name == "re_plus20":
        return 1.2
    return 1.0


def main() -> None:
    data = load_data()
    raw_base = data.available_re_raw  # [R, T]
    regions = list(dict.fromkeys(pd.read_csv(TABLES / "power_baseline_joint.csv")["Region"]))

    # Build region -> index map matching data.available_re_raw order via config REGIONS
    from config import REGIONS

    region_to_i = {r: i for i, r in enumerate(REGIONS)}

    summary_rows = []
    for metrics_path in sorted(TABLES.glob("metrics_*.json")):
        name = metrics_path.stem.replace("metrics_", "", 1)
        power_path = TABLES / f"power_{name}.csv"
        if not power_path.exists():
            print(f"skip {name}: missing power csv")
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        power = pd.read_csv(power_path)
        factor = re_scale_for_scenario(name)

        raw_vals = []
        for row in power.itertuples(index=False):
            ri = region_to_i[row.Region]
            t = int(row.Hour)
            raw_vals.append(float(raw_base[ri, t] * factor))
        power = power.copy()
        power["AvailableRE_Raw_MW"] = raw_vals

        absorbed = float((power["AvailableRE_MW"] - power["Curtailment_MW"]).sum())
        deliverable = float(power["AvailableRE_MW"].sum())
        raw = float(power["AvailableRE_Raw_MW"].sum())
        util_raw = absorbed / raw if raw > 1e-9 else 0.0
        util_deliv = absorbed / deliverable if deliverable > 1e-9 else 0.0

        metrics["renewable_utilization"] = util_raw
        metrics["renewable_utilization_of_deliverable"] = util_deliv
        metrics["absorbed_re_mwh"] = absorbed
        metrics["available_re_raw_mwh"] = raw
        metrics["available_re_deliverable_mwh"] = deliverable
        metrics["curtail_deliverable_mwh"] = deliverable - absorbed

        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        # Keep column order sensible: insert raw RE after AvailableRE_MW if present
        cols = list(power.columns)
        if "AvailableRE_Raw_MW" in cols:
            cols.remove("AvailableRE_Raw_MW")
            if "AvailableRE_MW" in cols:
                cols.insert(cols.index("AvailableRE_MW") + 1, "AvailableRE_Raw_MW")
            else:
                cols.append("AvailableRE_Raw_MW")
            power = power[cols]
        power.to_csv(power_path, index=False, encoding="utf-8-sig")

        row = dict(metrics)
        peaks = row.pop("peak_net_import_by_region_MW", {}) or {}
        checks = row.pop("resource_checks", {}) or {}
        for region, val in peaks.items():
            row[f"peak_{region}"] = val
        row["resource_hard_pass"] = checks.get("hard_pass", True)
        summary_rows.append(row)
        print(f"{name}: util_raw={util_raw:.4f} util_deliv={util_deliv:.6f} absorbed={absorbed:.1f} raw={raw:.1f}")

    summary = pd.DataFrame(summary_rows)
    # Preserve a stable scenario order from previous summary if available
    old = TABLES / "scenario_summary.csv"
    if old.exists():
        order = pd.read_csv(old)["scenario"].tolist()
        summary["_ord"] = summary["scenario"].apply(lambda s: order.index(s) if s in order else 10**9)
        summary = summary.sort_values("_ord").drop(columns=["_ord"])
    summary.to_csv(TABLES / "scenario_summary.csv", index=False, encoding="utf-8-sig")

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
        "renewable_utilization_of_deliverable": float(rec["renewable_utilization_of_deliverable"]),
        "absorbed_re_mwh": float(rec["absorbed_re_mwh"]),
        "available_re_raw_mwh": float(rec["available_re_raw_mwh"]),
        "peak_net_import_sum_MW": float(rec["peak_net_import_sum_MW"]),
        "mean_wait_hour": float(rec["mean_wait_hour"]),
        "mean_network_latency_ms": float(rec["mean_network_latency_ms"]),
        "qos_loss": float(rec["qos_loss"]),
        "hard_pass": bool(rec["hard_pass"]),
        "note_re_utilization": "renewable_utilization = absorbed / attachment AvailableRenewable (6-region sum). renewable_utilization_of_deliverable is LP fill-rate vs deliverable ceiling.",
        "n_tasks_scheduled": int(rec.get("n_tasks", 50000)),
    }
    (TABLES / "recommended_q4.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_scenario_bars(summary, FIGURES / "01_scenario_metrics.png")
    plot_carbon_tradeoff(summary, FIGURES / "02_cost_carbon_tradeoff.png")

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
        "renewable_utilization_of_deliverable",
        "peak_net_import_sum_MW",
        "mean_wait_hour",
        "mean_network_latency_ms",
        "hard_pass",
    ]
    md.append(summary[show_cols].to_markdown(index=False))
    md.append(
        "\n\n**新能源利用率口径**：`renewable_utilization` = 消纳量 / 附件 AvailableRenewable（六区加总）；"
        "`renewable_utilization_of_deliverable` 为相对可消纳上界的 LP 填充率。"
        "LP 仍使用可消纳上界，避免附件六区重复 AvailableRE 导致近零购电。\n"
    )
    (OUT / "q4_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
