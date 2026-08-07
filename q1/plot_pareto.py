"""从 optimization_experiments.csv 重绘 Q1 等待—峰值 Pareto 图。

与 optimize_q1.run_experiments 末尾的画图逻辑一致，但从已持久化的 CSV 读取，
可在 ε 重跑后独立执行。
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

import optimize_q1 as oq

TABLES = oq.TABLES
FIGURES = oq.FIGURES


def main() -> None:
    opt = pd.read_csv(TABLES / "optimization_experiments.csv")
    rec = json.loads((TABLES / "recommended_schedule.json").read_text(encoding="utf-8"))
    feasible = opt[(opt["scenario"] != "wait_stage1") & opt["all_hard_constraints_pass"]].copy()
    old = pd.read_csv(TABLES / "schedule_method_comparison.csv")
    plot = pd.concat([
        old[["name", "mean_wait_hour", "peak_gpu_utilization"]].rename(columns={"name": "scenario"}),
        feasible[["scenario", "mean_wait_hour", "peak_gpu_utilization"]]
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(plot.mean_wait_hour, plot.peak_gpu_utilization, s=65)
    for r in plot.itertuples():
        ax.annotate(r.scenario, (r.mean_wait_hour, r.peak_gpu_utilization),
                    xytext=(4, 5), textcoords="offset points", fontsize=8)
    pr = plot[plot.scenario == rec["preferred_scenario"]].iloc[0]
    ax.scatter([pr.mean_wait_hour], [pr.peak_gpu_utilization], marker="*", s=230,
               color="tab:red", label="recommended (wait <= 0.10 h)")
    ax.set_xlabel("Mean waiting time (hour)")
    ax.set_ylabel("Peak GPU utilization")
    ax.set_title("Q1 service–peak trade-off under validated schedules")
    ax.grid(alpha=.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "09_full_pareto.png", dpi=180)
    plt.close()
    print("09_full_pareto.png written")


if __name__ == "__main__":
    main()
