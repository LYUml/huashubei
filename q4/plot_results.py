from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, REGIONS


def plot_scenario_bars(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("operating_cost_CNY", "Operating cost (CNY)"),
        ("carbon_tCO2", "Carbon (tCO2)"),
        ("renewable_utilization", "RE util vs AvailableRE"),
        ("peak_net_import_sum_MW", "Peak net-import sum (MW)"),
        ("mean_wait_hour", "Mean wait (h)"),
        ("mean_network_latency_ms", "Mean latency (ms)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    names = summary["scenario"].tolist()
    x = np.arange(len(names))
    for ax, (col, title) in zip(axes, metrics):
        vals = summary[col].to_numpy(float)
        ax.bar(x, vals, color="tab:blue", alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def plot_carbon_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[summary["scenario"].str.startswith("carbon_") | (summary["scenario"] == "baseline_joint")].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["carbon_tCO2"], sub["operating_cost_CNY"], s=70)
    for r in sub.itertuples():
        ax.annotate(r.scenario, (r.carbon_tCO2, r.operating_cost_CNY), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Carbon emission (tCO2)")
    ax.set_ylabel("Operating cost (CNY)")
    ax.set_title("Cost–carbon trade-off under carbon budgets")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_region_net_import(power_csv: Path, path: Path, title: str) -> None:
    if not power_csv.exists():
        return
    df = pd.read_csv(power_csv)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    for i, region in enumerate(REGIONS):
        sub = df[df["Region"] == region]
        axes[i].plot(sub["Hour"], sub["NetImport_MW"], lw=0.8)
        axes[i].set_title(region)
        axes[i].grid(alpha=0.2)
        axes[i].set_ylabel("MW")
    axes[-1].set_xlabel("Hour")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_soc(power_csv: Path, path: Path, title: str) -> None:
    if not power_csv.exists():
        return
    df = pd.read_csv(power_csv)
    fig, ax = plt.subplots(figsize=(11, 5))
    for region in REGIONS:
        sub = df[df["Region"] == region]
        ax.plot(sub["Hour"], sub["SOC_MWh"], label=region, lw=1.0)
    ax.set_xlabel("Hour")
    ax.set_ylabel("SOC (MWh)")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
