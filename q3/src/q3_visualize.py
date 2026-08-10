#!/usr/bin/env python3
"""华数杯 C 题第三问：从优化结果生成论文级静态图表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#3B6FB6",
    "blue_dark": "#234A7C",
    "blue_light": "#AFC8E8",
    "gold": "#C79A3B",
    "gold_light": "#E9D39B",
    "orange": "#D97941",
    "olive": "#718355",
    "pink": "#B56576",
    "ink": "#20262E",
    "muted": "#6B7280",
    "grid": "#D8DEE8",
    "paper": "#FBFCFE",
}

SCENARIO_ORDER = [
    "no_storage",
    "cost_min",
    "carbon_min",
    "peak_min",
    "smooth_min",
    "balanced",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="生成第三问论文图表")
    parser.add_argument(
        "--results-dir", type=Path, default=project_dir / "results"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=project_dir / "figures"
    )
    return parser.parse_args()


def choose_chinese_font() -> str:
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "SimHei",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    for candidate in candidates:
        if candidate in installed:
            return candidate
    return "DejaVu Sans"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [choose_chinese_font(), "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "savefig.facecolor": COLORS["paper"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"saved {path}")


def subtitle(fig: plt.Figure, text: str, y: float = 0.955) -> None:
    fig.text(0.5, y, text, ha="center", va="top", fontsize=9, color=COLORS["muted"])


def scenario_metric_comparison(system: pd.DataFrame, output: Path) -> None:
    selected = system.loc[
        system["Scenario"].isin(["cost_min", "smooth_min", "balanced"])
    ].copy()
    selected["Scenario"] = pd.Categorical(
        selected["Scenario"], ["cost_min", "smooth_min", "balanced"], ordered=True
    )
    selected = selected.sort_values("Scenario")
    metrics = [
        ("CostReduction_pct", "净电费改善"),
        ("CarbonReduction_pct", "碳排放降低"),
        ("RegionalPeakReduction_pct", "区域峰值降低"),
        ("RegionalFluctuationReduction_pct", "区域波动降低"),
    ]
    palette = [COLORS["blue"], COLORS["gold"], COLORS["orange"]]
    x = np.arange(len(metrics))
    width = 0.23

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for i, (_, row) in enumerate(selected.iterrows()):
        values = [float(row[column]) for column, _ in metrics]
        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=row["Scenario_CN"],
            color=palette[i],
            edgecolor=COLORS["ink"],
            linewidth=0.55,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=COLORS["ink"])
    ax.axhline(0, color=COLORS["ink"], linewidth=0.9)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("相对无储能反事实的改善幅度（%）")
    ax.set_ylim(min(-2, selected["CostReduction_pct"].min() - 5), 112)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax.set_title("储能优化策略的系统级指标比较", pad=42)
    subtitle(fig, "完整时域：第 0-2406 小时；正值表示相对无储能反事实得到改善")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output)


def regional_balanced_impact(regional: pd.DataFrame, output: Path) -> None:
    base = regional.loc[regional["Scenario"] == "no_storage"].set_index("Region")
    balanced = regional.loc[regional["Scenario"] == "balanced"].set_index("Region")
    regions = list(base.index)
    measures = [
        (
            (base["ElectricityCost_CNY"] - balanced["ElectricityCost_CNY"]) / 1e6,
            "净电费改善（百万元）",
            COLORS["blue"],
        ),
        (
            base["CarbonEmission_tCO2"] - balanced["CarbonEmission_tCO2"],
            "碳排放降低（tCO$_2$）",
            COLORS["olive"],
        ),
        (
            base["PeakNetGridImport_MW"] - balanced["PeakNetGridImport_MW"],
            "峰值净购电降低（MW）",
            COLORS["gold"],
        ),
        (
            base["StdNetGridImport_MW"] - balanced["StdNetGridImport_MW"],
            "净购电标准差降低（MW）",
            COLORS["orange"],
        ),
        (
            balanced["RenewableUtilization_pct"] - base["RenewableUtilization_pct"],
            "新能源利用率提升（百分点）",
            COLORS["pink"],
        ),
        (
            balanced["EquivalentFullCycles"],
            "全时域等效完整循环次数",
            COLORS["blue_dark"],
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    for ax, (series, title, color) in zip(axes.flat, measures):
        values = series.reindex(regions).to_numpy(float)
        bars = ax.bar(
            regions,
            values,
            color=color,
            edgecolor=COLORS["ink"],
            linewidth=0.5,
        )
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.7)
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("多目标折中策略的区域影响", fontsize=15, fontweight="semibold", y=0.995)
    subtitle(fig, "以无储能反事实为参照；A-C 区域因新能源始终覆盖负荷且不能外送，储能不动作", y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5, w_pad=1.7)
    save_figure(fig, output)


def representative_week(
    balanced: pd.DataFrame,
    no_storage: pd.DataFrame,
    output: Path,
    region: str = "RegionF",
) -> None:
    opt = balanced.loc[balanced["Region"] == region].sort_values("Hour").copy()
    base = no_storage.loc[no_storage["Region"] == region].sort_values("Hour").copy()
    throughput = opt["ChargePower_MW"] + opt["DischargePower_MW"]
    rolling = throughput.rolling(168, min_periods=168).sum()
    end_idx = int(rolling.idxmax())
    local_position = opt.index.get_loc(end_idx)
    start_pos = max(0, local_position - 167)
    end_pos = min(len(opt), start_pos + 168)
    opt = opt.iloc[start_pos:end_pos].copy()
    hours = opt["Hour"].to_numpy(int)
    base = base.set_index("Hour").loc[hours].reset_index()
    capacity = float(opt["SOC_MWh"].max())
    capacity = max(capacity, 1.0)

    fig, axes = plt.subplots(
        4, 1, figsize=(12.2, 9.6), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.9]}
    )
    axes[0].plot(hours, opt["FacilityLoad_MW"], color=COLORS["ink"], lw=1.5, label="设施负荷")
    axes[0].plot(hours, opt["AvailableRenewable_MW"], color=COLORS["olive"], lw=1.5, ls="--", label="可用新能源")
    axes[0].set_ylabel("功率（MW）")
    axes[0].legend(ncol=2, frameon=False, loc="upper right")

    axes[1].plot(hours, base["NetGridImport_MW"], color=COLORS["muted"], lw=1.3, ls="--", label="无储能")
    axes[1].plot(hours, opt["NetGridImport_MW"], color=COLORS["blue"], lw=1.7, label="多目标折中")
    axes[1].axhline(0, color=COLORS["ink"], lw=0.8)
    axes[1].set_ylabel("净购电（MW）")
    axes[1].legend(ncol=2, frameon=False, loc="upper right")

    axes[2].fill_between(hours, 0, opt["ChargePower_MW"], color=COLORS["blue_light"], edgecolor=COLORS["blue_dark"], linewidth=0.4, label="充电")
    axes[2].fill_between(hours, 0, -opt["DischargePower_MW"], color=COLORS["gold_light"], edgecolor=COLORS["gold"], linewidth=0.4, label="放电")
    axes[2].axhline(0, color=COLORS["ink"], lw=0.8)
    axes[2].set_ylabel("储能功率（MW）")
    axes[2].legend(ncol=2, frameon=False, loc="upper right")

    axes[3].plot(hours, opt["SOC_MWh"], color=COLORS["orange"], lw=1.7)
    axes[3].fill_between(hours, opt["SOC_MWh"], color=COLORS["orange"], alpha=0.12)
    axes[3].set_ylabel("SOC（MWh）")
    axes[3].set_xlabel("小时")
    axes[3].set_ylim(bottom=0, top=max(capacity * 1.08, opt["SOC_MWh"].max() * 1.08))

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    start_hour, end_hour = int(hours[0]), int(hours[-1])
    fig.suptitle(f"{region} 储能动作最活跃的一周", fontsize=15, fontweight="semibold", y=0.995)
    subtitle(fig, f"小时 {start_hour}-{end_hour}；充电为正、放电为负，净购电小于零表示向电网外送", y=0.967)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6)
    save_figure(fig, output)


def soc_heatmap(balanced: pd.DataFrame, storage_path: Path, output: Path) -> None:
    storage = pd.read_excel(storage_path, sheet_name="storage_information").set_index("Region")
    frame = balanced.copy()
    frame["Day"] = frame["Hour"] // 24
    frame["SOC_pct"] = 100.0 * frame.apply(
        lambda row: row["SOC_MWh"] / storage.loc[row["Region"], "StorageCapacity_MWh"], axis=1
    )
    daily = frame.pivot_table(index="Region", columns="Day", values="SOC_pct", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12.4, 4.5))
    image = ax.imshow(daily.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(np.arange(len(daily.index)), daily.index)
    tick_days = np.arange(0, daily.shape[1], 10)
    ax.set_xticks(tick_days, tick_days)
    ax.set_xlabel("运行日（Day）")
    ax.set_ylabel("区域")
    ax.set_title("多目标折中策略下的日均 SOC", pad=38)
    subtitle(fig, "SOC 按各区域储能额定容量归一化；完整时域第 0-2406 小时")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("SOC（%）")
    ax.grid(False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, output)


def regional_peak_window(balanced: pd.DataFrame, no_storage: pd.DataFrame, output: Path) -> None:
    base_f = no_storage.loc[no_storage["Region"] == "RegionF"].copy()
    peak_hour = int(base_f.loc[base_f["NetGridImport_MW"].idxmax(), "Hour"])
    start = max(0, peak_hour - 36)
    end = min(2406, peak_hour + 36)
    regions = sorted(balanced["Region"].unique())

    fig, axes = plt.subplots(3, 2, figsize=(12.2, 8.6), sharex=True)
    for ax, region in zip(axes.flat, regions):
        b = no_storage.loc[
            (no_storage["Region"] == region)
            & no_storage["Hour"].between(start, end)
        ]
        o = balanced.loc[
            (balanced["Region"] == region) & balanced["Hour"].between(start, end)
        ]
        ax.plot(b["Hour"], b["NetGridImport_MW"], color=COLORS["muted"], lw=1.2, ls="--", label="无储能")
        ax.plot(o["Hour"], o["NetGridImport_MW"], color=COLORS["blue"], lw=1.55, label="多目标折中")
        ax.axhline(0, color=COLORS["ink"], lw=0.7)
        ax.set_title(region)
        ax.set_ylabel("净购电（MW）")
        if np.allclose(b["NetGridImport_MW"], 0.0) and np.allclose(
            o["NetGridImport_MW"], 0.0
        ):
            ax.set_ylim(-1.0, 1.0)
            ax.text(
                0.5,
                0.68,
                "该窗口净购电恒为 0",
                transform=ax.transAxes,
                ha="center",
                color=COLORS["muted"],
                fontsize=9,
            )
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("小时")
    axes[-1, 1].set_xlabel("小时")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("系统峰值窗口内各区域净购电轨迹", fontsize=15, fontweight="semibold", y=0.995)
    subtitle(fig, f"窗口：小时 {start}-{end}，以无储能 RegionF 峰值小时 {peak_hour} 为中心", y=0.967)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.6, w_pad=1.6)
    save_figure(fig, output)


def strategy_tradeoff(system: pd.DataFrame, output: Path) -> None:
    frame = system.loc[system["Scenario"].isin(SCENARIO_ORDER)].copy()
    frame["Scenario"] = pd.Categorical(frame["Scenario"], SCENARIO_ORDER, ordered=True)
    frame = frame.sort_values("Scenario")
    x = -frame["ElectricityCost_CNY"].to_numpy(float) / 1e8
    y = frame["MeanRegionalStd_MW"].to_numpy(float)
    color = frame["RenewableUtilization_pct"].to_numpy(float)
    throughput = frame["StorageThroughput_MWh"].to_numpy(float)
    sizes = 80 + 520 * throughput / max(float(throughput.max()), 1.0)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    scatter = ax.scatter(
        x,
        y,
        c=color,
        s=sizes,
        cmap="Blues",
        vmin=float(color.min()) - 1,
        vmax=float(color.max()) + 1,
        edgecolors=COLORS["ink"],
        linewidths=0.65,
        alpha=0.9,
    )
    lookup = {
        str(row["Scenario"]): (xi, yi, row["Scenario_CN"])
        for (_, row), xi, yi in zip(frame.iterrows(), x, y)
    }
    for scenario, offset in {
        "no_storage": (6, 8),
        "smooth_min": (7, 8),
        "balanced": (7, -16),
    }.items():
        xi, yi, label = lookup[scenario]
        ax.annotate(label, (xi, yi), xytext=offset, textcoords="offset points", fontsize=8.5)

    anchor_points = np.array(
        [[lookup[key][0], lookup[key][1]] for key in ["cost_min", "carbon_min", "peak_min"]]
    )
    anchor_xy = anchor_points.mean(axis=0)
    ax.annotate(
        "成本/碳排/削峰最优\n（三者近重合）",
        anchor_xy,
        xytext=(-145, 24),
        textcoords="offset points",
        fontsize=8.5,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "lw": 0.8},
    )
    ax.set_xlabel("净售电收益（亿元，数值越大越优）")
    ax.set_ylabel("区域净购电标准差均值（MW，越小越优）")
    ax.set_title("储能策略的收益-平滑性权衡", pad=40)
    subtitle(fig, "气泡大小表示储能总吞吐量，颜色表示新能源利用率；完整时域第 0-2406 小时")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("新能源利用率（%）")
    ax.set_xlim(float(x.min()) - 0.02, float(x.max()) + 0.08)
    ax.set_ylim(bottom=-1.0, top=float(y.max()) + 1.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, output)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    system = pd.read_csv(args.results_dir / "q3_system_metrics.csv")
    regional = pd.read_csv(args.results_dir / "q3_regional_metrics.csv")
    balanced = pd.read_csv(args.results_dir / "q3_schedule_balanced.csv")
    no_storage = pd.read_csv(args.results_dir / "q3_schedule_no_storage.csv")

    scenario_metric_comparison(system, args.figures_dir / "fig01_scenario_metrics.png")
    regional_balanced_impact(regional, args.figures_dir / "fig02_regional_impact.png")
    representative_week(balanced, no_storage, args.figures_dir / "fig03_regionf_typical_week.png")
    soc_heatmap(
        balanced,
        args.results_dir.parent / "data" / "storage_information.xlsx",
        args.figures_dir / "fig04_soc_heatmap.png",
    )
    regional_peak_window(balanced, no_storage, args.figures_dir / "fig05_peak_window.png")
    strategy_tradeoff(system, args.figures_dir / "fig06_strategy_tradeoff.png")


if __name__ == "__main__":
    main()
