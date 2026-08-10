#!/usr/bin/env python3
"""华数杯 C 题第三问：区域储能协同多目标优化。

模型固定使用题目给定的逐时 AI/非 AI IT 负荷，不再调整任务迁移和开工时段。
每个区域建立连续线性规划，并以电费、碳排放、峰值净购电和净购电爬坡
总变差为四个目标。先求四个单目标锚点，再基于 payoff matrix 做等权归一化
折中，得到可复现的 Pareto 妥协解。

连续松弛没有显式充放电互斥二进制变量。由于电价非负、往返效率小于 1，
且目标中加入了极小的吞吐正则项，同时充放电不会改善任一主目标；程序还会
在结果校验中报告同时充放电量，作为松弛是否精确的证据。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack


STRATEGY_LABELS = {
    "attachment_baseline": "附件参考策略",
    "no_storage": "无储能反事实",
    "cost_min": "成本最优",
    "carbon_min": "碳排最优",
    "peak_min": "削峰最优",
    "smooth_min": "平滑最优",
    "balanced": "多目标折中",
}

ANCHOR_TO_STRATEGY = {
    "cost": "cost_min",
    "carbon": "carbon_min",
    "peak": "peak_min",
    "ramp": "smooth_min",
}

BALANCED_WEIGHTS = {
    "cost": 0.25,
    "carbon": 0.25,
    "peak": 0.25,
    "ramp": 0.25,
}


@dataclass(frozen=True)
class IndexBlocks:
    """线性规划变量分块索引。"""

    grid: np.ndarray
    sell: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    curtail: np.ndarray
    peak: int
    ramp: np.ndarray
    n_vars: int


@dataclass
class LPModel:
    """某区域完整时域的稀疏线性规划。"""

    A_eq: csr_matrix
    b_eq: np.ndarray
    A_ub: csr_matrix
    b_ub: np.ndarray
    bounds: list[tuple[float, float]]
    idx: IndexBlocks
    frame: pd.DataFrame
    storage: pd.Series


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="求解华数杯 C 题第三问储能协同优化")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data",
        help="包含 region_time_data.xlsx、storage_information.xlsx、GPU_information.xlsx 的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results",
        help="CSV 与 JSON 结果输出目录",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} 缺少字段: {missing}")


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取并验证第三问的三个必需工作簿。"""

    paths = {
        "region_time": data_dir / "region_time_data.xlsx",
        "storage": data_dir / "storage_information.xlsx",
        "gpu": data_dir / "GPU_information.xlsx",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件:\n" + "\n".join(missing))

    rtd = pd.read_excel(paths["region_time"], sheet_name="region_time_data")
    storage = pd.read_excel(paths["storage"], sheet_name="storage_information")
    gpu = pd.read_excel(paths["gpu"], sheet_name="GPU中心基础情况")

    require_columns(
        rtd,
        [
            "Hour",
            "Region",
            "ElectricityPrice_CNY_per_MWh",
            "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh",
            "AvailableRenewable_MW",
            "Baseline_AI_IT_Load_MW",
            "NonAI_IT_Load_MW",
            "GridPurchase_MW",
            "GridSell_MW",
            "Curtailment_MW",
            "SOC_MWh",
            "ChargePower_MW",
            "DischargePower_MW",
        ],
        "region_time_data.xlsx",
    )
    require_columns(
        storage,
        [
            "Region",
            "StorageCapacity_MWh",
            "MinSOC_MWh",
            "InitialSOC_MWh",
            "MaxChargePower_MW",
            "MaxDischargePower_MW",
            "ChargeEfficiency",
            "DischargeEfficiency",
            "MaxGridImport_MW",
            "MaxGridExport_MW",
        ],
        "storage_information.xlsx",
    )
    require_columns(gpu, ["Region", "PUE"], "GPU_information.xlsx")

    if rtd.isna().any().any() or storage.isna().any().any() or gpu[["Region", "PUE"]].isna().any().any():
        raise ValueError("第三问输入表存在空值，请先检查原始工作簿。")
    if rtd.duplicated(["Region", "Hour"]).any():
        raise ValueError("region_time_data 存在重复的 Region-Hour 记录。")

    expected_regions = set(storage["Region"])
    if set(rtd["Region"]) != expected_regions or set(gpu["Region"]) != expected_regions:
        raise ValueError("三个工作簿的区域集合不一致。")

    hours = np.sort(rtd["Hour"].unique())
    if not np.array_equal(hours, np.arange(0, 2407)):
        raise ValueError("逐时数据必须完整覆盖第 0-2406 小时。")
    counts = rtd.groupby("Region").size()
    if not (counts == 2407).all():
        raise ValueError(f"每个区域应有 2407 行，实际为: {counts.to_dict()}")

    pue = gpu.set_index("Region")["PUE"]
    rtd = rtd.copy()
    rtd["PUE"] = rtd["Region"].map(pue)
    rtd["FacilityLoad_MW"] = (
        rtd["Baseline_AI_IT_Load_MW"] + rtd["NonAI_IT_Load_MW"]
    ) * rtd["PUE"]
    rtd = rtd.sort_values(["Region", "Hour"]).reset_index(drop=True)
    storage = storage.sort_values("Region").reset_index(drop=True)

    if (rtd["AvailableRenewable_MW"] < -1e-9).any():
        raise ValueError("可用新能源出力不得为负。")
    if (rtd["ElectricityPrice_CNY_per_MWh"] < -1e-9).any():
        raise ValueError("本模型的连续松弛依赖非负购电价。")
    if (rtd["SellPrice_CNY_per_MWh"] > rtd["ElectricityPrice_CNY_per_MWh"] + 1e-9).any():
        raise ValueError("发现售电价高于购电价，需改用购售电互斥混合整数模型。")

    return rtd, storage


def make_indices(T: int) -> IndexBlocks:
    grid = np.arange(0, T)
    sell = np.arange(T, 2 * T)
    charge = np.arange(2 * T, 3 * T)
    discharge = np.arange(3 * T, 4 * T)
    soc = np.arange(4 * T, 5 * T)
    curtail = np.arange(5 * T, 6 * T)
    peak = 6 * T
    ramp = np.arange(6 * T + 1, 7 * T)
    return IndexBlocks(grid, sell, charge, discharge, soc, curtail, peak, ramp, 7 * T)


def build_lp(frame: pd.DataFrame, storage: pd.Series) -> LPModel:
    """构造一个区域完整时域的稀疏线性规划矩阵。"""

    frame = frame.sort_values("Hour").reset_index(drop=True).copy()
    T = len(frame)
    idx = make_indices(T)
    load = frame["FacilityLoad_MW"].to_numpy(float)
    renewable = frame["AvailableRenewable_MW"].to_numpy(float)
    eta_c = float(storage["ChargeEfficiency"])
    eta_d = float(storage["DischargeEfficiency"])
    initial = float(storage["InitialSOC_MWh"])

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq = np.zeros(2 * T)

    # 功率平衡: G + R + D = L + C + S + Curtailment.
    for t in range(T):
        row = t
        for column, value in (
            (idx.grid[t], 1.0),
            (idx.sell[t], -1.0),
            (idx.charge[t], -1.0),
            (idx.discharge[t], 1.0),
            (idx.curtail[t], -1.0),
        ):
            eq_rows.append(row)
            eq_cols.append(int(column))
            eq_data.append(value)
        b_eq[row] = load[t] - renewable[t]

    # 时段末 SOC: SOC_t = SOC_{t-1} + eta_c*C_t - D_t/eta_d.
    for t in range(T):
        row = T + t
        for column, value in (
            (idx.soc[t], 1.0),
            (idx.charge[t], -eta_c),
            (idx.discharge[t], 1.0 / eta_d),
        ):
            eq_rows.append(row)
            eq_cols.append(int(column))
            eq_data.append(value)
        if t == 0:
            b_eq[row] = initial
        else:
            eq_rows.append(row)
            eq_cols.append(int(idx.soc[t - 1]))
            eq_data.append(-1.0)

    A_eq = coo_matrix(
        (eq_data, (eq_rows, eq_cols)), shape=(2 * T, idx.n_vars)
    ).tocsr()

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    n_ub = T + 2 * (T - 1)
    b_ub = np.zeros(n_ub)

    # 区域峰值净购电变量: G_t - S_t <= P_peak.
    for t in range(T):
        row = t
        for column, value in (
            (idx.grid[t], 1.0),
            (idx.sell[t], -1.0),
            (idx.peak, -1.0),
        ):
            ub_rows.append(row)
            ub_cols.append(int(column))
            ub_data.append(value)

    # 净购电相邻时段绝对变化的线性化。
    for t in range(1, T):
        ramp_idx = idx.ramp[t - 1]
        row_pos = T + (t - 1)
        row_neg = T + (T - 1) + (t - 1)
        for column, value in (
            (idx.grid[t], 1.0),
            (idx.sell[t], -1.0),
            (idx.grid[t - 1], -1.0),
            (idx.sell[t - 1], 1.0),
            (ramp_idx, -1.0),
        ):
            ub_rows.append(row_pos)
            ub_cols.append(int(column))
            ub_data.append(value)
        for column, value in (
            (idx.grid[t], -1.0),
            (idx.sell[t], 1.0),
            (idx.grid[t - 1], 1.0),
            (idx.sell[t - 1], -1.0),
            (ramp_idx, -1.0),
        ):
            ub_rows.append(row_neg)
            ub_cols.append(int(column))
            ub_data.append(value)

    A_ub = coo_matrix(
        (ub_data, (ub_rows, ub_cols)), shape=(n_ub, idx.n_vars)
    ).tocsr()

    max_import = float(storage["MaxGridImport_MW"])
    max_export = float(storage["MaxGridExport_MW"])
    max_charge = float(storage["MaxChargePower_MW"])
    max_discharge = float(storage["MaxDischargePower_MW"])
    min_soc = float(storage["MinSOC_MWh"])
    capacity = float(storage["StorageCapacity_MWh"])

    bounds: list[tuple[float, float]] = []
    bounds.extend([(0.0, max_import)] * T)
    bounds.extend([(0.0, max_export)] * T)
    bounds.extend([(0.0, max_charge)] * T)
    bounds.extend([(0.0, max_discharge)] * T)
    bounds.extend([(min_soc, capacity)] * T)
    # 第 2406 小时为终端结算时点，SOC 不低于初始值。
    bounds[5 * T - 1] = (max(min_soc, initial), capacity)
    bounds.extend([(0.0, float(value)) for value in renewable])
    bounds.append((0.0, max_import))
    bounds.extend([(0.0, max_import + max_export)] * (T - 1))

    return LPModel(A_eq, b_eq, A_ub, b_ub, bounds, idx, frame, storage)


def metric_vector(model: LPModel, metric: str) -> np.ndarray:
    """返回某个评价指标对应的线性目标系数。"""

    vector = np.zeros(model.idx.n_vars)
    if metric == "cost":
        vector[model.idx.grid] = model.frame[
            "ElectricityPrice_CNY_per_MWh"
        ].to_numpy(float)
        vector[model.idx.sell] = -model.frame[
            "SellPrice_CNY_per_MWh"
        ].to_numpy(float)
    elif metric == "carbon":
        vector[model.idx.grid] = model.frame[
            "CarbonIntensity_tCO2_per_MWh"
        ].to_numpy(float)
    elif metric == "peak":
        vector[model.idx.peak] = 1.0
    elif metric == "ramp":
        vector[model.idx.ramp] = 1.0
    else:
        raise KeyError(metric)
    return vector


def objective_with_regularization(
    model: LPModel,
    components: Mapping[str, float],
    cycle_weight: float = 1e-7,
    cost_tie_weight: float = 0.0,
) -> np.ndarray:
    """组合已归一化的线性目标并加入极小吞吐正则。"""

    objective = np.zeros(model.idx.n_vars)
    for metric, coefficient in components.items():
        objective += coefficient * metric_vector(model, metric)

    max_power = max(
        float(model.storage["MaxChargePower_MW"]),
        float(model.storage["MaxDischargePower_MW"]),
        1.0,
    )
    T = len(model.frame)
    objective[model.idx.charge] += cycle_weight / (T * max_power)
    objective[model.idx.discharge] += cycle_weight / (T * max_power)

    if cost_tie_weight > 0:
        scale = max(
            float(
                np.sum(
                    np.abs(
                        model.frame["ElectricityPrice_CNY_per_MWh"].to_numpy(float)
                        * model.frame["FacilityLoad_MW"].to_numpy(float)
                    )
                )
            ),
            1.0,
        )
        objective += cost_tie_weight * metric_vector(model, "cost") / scale
    return objective


def solve_lp(model: LPModel, objective: np.ndarray, label: str) -> np.ndarray:
    start = time.time()
    result = linprog(
        c=objective,
        A_ub=model.A_ub,
        b_ub=model.b_ub,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=model.bounds,
        method="highs",
        options={"presolve": True},
    )
    elapsed = time.time() - start
    if not result.success:
        raise RuntimeError(f"{label} 求解失败: status={result.status}, {result.message}")
    print(f"  {label:<16s} solved in {elapsed:6.2f}s, iterations={result.nit}")
    return np.asarray(result.x)


def solve_lp_lexicographic(
    model: LPModel, primary_objective: np.ndarray, label: str
) -> np.ndarray:
    """按“主目标-电费-储能吞吐”三层优先级消除连续松弛退化解。"""

    primary_x = solve_lp(model, primary_objective, label)
    primary_value = float(primary_objective @ primary_x)
    tolerance = max(1e-10, abs(primary_value) * 1e-9)

    A_primary = vstack(
        [model.A_ub, csr_matrix(primary_objective.reshape(1, -1))]
    ).tocsr()
    b_primary = np.append(model.b_ub, primary_value + tolerance)

    cost_vector = metric_vector(model, "cost")
    cost_scale = max(
        float(
            np.sum(
                np.abs(
                    model.frame["ElectricityPrice_CNY_per_MWh"].to_numpy(float)
                    * model.frame["FacilityLoad_MW"].to_numpy(float)
                )
            )
        ),
        1.0,
    )
    normalized_cost = cost_vector / cost_scale

    start = time.time()
    cost_result = linprog(
        c=normalized_cost,
        A_ub=A_primary,
        b_ub=b_primary,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=model.bounds,
        method="highs",
        options={"presolve": True},
    )
    elapsed = time.time() - start
    if not cost_result.success:
        raise RuntimeError(
            f"{label} 二阶段成本去退化求解失败: "
            f"status={cost_result.status}, {cost_result.message}"
        )
    print(
        f"  {'  + min cost':<16s} solved in {elapsed:6.2f}s, "
        f"iterations={cost_result.nit}"
    )
    cost_value = float(normalized_cost @ cost_result.x)
    cost_tolerance = max(1e-10, abs(cost_value) * 1e-9)
    A_ub = vstack(
        [A_primary, csr_matrix(normalized_cost.reshape(1, -1))]
    ).tocsr()
    b_ub = np.append(b_primary, cost_value + cost_tolerance)

    T = len(model.frame)
    max_power = max(
        float(model.storage["MaxChargePower_MW"]),
        float(model.storage["MaxDischargePower_MW"]),
        1.0,
    )
    secondary = np.zeros(model.idx.n_vars)
    secondary[model.idx.charge] = 1.0 / (T * max_power)
    secondary[model.idx.discharge] = 1.0 / (T * max_power)

    start = time.time()
    result = linprog(
        c=secondary,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=model.bounds,
        method="highs",
        options={"presolve": True},
    )
    elapsed = time.time() - start
    if not result.success:
        raise RuntimeError(
            f"{label} 三阶段吞吐去退化求解失败: status={result.status}, {result.message}"
        )
    print(
        f"  {'  + min throughput':<16s} solved in {elapsed:6.2f}s, "
        f"iterations={result.nit}"
    )
    return np.asarray(result.x)


def decode_solution(model: LPModel, x: np.ndarray, scenario: str) -> pd.DataFrame:
    """把优化向量还原为逐时调度表，并按优先级分解新能源用途。"""

    frame = model.frame.copy()
    idx = model.idx
    grid = np.maximum(x[idx.grid], 0.0)
    sell = np.maximum(x[idx.sell], 0.0)
    charge = np.maximum(x[idx.charge], 0.0)
    discharge = np.maximum(x[idx.discharge], 0.0)
    soc = x[idx.soc]
    curtail = np.maximum(x[idx.curtail], 0.0)
    renewable = frame["AvailableRenewable_MW"].to_numpy(float)
    load = frame["FacilityLoad_MW"].to_numpy(float)
    renewable_used_total = np.maximum(renewable - curtail, 0.0)

    # 仅用于结果解释的能量归因：先供负荷，再充电，最后外送。
    direct_renewable = np.minimum(load, renewable_used_total)
    remainder = np.maximum(renewable_used_total - direct_renewable, 0.0)
    renewable_charge = np.minimum(charge, remainder)
    grid_charge = np.maximum(charge - renewable_charge, 0.0)

    result = pd.DataFrame(
        {
            "Scenario": scenario,
            "Scenario_CN": STRATEGY_LABELS[scenario],
            "Hour": frame["Hour"].to_numpy(int),
            "Region": frame["Region"].to_numpy(),
            "FacilityLoad_MW": load,
            "AvailableRenewable_MW": renewable,
            "ElectricityPrice_CNY_per_MWh": frame[
                "ElectricityPrice_CNY_per_MWh"
            ].to_numpy(float),
            "SellPrice_CNY_per_MWh": frame[
                "SellPrice_CNY_per_MWh"
            ].to_numpy(float),
            "CarbonIntensity_tCO2_per_MWh": frame[
                "CarbonIntensity_tCO2_per_MWh"
            ].to_numpy(float),
            "GridPurchase_MW": grid,
            "GridSell_MW": sell,
            "NetGridImport_MW": grid - sell,
            "ChargePower_MW": charge,
            "DischargePower_MW": discharge,
            "SOC_MWh": soc,
            "Curtailment_MW": curtail,
            "RenewableUsedTotal_MW": renewable_used_total,
            "DirectRenewable_MW": direct_renewable,
            "RenewableCharge_MW": renewable_charge,
            "GridCharge_MW": grid_charge,
        }
    )
    return result


def make_no_storage_schedule(
    rtd: pd.DataFrame, storage: pd.DataFrame
) -> pd.DataFrame:
    """构造严格功率平衡且不使用储能的反事实基线。"""

    storage_map = storage.set_index("Region")
    parts = []
    for region, group in rtd.groupby("Region", sort=True):
        group = group.sort_values("Hour")
        load = group["FacilityLoad_MW"].to_numpy(float)
        renewable = group["AvailableRenewable_MW"].to_numpy(float)
        surplus = np.maximum(renewable - load, 0.0)
        deficit = np.maximum(load - renewable, 0.0)
        export_limit = float(storage_map.loc[region, "MaxGridExport_MW"])
        grid_sell = np.minimum(surplus, export_limit)
        curtail = np.maximum(surplus - grid_sell, 0.0)
        initial = float(storage_map.loc[region, "InitialSOC_MWh"])
        T = len(group)
        parts.append(
            pd.DataFrame(
                {
                    "Scenario": "no_storage",
                    "Scenario_CN": STRATEGY_LABELS["no_storage"],
                    "Hour": group["Hour"].to_numpy(int),
                    "Region": region,
                    "FacilityLoad_MW": load,
                    "AvailableRenewable_MW": renewable,
                    "ElectricityPrice_CNY_per_MWh": group[
                        "ElectricityPrice_CNY_per_MWh"
                    ].to_numpy(float),
                    "SellPrice_CNY_per_MWh": group[
                        "SellPrice_CNY_per_MWh"
                    ].to_numpy(float),
                    "CarbonIntensity_tCO2_per_MWh": group[
                        "CarbonIntensity_tCO2_per_MWh"
                    ].to_numpy(float),
                    "GridPurchase_MW": deficit,
                    "GridSell_MW": grid_sell,
                    "NetGridImport_MW": deficit - grid_sell,
                    "ChargePower_MW": np.zeros(T),
                    "DischargePower_MW": np.zeros(T),
                    "SOC_MWh": np.full(T, initial),
                    "Curtailment_MW": curtail,
                    "RenewableUsedTotal_MW": renewable - curtail,
                    "DirectRenewable_MW": np.minimum(load, renewable),
                    "RenewableCharge_MW": np.zeros(T),
                    "GridCharge_MW": np.zeros(T),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def make_attachment_schedule(rtd: pd.DataFrame) -> pd.DataFrame:
    """把附件基准字段重排成与优化结果一致的比较表。"""

    charge = rtd["ChargePower_MW"].to_numpy(float)
    renewable_charge = rtd.get("RenewableCharge_MW", pd.Series(np.zeros(len(rtd)))).to_numpy(float)
    return pd.DataFrame(
        {
            "Scenario": "attachment_baseline",
            "Scenario_CN": STRATEGY_LABELS["attachment_baseline"],
            "Hour": rtd["Hour"].to_numpy(int),
            "Region": rtd["Region"].to_numpy(),
            "FacilityLoad_MW": rtd["FacilityLoad_MW"].to_numpy(float),
            "AvailableRenewable_MW": rtd["AvailableRenewable_MW"].to_numpy(float),
            "ElectricityPrice_CNY_per_MWh": rtd[
                "ElectricityPrice_CNY_per_MWh"
            ].to_numpy(float),
            "SellPrice_CNY_per_MWh": rtd["SellPrice_CNY_per_MWh"].to_numpy(float),
            "CarbonIntensity_tCO2_per_MWh": rtd[
                "CarbonIntensity_tCO2_per_MWh"
            ].to_numpy(float),
            "GridPurchase_MW": rtd["GridPurchase_MW"].to_numpy(float),
            "GridSell_MW": rtd["GridSell_MW"].to_numpy(float),
            "NetGridImport_MW": (
                rtd["GridPurchase_MW"] - rtd["GridSell_MW"]
            ).to_numpy(float),
            "ChargePower_MW": charge,
            "DischargePower_MW": rtd["DischargePower_MW"].to_numpy(float),
            "SOC_MWh": rtd["SOC_MWh"].to_numpy(float),
            "Curtailment_MW": rtd["Curtailment_MW"].to_numpy(float),
            "RenewableUsedTotal_MW": (
                rtd["AvailableRenewable_MW"] - rtd["Curtailment_MW"]
            ).to_numpy(float),
            "DirectRenewable_MW": rtd.get(
                "UsedRenewable_MW", pd.Series(np.zeros(len(rtd)))
            ).to_numpy(float),
            "RenewableCharge_MW": renewable_charge,
            "GridCharge_MW": np.maximum(charge - renewable_charge, 0.0),
        }
    )


def linear_objective_values(schedule: pd.DataFrame) -> Dict[str, float]:
    net = schedule["NetGridImport_MW"].to_numpy(float)
    return {
        "cost": float(
            np.dot(
                schedule["ElectricityPrice_CNY_per_MWh"],
                schedule["GridPurchase_MW"],
            )
            - np.dot(
                schedule["SellPrice_CNY_per_MWh"], schedule["GridSell_MW"]
            )
        ),
        "carbon": float(
            np.dot(
                schedule["CarbonIntensity_tCO2_per_MWh"],
                schedule["GridPurchase_MW"],
            )
        ),
        "peak": float(max(0.0, np.max(net))),
        "ramp": float(np.sum(np.abs(np.diff(net)))),
    }


def anchor_objective(model: LPModel, anchor: str, no_storage: pd.DataFrame) -> np.ndarray:
    base = linear_objective_values(no_storage)
    scale = max(abs(base[anchor]), 1.0)
    components = {anchor: 1.0 / scale}
    tie = 0.0 if anchor == "cost" else 1e-7
    return objective_with_regularization(model, components, cost_tie_weight=tie)


def solve_region(
    frame: pd.DataFrame,
    storage_row: pd.Series,
    no_storage: pd.DataFrame,
) -> tuple[Dict[str, pd.DataFrame], list[dict[str, float | str]]]:
    region = str(storage_row.name)
    print(f"\n[{region}] building {len(frame)}-hour sparse LP")
    model = build_lp(frame, storage_row)
    schedules: Dict[str, pd.DataFrame] = {}
    payoff_rows: list[dict[str, float | str]] = []

    for anchor, strategy in ANCHOR_TO_STRATEGY.items():
        x = solve_lp_lexicographic(
            model, anchor_objective(model, anchor, no_storage), strategy
        )
        schedule = decode_solution(model, x, strategy)
        schedules[strategy] = schedule
        values = linear_objective_values(schedule)
        payoff_rows.append(
            {
                "Region": region,
                "AnchorStrategy": strategy,
                "AnchorStrategy_CN": STRATEGY_LABELS[strategy],
                "ElectricityCost_CNY": values["cost"],
                "CarbonEmission_tCO2": values["carbon"],
                "PeakNetGridImport_MW": values["peak"],
                "TotalAbsRamp_MW": values["ramp"],
            }
        )

    payoff = {
        strategy: linear_objective_values(schedule)
        for strategy, schedule in schedules.items()
    }
    balanced_components: Dict[str, float] = {}
    for metric, weight in BALANCED_WEIGHTS.items():
        values = [item[metric] for item in payoff.values()]
        spread = max(values) - min(values)
        if spread > 1e-8:
            balanced_components[metric] = weight / spread

    active_weight = sum(
        BALANCED_WEIGHTS[metric] for metric in balanced_components
    )
    if active_weight <= 0:
        balanced_components = {
            "cost": 1.0 / max(abs(linear_objective_values(no_storage)["cost"]), 1.0)
        }
    else:
        balanced_components = {
            metric: coefficient / active_weight
            for metric, coefficient in balanced_components.items()
        }

    balanced_objective = objective_with_regularization(
        model, balanced_components, cost_tie_weight=1e-8
    )
    x_balanced = solve_lp_lexicographic(model, balanced_objective, "balanced")
    schedules["balanced"] = decode_solution(model, x_balanced, "balanced")
    return schedules, payoff_rows


def check_schedule(
    schedule: pd.DataFrame, storage: pd.DataFrame
) -> Dict[str, float]:
    """返回功率平衡、SOC 递推和互斥松弛的最大残差。"""

    storage_map = storage.set_index("Region")
    energy_residuals: list[float] = []
    soc_residuals: list[float] = []
    simultaneous: list[float] = []
    simultaneous_buy_sell: list[float] = []
    terminal_shortfalls: list[float] = []

    for region, group in schedule.groupby("Region", sort=True):
        group = group.sort_values("Hour")
        st = storage_map.loc[region]
        energy = (
            group["GridPurchase_MW"]
            + group["AvailableRenewable_MW"]
            + group["DischargePower_MW"]
            - group["FacilityLoad_MW"]
            - group["ChargePower_MW"]
            - group["GridSell_MW"]
            - group["Curtailment_MW"]
        )
        energy_residuals.extend(energy.to_numpy(float))

        previous = float(st["InitialSOC_MWh"])
        eta_c = float(st["ChargeEfficiency"])
        eta_d = float(st["DischargeEfficiency"])
        for row in group.itertuples(index=False):
            expected = previous + eta_c * row.ChargePower_MW - row.DischargePower_MW / eta_d
            soc_residuals.append(row.SOC_MWh - expected)
            previous = row.SOC_MWh
            simultaneous.append(min(row.ChargePower_MW, row.DischargePower_MW))
            simultaneous_buy_sell.append(min(row.GridPurchase_MW, row.GridSell_MW))
        terminal_shortfalls.append(max(float(st["InitialSOC_MWh"]) - previous, 0.0))

    return {
        "MaxAbsEnergyBalanceResidual_MW": float(np.max(np.abs(energy_residuals))),
        "MaxAbsSOCDynamicsResidual_MWh": float(np.max(np.abs(soc_residuals))),
        "MaxSimultaneousChargeDischarge_MW": float(np.max(simultaneous)),
        "MaxSimultaneousBuySell_MW": float(np.max(simultaneous_buy_sell)),
        "MaxTerminalSOCShortfall_MWh": float(np.max(terminal_shortfalls)),
    }


def regional_metrics(
    schedule: pd.DataFrame, storage: pd.DataFrame
) -> pd.DataFrame:
    storage_map = storage.set_index("Region")
    rows = []
    for region, group in schedule.groupby("Region", sort=True):
        group = group.sort_values("Hour")
        st = storage_map.loc[region]
        net = group["NetGridImport_MW"].to_numpy(float)
        renewable_total = float(group["AvailableRenewable_MW"].sum())
        electricity_cost = float(
            np.dot(
                group["ElectricityPrice_CNY_per_MWh"], group["GridPurchase_MW"]
            )
            - np.dot(group["SellPrice_CNY_per_MWh"], group["GridSell_MW"])
        )
        carbon = float(
            np.dot(
                group["CarbonIntensity_tCO2_per_MWh"], group["GridPurchase_MW"]
            )
        )
        throughput = float(
            (group["ChargePower_MW"] + group["DischargePower_MW"]).sum()
        )
        initial = float(st["InitialSOC_MWh"])
        terminal = float(group["SOC_MWh"].iloc[-1])
        rows.append(
            {
                "Scenario": group["Scenario"].iloc[0],
                "Scenario_CN": group["Scenario_CN"].iloc[0],
                "Region": region,
                "ElectricityCost_CNY": electricity_cost,
                "CarbonEmission_tCO2": carbon,
                "PeakNetGridImport_MW": float(max(0.0, np.max(net))),
                "StdNetGridImport_MW": float(np.std(net, ddof=0)),
                "MeanAbsRamp_MW": float(np.mean(np.abs(np.diff(net)))),
                "MaxAbsRamp_MW": float(np.max(np.abs(np.diff(net)))),
                "RenewableUtilization_pct": 100.0
                * (1.0 - float(group["Curtailment_MW"].sum()) / renewable_total),
                "StorageThroughput_MWh": throughput,
                "EquivalentFullCycles": throughput
                / (2.0 * float(st["StorageCapacity_MWh"])),
                "InitialSOC_MWh": initial,
                "TerminalSOC_MWh": terminal,
                "TerminalSOCFeasible": bool(terminal >= initial - 1e-5),
            }
        )
    return pd.DataFrame(rows)


def system_metrics(
    schedule: pd.DataFrame, storage: pd.DataFrame
) -> dict[str, float | str | bool]:
    region_table = regional_metrics(schedule, storage)
    by_hour = schedule.groupby("Hour", as_index=False).agg(
        NetGridImport_MW=("NetGridImport_MW", "sum"),
        GridPurchase_MW=("GridPurchase_MW", "sum"),
        GridSell_MW=("GridSell_MW", "sum"),
        AvailableRenewable_MW=("AvailableRenewable_MW", "sum"),
        Curtailment_MW=("Curtailment_MW", "sum"),
    )
    net = by_hour["NetGridImport_MW"].to_numpy(float)
    scenario = str(schedule["Scenario"].iloc[0])
    return {
        "Scenario": scenario,
        "Scenario_CN": STRATEGY_LABELS[scenario],
        "ElectricityCost_CNY": float(region_table["ElectricityCost_CNY"].sum()),
        "CarbonEmission_tCO2": float(region_table["CarbonEmission_tCO2"].sum()),
        "SumRegionalPeak_MW": float(region_table["PeakNetGridImport_MW"].sum()),
        "SystemCoincidentPeak_MW": float(max(0.0, np.max(net))),
        "SystemNetImportStd_MW": float(np.std(net, ddof=0)),
        "SystemMeanAbsRamp_MW": float(np.mean(np.abs(np.diff(net)))),
        "MeanRegionalStd_MW": float(region_table["StdNetGridImport_MW"].mean()),
        "MeanRegionalAbsRamp_MW": float(region_table["MeanAbsRamp_MW"].mean()),
        "RenewableUtilization_pct": 100.0
        * (
            1.0
            - float(by_hour["Curtailment_MW"].sum())
            / float(by_hour["AvailableRenewable_MW"].sum())
        ),
        "StorageThroughput_MWh": float(region_table["StorageThroughput_MWh"].sum()),
        "AllTerminalSOCFeasible": bool(region_table["TerminalSOCFeasible"].all()),
    }


def add_reductions(system_table: pd.DataFrame) -> pd.DataFrame:
    """相对无储能反事实计算主指标降幅；成本允许为负，仍按绝对差报告。"""

    table = system_table.copy()
    base = table.loc[table["Scenario"] == "no_storage"].iloc[0]
    mapping = {
        "ElectricityCost_CNY": "CostReduction_pct",
        "CarbonEmission_tCO2": "CarbonReduction_pct",
        "SumRegionalPeak_MW": "RegionalPeakReduction_pct",
        "MeanRegionalStd_MW": "RegionalFluctuationReduction_pct",
        "MeanRegionalAbsRamp_MW": "RegionalRampReduction_pct",
    }
    for source, target in mapping.items():
        denominator = abs(float(base[source]))
        if denominator < 1e-12:
            table[target] = np.nan
        else:
            table[target] = 100.0 * (float(base[source]) - table[source]) / denominator
    return table


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("华数杯 C 题第三问：储能协同多目标优化")
    print(f"data   : {args.data_dir.resolve()}")
    print(f"output : {args.output_dir.resolve()}")
    print("=" * 72)

    rtd, storage = load_inputs(args.data_dir)
    no_storage = make_no_storage_schedule(rtd, storage)
    attachment = make_attachment_schedule(rtd)

    scenario_parts: Dict[str, list[pd.DataFrame]] = {
        "cost_min": [],
        "carbon_min": [],
        "peak_min": [],
        "smooth_min": [],
        "balanced": [],
    }
    payoff_rows: list[dict[str, float | str]] = []
    storage_map = storage.set_index("Region")

    for region, frame in rtd.groupby("Region", sort=True):
        region_no_storage = no_storage.loc[no_storage["Region"] == region].copy()
        schedules, region_payoff = solve_region(
            frame,
            storage_map.loc[region],
            region_no_storage,
        )
        for strategy, schedule in schedules.items():
            scenario_parts[strategy].append(schedule)
        payoff_rows.extend(region_payoff)

    scenario_schedules: Dict[str, pd.DataFrame] = {
        "attachment_baseline": attachment,
        "no_storage": no_storage,
    }
    for strategy, parts in scenario_parts.items():
        scenario_schedules[strategy] = pd.concat(parts, ignore_index=True).sort_values(
            ["Region", "Hour"]
        )

    # 可行性验证。附件基准允许终端 SOC 不满足题目新增约束，但必须显式报告。
    check_rows = []
    for scenario, schedule in scenario_schedules.items():
        check = check_schedule(schedule, storage)
        check_rows.append({"Scenario": scenario, "Scenario_CN": STRATEGY_LABELS[scenario], **check})
        print(
            f"check {scenario:<20s} "
            f"energy={check['MaxAbsEnergyBalanceResidual_MW']:.3e}, "
            f"soc={check['MaxAbsSOCDynamicsResidual_MWh']:.3e}, "
            f"simult={check['MaxSimultaneousChargeDischarge_MW']:.3e}, "
            f"buy_sell={check['MaxSimultaneousBuySell_MW']:.3e}, "
            f"terminal_shortfall={check['MaxTerminalSOCShortfall_MWh']:.3f}"
        )

    regional_tables = [
        regional_metrics(schedule, storage)
        for schedule in scenario_schedules.values()
    ]
    regional_table = pd.concat(regional_tables, ignore_index=True)
    system_table = pd.DataFrame(
        [system_metrics(schedule, storage) for schedule in scenario_schedules.values()]
    )
    system_table = add_reductions(system_table)

    payoff_table = pd.DataFrame(payoff_rows)
    checks_table = pd.DataFrame(check_rows)

    # 只交付主策略及基线逐时明细；所有单目标策略保留汇总指标和 payoff matrix。
    scenario_schedules["balanced"].to_csv(
        args.output_dir / "q3_schedule_balanced.csv", index=False
    )
    scenario_schedules["no_storage"].to_csv(
        args.output_dir / "q3_schedule_no_storage.csv", index=False
    )
    scenario_schedules["attachment_baseline"].to_csv(
        args.output_dir / "q3_schedule_attachment_baseline.csv", index=False
    )
    system_table.to_csv(args.output_dir / "q3_system_metrics.csv", index=False)
    regional_table.to_csv(args.output_dir / "q3_regional_metrics.csv", index=False)
    payoff_table.to_csv(args.output_dir / "q3_payoff_matrix.csv", index=False)
    checks_table.to_csv(args.output_dir / "q3_feasibility_checks.csv", index=False)

    manifest = {
        "model": "continuous linear multi-objective storage dispatch",
        "time_horizon": [0, 2406],
        "hours": 2407,
        "regions": sorted(rtd["Region"].unique().tolist()),
        "balanced_weights": BALANCED_WEIGHTS,
        "normalization": "per-region payoff-matrix min-max spread",
        "power_balance": "GridPurchase + Renewable + Discharge = FacilityLoad + Charge + GridSell + Curtailment",
        "soc_terminal_rule": "SOC_2406 >= InitialSOC",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (args.output_dir / "q3_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display_columns = [
        "Scenario_CN",
        "ElectricityCost_CNY",
        "CarbonEmission_tCO2",
        "SumRegionalPeak_MW",
        "MeanRegionalStd_MW",
        "RenewableUtilization_pct",
        "CostReduction_pct",
        "CarbonReduction_pct",
        "RegionalPeakReduction_pct",
        "RegionalFluctuationReduction_pct",
        "AllTerminalSOCFeasible",
    ]
    print("\nSystem comparison")
    print(system_table[display_columns].round(4).to_string(index=False))
    print(f"\nResults written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
