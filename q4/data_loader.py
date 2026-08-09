from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_CANDIDATES, EXECUTION_END, POWER_HOURS, REGIONS, TASK_TYPES


def find_data_dir() -> Path:
    for p in DATA_CANDIDATES:
        if (p / "workload_trace.xlsx").exists():
            return p
    raise FileNotFoundError(f"Cannot find workload_trace.xlsx in {DATA_CANDIDATES}")


@dataclass
class Q4Data:
    tasks: pd.DataFrame
    gpu_info: pd.DataFrame
    latency: pd.DataFrame
    power: pd.Series
    region_hour: pd.DataFrame
    storage: pd.DataFrame
    # arrays indexed [region_idx, hour]
    price: np.ndarray
    sell_price: np.ndarray
    carbon: np.ndarray
    available_re: np.ndarray
    available_re_raw: np.ndarray
    non_ai: np.ndarray
    gpu_cap: np.ndarray
    max_it: np.ndarray
    pue: np.ndarray
    max_fac: np.ndarray
    # storage scalars per region
    storage_cap: np.ndarray
    min_soc: np.ndarray
    init_soc: np.ndarray
    max_charge: np.ndarray
    max_discharge: np.ndarray
    eta_c: np.ndarray
    eta_d: np.ndarray
    sell_limit: np.ndarray
    max_import: np.ndarray
    max_export: np.ndarray
    latency_map: dict
    eligible: dict


def load_data() -> Q4Data:
    data_dir = find_data_dir()
    tasks = pd.read_excel(data_dir / "workload_trace.xlsx", sheet_name=0)
    gpu_info = pd.read_excel(data_dir / "GPU_information.xlsx", sheet_name=0)
    latency_df = pd.read_excel(data_dir / "network_latency.xlsx", sheet_name=0)
    power_df = pd.read_excel(data_dir / "power_mapping.xlsx", sheet_name=0)
    region_hour = pd.read_excel(data_dir / "region_time_data.xlsx", sheet_name=0)
    storage = pd.read_excel(data_dir / "storage_information.xlsx", sheet_name=0)

    power = power_df.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"]
    tasks = tasks.copy()
    tasks["PowerPerGPU"] = tasks["TaskType"].map(power)

    info = gpu_info.set_index("Region").loc[REGIONS]
    st = storage.set_index("Region").loc[REGIONS]

    def pivot(col: str) -> np.ndarray:
        wide = region_hour.pivot(index="Hour", columns="Region", values=col)
        wide = wide.reindex(index=range(POWER_HOURS), columns=REGIONS)
        return wide.to_numpy(dtype=float).T  # [R, T]

    price = pivot("ElectricityPrice_CNY_per_MWh")
    sell_price = pivot("SellPrice_CNY_per_MWh")
    carbon = pivot("CarbonIntensity_tCO2_per_MWh")
    # Attachment AvailableRenewable_MW is identical across all six regions
    # (~800 MW mean). Using it per-region in the energy balance would 6×-count
    # system renewables and force near-zero grid purchase. The LP therefore uses
    # each region's baseline *deliverable* ceiling:
    #   UsedRenewable + RenewableCharge + GridSell
    # capped by attachment AvailableRenewable. Raw AvailableRenewable is kept
    # separately for reporting utilization (消纳 / 附件可用出力).
    available_raw = pivot("AvailableRenewable_MW")
    used = pivot("UsedRenewable_MW")
    re_ch = pivot("RenewableCharge_MW")
    sell = pivot("GridSell_MW")
    available_re = np.maximum(used + re_ch + sell, 0.0)
    available_re = np.minimum(available_re, available_raw)
    non_ai = pivot("NonAI_IT_Load_MW")

    latency_map = {
        (row.FromRegion, row.ToRegion): float(row.NetworkLatency_ms)
        for row in latency_df.itertuples(index=False)
    }
    eligible = {}
    for row in tasks.itertuples(index=False):
        eligible[int(row.TaskID)] = [
            r for r in REGIONS if latency_map[(row.SourceRegion, r)] <= float(row.MaxLatency_ms)
        ]

    return Q4Data(
        tasks=tasks,
        gpu_info=gpu_info,
        latency=latency_df,
        power=power,
        region_hour=region_hour,
        storage=storage,
        price=price,
        sell_price=sell_price,
        carbon=carbon,
        available_re=available_re,
        available_re_raw=available_raw,
        non_ai=non_ai,
        gpu_cap=info["Available_GPU"].to_numpy(float),
        max_it=info["Max_IT_Power_MW"].to_numpy(float),
        pue=info["PUE"].to_numpy(float),
        max_fac=info["Max_Facility_Power_MW"].to_numpy(float),
        storage_cap=st["StorageCapacity_MWh"].to_numpy(float),
        min_soc=st["MinSOC_MWh"].to_numpy(float),
        init_soc=st["InitialSOC_MWh"].to_numpy(float),
        max_charge=st["MaxChargePower_MW"].to_numpy(float),
        max_discharge=st["MaxDischargePower_MW"].to_numpy(float),
        eta_c=st["ChargeEfficiency"].to_numpy(float),
        eta_d=st["DischargeEfficiency"].to_numpy(float),
        sell_limit=st["SellLimit_MW"].to_numpy(float),
        max_import=st["MaxGridImport_MW"].to_numpy(float),
        max_export=st["MaxGridExport_MW"].to_numpy(float),
        latency_map=latency_map,
        eligible=eligible,
    )


def apply_price_mechanism(data: Q4Data, mechanism: str) -> Q4Data:
    """Return a shallow-copied data object with modified price arrays."""
    price = data.price.copy()
    sell = data.sell_price.copy()
    if mechanism == "baseline":
        pass
    elif mechanism == "peak_valley_amplify":
        # enlarge TOU spread around regional hourly mean
        mean = price.mean(axis=1, keepdims=True)
        price = mean + 1.6 * (price - mean)
        price = np.clip(price, 50.0, None)
        sell = sell * 1.1
    elif mechanism == "flat":
        flat = price.mean(axis=1, keepdims=True)
        price = np.repeat(flat, price.shape[1], axis=1)
    elif mechanism == "carbon_linked":
        # Price' = Price + kappa * carbon intensity
        kappa = 80.0  # CNY/tCO2
        price = price + kappa * data.carbon
    else:
        raise KeyError(mechanism)
    out = Q4Data(**{**data.__dict__, "price": price, "sell_price": sell})
    return out


def scale_renewables(data: Q4Data, factor: float) -> Q4Data:
    f = float(factor)
    return Q4Data(
        **{
            **data.__dict__,
            "available_re": data.available_re * f,
            "available_re_raw": data.available_re_raw * f,
        }
    )
