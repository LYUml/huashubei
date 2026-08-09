#!/usr/bin/env python3
"""
贪心调度结果输出脚本
==============================================
功能：
  对每一种贪婪权重配置，独立运行贪心调度算法，
  并将每种方案的【完整任务级调度表】和【汇总指标】
  分别输出为 CSV 文件。

输出文件（位于 output_greedy/ 目录下）：
  greedy_{name}_schedule.csv   —— 每个任务的详细调度结果
  greedy_{name}_metrics.csv    —— 该方案的四条核心指标
  greedy_all_compare.csv       —— 所有方案横向对比总表
  greedy_all_schedules.csv     —— 所有方案合并的完整调度表

运行前提：
  确保 preprocessed_data.pkl 与本脚本在同一目录的 output/ 子目录下。
"""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 路径设置 ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR = SCRIPT_DIR / "output_greedy"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(2024)

# ============================================================
# 1. 加载预处理数据
# ============================================================
print("=" * 60)
print("Step 1: 加载预处理数据")
print("=" * 60)

pkl_path = DATA_DIR / "preprocessed_data.pkl"
with open(pkl_path, "rb") as f:
    D = pickle.load(f)

workload = D["workload"]
regions = D["regions"]
task_types = D["task_types"]
rtd = D["region_time"]
latency_matrix = D["latency_matrix"]
power_dict = D["power_dict"]
region_info = D["region_info"]

N = len(workload)
print(f"  任务总数 : {N}")
print(f"  区域     : {regions}")
print(f"  任务类型 : {task_types}")

# 电价/碳强度/新能源查找表  {(region, hour) -> dict}
pp = {}
for _, row in rtd.iterrows():
    r, h = row["Region"], int(row["Hour"])
    pp[(r, h)] = {
        "price": float(row["Price_Yuan_per_MWh"]),
        "carbon": float(row["CarbonIntensity_tCO2_per_MWh"]),
        "renew": float(row["AvailableRenewable_MW"]),
        "nonai": float(row["NonAI_IT_Load_MW"]),
    }

# 区域容量参数
rc = {}
for r, info in region_info.items():
    rc[r] = {
        "gpu": int(info["Available_GPU"]),
        "max_it": float(info["Max_IT_Power_MW"]),
        "max_fac": float(info["Max_Facility_Power_MW"]),
        "pue": float(info["PUE"]),
    }

priority = {"RealTimeInference": 0, "BatchInference": 1, "AITraining": 2}

total_renewable = sum(v["renew"] for v in pp.values())
print(f"  全时域可用新能源总量: {total_renewable:.0f} MW·h")


# ============================================================
# 2. 辅助函数
# ============================================================
def dur_h(row):
    """将分钟级预估时长转换为小时数（向上取整）"""
    return int(np.ceil(row["EstimatedDuration_min"] / 60))


def compute_all_metrics(sched):
    """
    根据调度字典计算四项目标指标
    返回: (cost, carbon, avg_latency, renew_util, n_migrated, n_tasks)
    """
    cost = carbon = renew_used = total_fac = 0.0
    lat_sum = 0.0
    mig_n = 0
    n = len(sched)

    for tid, info in sched.items():
        r, s, e = info["region"], info["start"], info["end"]
        fac = info["fac_pw"]
        for h in range(s, e + 1):
            p = pp.get((r, h), {"price": 0, "carbon": 0, "renew": 0})
            cost += fac * p["price"]
            carbon += fac * p["carbon"]
            renew_used += min(p["renew"], fac)
            total_fac += fac
        if info["migrated"]:
            mig_n += 1
            try:
                lat_sum += float(latency_matrix.loc[info["source"], r])
            except Exception:
                pass

    renew_util = renew_used / total_renewable if total_renewable > 0 else 0
    avg_lat = lat_sum / max(mig_n, 1)
    return cost, carbon, avg_lat, renew_util, mig_n, n


# ============================================================
# 3. 贪心调度核心算法
# ============================================================
def greedy_schedule(workload_df, alpha):
    """
    加权贪心调度器
    --------------------
    alpha = (a_cost, a_carbon, a_latency, a_renew)
    每一步为当前任务在所有 (区域 × 起始时刻) 组合中，
    选择使加权综合代价最小的方案。
    """
    a1, a2, a3, a4 = alpha

    df = workload_df.copy()
    df["_pri"] = df["TaskType"].map(priority)
    df = df.sort_values(["_pri", "ArrivalHour"]).reset_index(drop=True)

    # 按 (区域, 小时) 记录已用资源
    gpu_use = {r: defaultdict(float) for r in regions}
    it_use = {r: defaultdict(float) for r in regions}
    sched = {}

    for idx, task in df.iterrows():
        tid = int(task["TaskID"])
        src = task["SourceRegion"]
        gpu = float(task["GPU_Demand"])
        dur = dur_h(task)
        maxlat = float(task["MaxLatency_ms"])
        tt = task["TaskType"]
        arr = int(task["ArrivalHour"])
        lfh = int(task["LatestFinishHour"])
        it_pw = gpu * float(power_dict.get(tt, 0.003))

        # ---- 构建可行起始时刻列表 ----
        if tt == "RealTimeInference":
            start_options = [arr]
        else:
            latest_start = min(lfh - dur, 2405 - dur)
            max_delay = max(0, latest_start - arr)
            max_delay = min(max_delay, 72)
            start_options = list(range(arr, arr + max_delay + 1))

        # ---- 构建候选区域列表（满足时延约束）----
        cand = [
            r
            for r in regions
            if r == src or float(latency_matrix.loc[src, r]) <= maxlat
        ]
        if not cand:
            cand = [src]

        # ---- 枚举 (区域, 起始时刻)，找最优 ----
        best = None
        best_score = float("inf")

        for r in cand:
            cap_gpu = rc[r]["gpu"]
            cap_it = rc[r]["max_it"]
            pue = rc[r]["pue"]

            for s in start_options:
                e = s + dur - 1
                if e >= 2406:
                    continue

                # 容量可行性检查
                ok = True
                for h in range(s, e + 1):
                    if gpu_use[r][h] + gpu > cap_gpu or it_use[r][h] + it_pw > cap_it:
                        ok = False
                        break
                if not ok:
                    continue

                # 计算该组合的代价
                cost = carbon = renew = 0.0
                for h in range(s, e + 1):
                    p = pp.get((r, h), {"price": 0, "carbon": 0, "renew": 0})
                    fac = it_pw * pue
                    cost += fac * p["price"]
                    carbon += fac * p["carbon"]
                    renew += min(p["renew"], fac)

                lat_pen = 0.0
                if src != r:
                    try:
                        lat_pen = float(latency_matrix.loc[src, r]) * 0.001
                    except Exception:
                        pass

                # 加权综合得分（越小越好）
                sc = a1 * cost + a2 * carbon + a3 * lat_pen - a4 * renew
                if sc < best_score:
                    best_score = sc
                    best = (r, s, e)

        # ---- 兜底：若全部不可行，则延迟到本区域有空位 ----
        if best is None:
            r = src
            s = arr
            attempts = 0
            while attempts < 500:
                e = s + dur - 1
                if e >= 2406:
                    s = max(0, 2405 - dur + 1)
                    e = 2405
                    break
                ok = all(
                    gpu_use[r].get(h, 0) + gpu <= rc[r]["gpu"] for h in range(s, e + 1)
                )
                if ok:
                    break
                s += 1
                attempts += 1
            e = s + dur - 1
            if e >= 2406:
                s = max(0, 2405 - dur + 1)
                e = 2405
            best = (r, s, e)

        # ---- 写入调度结果并更新资源占用 ----
        r, s, e = best
        fac_pw = it_pw * rc[r]["pue"]
        sched[tid] = {
            "region": r,
            "start": s,
            "end": e,
            "dur": dur,
            "gpu": gpu,
            "task_type": tt,
            "source": src,
            "it_pw": it_pw,
            "fac_pw": fac_pw,
            "migrated": (r != src),
        }
        for h in range(s, e + 1):
            gpu_use[r][h] += gpu
            it_use[r][h] += it_pw

    return sched


# ============================================================
# 4. 定义权重配置并依次运行
# ============================================================
weight_configs = {
    "balanced": (1.0, 1.0, 0.5, 0.5),  # 均衡
    "cost": (1.0, 0.1, 0.1, 0.0),  # 侧重成本
    "carbon": (0.1, 1.0, 0.1, 0.0),  # 侧重碳排放
    "renew": (0.1, 0.1, 0.1, 1.0),  # 侧重新能源
    "latency": (0.1, 0.1, 1.0, 0.0),  # 侧重时延
    "cost2": (2.0, 0.5, 0.3, 0.2),  # 成本优先混合
    "bal2": (0.5, 2.0, 0.5, 0.3),  # 碳优先混合
    "mix": (1.5, 1.5, 0.8, 0.6),  # 全面混合
}

print("\n" + "=" * 60)
print("Step 2: 依次运行 8 组贪心权重配置")
print("=" * 60)

all_schedules = {}  # {name: sched_dict}
all_metrics = []  # 汇总指标列表
all_rows = []  # 合并调度表行列表

# 打印表头
print(
    f"\n{'方案':<22s} {'成本(元)':>14s} {'碳(tCO2)':>12s} "
    f"{'时延(ms)':>10s} {'新能源利用':>10s} {'迁移数':>6s}"
)
print("-" * 82)

for name, w in weight_configs.items():
    scheme_name = f"greedy_{name}"
    print(f"  ▶ 正在运行 {scheme_name} ...", end="", flush=True)

    sched = greedy_schedule(workload, w)
    c, co, al, ru, mn, nt = compute_all_metrics(sched)

    all_schedules[name] = sched
    all_metrics.append(
        {
            "scheme": scheme_name,
            "cost_yuan": c,
            "carbon_tco2": co,
            "avg_latency_ms": al,
            "renew_utilization": ru,
            "migrated": mn,
            "n_tasks": nt,
            "weights": str(w),
        }
    )

    print(
        f"\r  ✅ {scheme_name:<20s} {c:>14.0f} {co:>12.2f} "
        f"{al:>10.1f} {ru:>10.4f} {mn:>6d}"
    )

    # ---- 输出该方案的逐任务调度表 ----
    rows = []
    for tid, info in sched.items():
        rows.append(
            {
                "TaskID": tid,
                "TaskType": info["task_type"],
                "SourceRegion": info["source"],
                "AssignedRegion": info["region"],
                "StartHour": info["start"],
                "EndHour": info["end"],
                "Duration_h": info["dur"],
                "GPU_Demand": info["gpu"],
                "Migrated": info["migrated"],
                "IT_Power_MW": info["it_pw"],
                "Facility_Power_MW": info["fac_pw"],
            }
        )

    sched_df = pd.DataFrame(rows)
    sched_df = sched_df.sort_values("TaskID").reset_index(drop=True)
    sched_df.to_csv(OUTPUT_DIR / f"greedy_{name}_schedule.csv", index=False)

    # 同时收集到总表
    sched_df_with_scheme = sched_df.copy()
    sched_df_with_scheme["Scheme"] = scheme_name
    all_rows.append(sched_df_with_scheme)

# ============================================================
# 5. 输出每方案的指标 CSV
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 写入指标与汇总文件")
print("=" * 60)

# 单方案指标（每个方案一个文件）
for name, metrics_row in zip(weight_configs.keys(), all_metrics):
    mdf = pd.DataFrame([metrics_row])
    mdf.to_csv(OUTPUT_DIR / f"greedy_{name}_metrics.csv", index=False)

# 横向对比总表
cmp_df = pd.DataFrame(all_metrics)
cmp_df.to_csv(OUTPUT_DIR / "greedy_all_compare.csv", index=False)

# 所有方案合并的逐任务调度总表
full_df = pd.concat(all_rows, ignore_index=True)
full_df.to_csv(OUTPUT_DIR / "greedy_all_schedules.csv", index=False)

# ============================================================
# 6. 额外：按区域统计资源占用
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 生成各方案的区域资源占用统计")
print("=" * 60)

occupy_rows = []
for name, sched in all_schedules.items():
    gpu_per_region = defaultdict(float)
    hours_per_region = defaultdict(int)
    for tid, info in sched.items():
        r = info["region"]
        hours = info["end"] - info["start"] + 1
        gpu_per_region[r] += info["gpu"] * hours
        hours_per_region[r] += hours

    for r in regions:
        occupy_rows.append(
            {
                "Scheme": f"greedy_{name}",
                "Region": r,
                "Total_GPU_h": gpu_per_region.get(r, 0),
                "Total_Hours": hours_per_region.get(r, 0),
                "Avg_GPU": gpu_per_region.get(r, 0)
                / max(hours_per_region.get(r, 1), 1),
            }
        )

occupy_df = pd.DataFrame(occupy_rows)
occupy_df.to_csv(OUTPUT_DIR / "greedy_region_occupancy.csv", index=False)

# ============================================================
# 7. 终端汇总打印
# ============================================================
print("\n" + "=" * 60)
print("📊 所有贪婪方案核心指标对比")
print("=" * 60)
print(
    f"\n{'方案':<25s} {'成本(元)':>12s} {'碳(tCO2)':>10s} "
    f"{'时延(ms)':>8s} {'新能源':>8s} {'迁移':>6s}"
)
print("-" * 78)
for m in all_metrics:
    print(
        f"{m['scheme']:<25s} {m['cost_yuan']:>12.0f} "
        f"{m['carbon_tco2']:>10.2f} {m['avg_latency_ms']:>8.1f} "
        f"{m['renew_utilization']:>8.4f} {int(m['migrated']):>6d}"
    )

print("\n" + "=" * 60)
print("📁 输出文件清单")
print("=" * 60)
for f in sorted(OUTPUT_DIR.glob("*.csv")):
    size_kb = f.stat().st_size / 1024
    print(f"  ✅ {f.name:<40s} ({size_kb:.1f} KB)")

print(f"\n🎉 全部完成！共输出 {len(weight_configs)} 种贪婪调度方案。")
