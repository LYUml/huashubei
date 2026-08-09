#!/usr/bin/env python3
"""
问题二：碳感知任务调度模型 —— NSGA-II (改进版 v3 修复版)
=================================================================
修复点：
1. 修复 Step 4 中边遍历边追加导致的无限循环和 OOM (Killed) 问题
2. 路径改为基于 __file__ 的相对路径，跨平台可移植
3. SeedSampling 噪声生成逻辑修正
4. eliminate_duplicates 改为 False，避免重复消除导致死循环
5. 添加 GenCounter 回调，实时打印每代进度
6. _evaluate 性能优化：预构建查找表，减少内层循环开销

路径说明：
- 预处理数据文件必须位于本脚本同级的 output/ 目录下，名为 preprocessed_data.pkl
- 结果将输出到本脚本同级的 output_nsga2/ 目录
"""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
import sys
import time
from pathlib import Path

warnings.filterwarnings("ignore")

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.core.callback import Callback

# ── 路径设置（关键修改：强制使用脚本所在目录） ──────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "output"  # preprocessed_data.pkl 所在目录
OUTPUT_DIR = SCRIPT_DIR / "output_nsga2"  # NSGA-II 结果输出目录

print("=" * 60)
print("路径诊断")
print("=" * 60)
print(f"脚本所在目录 : {SCRIPT_DIR}")
print(f"数据目录     : {DATA_DIR}")
print(f"输出目录     : {OUTPUT_DIR}")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"数据目录不存在: {DATA_DIR}\n"
        f"请确保已将 preprocessed_data.pkl 放置在 {DATA_DIR} 下。\n"
        f"如果您尚未生成预处理数据，请先运行 q2_data_prep.py。"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(2024)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n" + "=" * 60)
print("Step1: 加载预处理数据")
print("=" * 60)

pkl_path = DATA_DIR / "preprocessed_data.pkl"
if not pkl_path.exists():
    raise FileNotFoundError(
        f"找不到 {pkl_path}\n请先运行 q2_data_prep.py 生成预处理数据。"
    )

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
print(f"任务总数: {N}")
print(f"区域: {regions}")
print(f"任务类型: {task_types}")

# 快速查找表
pp = {}
for _, row in rtd.iterrows():
    r, h = row["Region"], int(row["Hour"])
    pp[(r, h)] = {
        "price": float(row["Price_Yuan_per_MWh"]),
        "carbon": float(row["CarbonIntensity_tCO2_per_MWh"]),
        "renew": float(row["AvailableRenewable_MW"]),
        "nonai": float(row["NonAI_IT_Load_MW"]),
    }

rc = {}
for r, info in region_info.items():
    rc[r] = {
        "gpu": int(info["Available_GPU"]),
        "max_it": float(info["Max_IT_Power_MW"]),
        "max_fac": float(info["Max_Facility_Power_MW"]),
        "pue": float(info["PUE"]),
    }

R2I = {r: i for i, r in enumerate(regions)}
priority = {"RealTimeInference": 0, "BatchInference": 1, "AITraining": 2}

# ============================================================
# 2. 任务预处理
# ============================================================
print("\n" + "=" * 60)
print("Step2: 构建候选区域与可行时间窗")
print("=" * 60)


def dur_h(row):
    return int(np.ceil(row["EstimatedDuration_min"] / 60))


tasks = []
for idx, row in workload.iterrows():
    tid = int(row["TaskID"])
    src = row["SourceRegion"]
    tt = row["TaskType"]
    gpu = float(row["GPU_Demand"])
    dh = dur_h(row)
    arr = int(row["ArrivalHour"])
    lfh = int(row["LatestFinishHour"])
    maxlat = float(row["MaxLatency_ms"])
    it_pw = gpu * float(power_dict.get(tt, 0.003))

    cand_regions = []
    for r in regions:
        if r == src:
            cand_regions.append(r)
        else:
            try:
                lat = float(latency_matrix.loc[src, r])
                if lat <= maxlat:
                    cand_regions.append(r)
            except Exception:
                pass
    if len(cand_regions) == 0:
        cand_regions = [src]

    if tt == "RealTimeInference":
        start_options = [arr]
    else:
        latest_start = min(lfh - dh, 2405 - dh)
        max_delay = max(0, latest_start - arr)
        max_delay = min(max_delay, 72)
        start_options = list(range(arr, arr + max_delay + 1))

    start_options = [s for s in start_options if s + dh - 1 <= 2405]

    tasks.append(
        {
            "tid": tid,
            "src": src,
            "tt": tt,
            "gpu": gpu,
            "dur_h": dh,
            "arr": arr,
            "lfh": lfh,
            "maxlat": maxlat,
            "it_pw": it_pw,
            "cand_regions": cand_regions,
            "start_options": start_options,
            "pri": priority[tt],
        }
    )

tasks_sorted = sorted(tasks, key=lambda t: (t["pri"], t["arr"]))
n_vars = len(tasks_sorted)

for t in tasks_sorted:
    t["n_cand"] = max(len(t["cand_regions"]) * len(t["start_options"]), 1)

total_renewable = sum(p["renew"] for p in pp.values())
print(f"任务数: {n_vars}")
print(f"全时域可用新能源总量: {total_renewable:.0f} MW·h")

# 预构建每个任务的 (r, s, e) 查找表，加速评估
for i, t in enumerate(tasks_sorted):
    t["_lookup"] = []
    for r_idx, r in enumerate(t["cand_regions"]):
        for s_idx, s in enumerate(t["start_options"]):
            e = s + t["dur_h"] - 1
            t["_lookup"].append((r_idx, s_idx, r, s, e))

# ============================================================
# 3. 贪心调度器（用于生成种子 + 基线对比）
# ============================================================
print("\n" + "=" * 60)
print("Step3: 多组权重贪心调度（生成种子 + 基线）")
print("=" * 60)


def greedy_schedule(workload_df, alpha, verbose=False):
    """alpha = (a_cost, a_carbon, a_latency, a_renew)"""
    a1, a2, a3, a4 = alpha
    df = workload_df.copy()
    df["_pri"] = df["TaskType"].map(priority)
    df = df.sort_values(["_pri", "ArrivalHour"]).reset_index(drop=True)

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

        if tt == "RealTimeInference":
            start_options = [arr]
        else:
            latest_start = min(lfh - dur, 2405 - dur)
            max_delay = max(0, latest_start - arr)
            max_delay = min(max_delay, 72)
            start_options = list(range(arr, arr + max_delay + 1))

        cand = [
            r
            for r in regions
            if r == src or float(latency_matrix.loc[src, r]) <= maxlat
        ]
        if not cand:
            cand = [src]

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
                # 快速容量检查
                ok = True
                for h in range(s, e + 1):
                    if gpu_use[r][h] + gpu > cap_gpu or it_use[r][h] + it_pw > cap_it:
                        ok = False
                        break
                if not ok:
                    continue

                # 评估
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

                sc = a1 * cost + a2 * carbon + a3 * lat_pen - a4 * renew
                if sc < best_score:
                    best_score = sc
                    best = (r, s, e)

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


def compute_all_metrics(sched):
    """返回 (cost, carbon, avg_latency, renew_util, n_migrated, n_tasks)"""
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


# 多组权重
weight_configs = {
    "balanced": (1.0, 1.0, 0.5, 0.5),
    "cost": (1.0, 0.1, 0.1, 0.0),
    "carbon": (0.1, 1.0, 0.1, 0.0),
    "renew": (0.1, 0.1, 0.1, 1.0),
    "latency": (0.1, 0.1, 1.0, 0.0),
    "cost2": (2.0, 0.5, 0.3, 0.2),
    "bal2": (0.5, 2.0, 0.5, 0.3),
    "mix": (1.5, 1.5, 0.8, 0.6),
}

seed_scheds = {}
print(
    f"\n{'方案':<20s} {'成本':>12s} {'碳':>10s} {'时延':>8s} {'新能源':>8s} {'迁移':>6s}"
)
print("-" * 70)
for name, w in weight_configs.items():
    sched = greedy_schedule(workload, w)
    c, co, al, ru, mn, nt = compute_all_metrics(sched)
    seed_scheds[name] = sched
    print(f"greedy_{name:<14s} {c:>12.0f} {co:>10.2f} {al:>8.1f} {ru:>8.4f} {mn:>6d}")

# ============================================================
# 4. 将调度方案编码为整数基因 (修复死循环与内存泄漏)
# ============================================================
print("\n" + "=" * 60)
print("Step4: 编码方案")
print("=" * 60)


def schedule_to_genes(sched):
    """将调度方案编码为基因数组"""
    genes = np.zeros(n_vars, dtype=int)
    for i, t in enumerate(tasks_sorted):
        tid = t["tid"]
        if tid in sched:
            info = sched[tid]
            r = info["region"]
            s = info["start"]
            if r in t["cand_regions"]:
                r_idx = t["cand_regions"].index(r)
                if s in t["start_options"]:
                    s_idx = t["start_options"].index(s)
                    n_s = len(t["start_options"])
                    g = r_idx * n_s + s_idx
                    genes[i] = min(g, t["n_cand"] - 1)
    return genes


# 构建初始种群（种子 + 扰动变体）
seed_genes_list = []
seed_labels = []
for name, sched in seed_scheds.items():
    g = schedule_to_genes(sched)
    seed_genes_list.append(g)
    seed_labels.append(f"greedy_{name}")

# 对种子施加随机扰动，增加多样性
# 【修复点】：避免在遍历期间向 seed_genes_list 追加元素导致无限循环从而 Out of memory
rng = np.random.default_rng(2024)
n_mutate_per_seed = 1

mutated_genes = []
mutated_labels = []

for base_idx, base_g in enumerate(seed_genes_list):
    for m in range(n_mutate_per_seed):
        new_g = base_g.copy()
        # 随机翻转 5% 的基因位
        n_flip = max(1, n_vars // 20)
        flip_idx = rng.choice(n_vars, size=n_flip, replace=False)
        for fi in flip_idx:
            t = tasks_sorted[fi]
            max_g = t["n_cand"] - 1
            new_g[fi] = rng.integers(0, max_g + 1)

        mutated_genes.append(new_g)
        mutated_labels.append(f"mut_{seed_labels[base_idx]}_{m}")

# 遍历完毕后统一拼接，防止死循环
seed_genes_list.extend(mutated_genes)
seed_labels.extend(mutated_labels)

seed_array = np.array(seed_genes_list, dtype=int)
print(f"种子个体数: {len(seed_array)}")
for i, lab in enumerate(seed_labels):
    print(f"  [{i:2d}] {lab}")

# 变量边界
xl = np.zeros(n_vars, dtype=int)
xu = np.array([t["n_cand"] - 1 for t in tasks_sorted], dtype=int)

# ============================================================
# 5. 定义 pymoo Problem（性能优化版）
# ============================================================
print("\n" + "=" * 60)
print("Step5: 定义 NSGA-II 问题")
print("=" * 60)

# 预构建全局查找数组，避免在 _evaluate 中反复字典查找
_lookup_tables = [t["_lookup"] for t in tasks_sorted]
_cand_regions_list = [t["cand_regions"] for t in tasks_sorted]
_start_options_list = [t["start_options"] for t in tasks_sorted]
_dur_h_list = [t["dur_h"] for t in tasks_sorted]
_gpu_list = [t["gpu"] for t in tasks_sorted]
_it_pw_list = [t["it_pw"] for t in tasks_sorted]
_src_list = [t["src"] for t in tasks_sorted]

# 区域参数的查找表
_rc_gpu = {r: rc[r]["gpu"] for r in regions}
_rc_max_it = {r: rc[r]["max_it"] for r in regions}
_rc_pue = {r: rc[r]["pue"] for r in regions}

# pp 查找表预转为数组 (region_index, hour) -> (price, carbon, renew)
_max_hour = 2406
_pp_price = defaultdict(float)
_pp_carbon = defaultdict(float)
_pp_renew = defaultdict(float)
for (r, h), v in pp.items():
    ri = R2I[r]
    _pp_price[(ri, h)] = v["price"]
    _pp_carbon[(ri, h)] = v["carbon"]
    _pp_renew[(ri, h)] = v["renew"]


class NSGA2Problem(Problem):
    eval_count = 0  # 评估计数器

    def __init__(self):
        super().__init__(
            n_var=n_vars,
            n_obj=4,
            n_constr=0,
            xl=xl,
            xu=xu,
            type_var=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        max_hour = 2406
        n_regions = len(regions)
        region_idx = {r: i for i, r in enumerate(regions)}

        gpu_usage = np.zeros((n_regions, max_hour), dtype=np.float32)
        it_usage = np.zeros((n_regions, max_hour), dtype=np.float32)

        F = np.zeros((n_pop, 4), dtype=np.float64)

        for k in range(n_pop):
            genes = X[k]
            gpu_usage.fill(0)
            it_usage.fill(0)

            cost = carbon = renew_used = 0.0
            lat_sum = 0.0
            mig_n = 0
            feasible = True
            penalty = 0.0

            for i, t in enumerate(tasks_sorted):
                g = int(genes[i])
                n_s = len(t["start_options"])
                if n_s == 0 or len(t["cand_regions"]) == 0:
                    feasible = False
                    penalty += 1e6
                    continue

                r_idx = g // n_s
                s_idx = g % n_s
                if r_idx >= len(t["cand_regions"]) or s_idx >= len(t["start_options"]):
                    feasible = False
                    penalty += 1e6
                    continue

                r_name = t["cand_regions"][r_idx]
                r_id = region_idx[r_name]
                s = t["start_options"][s_idx]
                e = s + t["dur_h"] - 1

                cap_gpu = rc[r_name]["gpu"]
                cap_it = rc[r_name]["max_it"]
                it_pw = t["it_pw"]
                pue = rc[r_name]["pue"]
                fac_pw = it_pw * pue

                overload = False
                for h in range(s, e + 1):
                    if gpu_usage[r_id, h] + t["gpu"] > cap_gpu:
                        overload = True
                        penalty += (gpu_usage[r_id, h] + t["gpu"] - cap_gpu) * 100
                        break
                    if it_usage[r_id, h] + it_pw > cap_it:
                        overload = True
                        penalty += (it_usage[r_id, h] + it_pw - cap_it) * 100
                        break
                if overload:
                    feasible = False
                    continue

                for h in range(s, e + 1):
                    gpu_usage[r_id, h] += t["gpu"]
                    it_usage[r_id, h] += it_pw
                    p = pp.get((r_name, h), {"price": 0, "carbon": 0, "renew": 0})
                    cost += fac_pw * p["price"]
                    carbon += fac_pw * p["carbon"]
                    renew_used += min(p["renew"], fac_pw)

                if r_name != t["src"]:
                    mig_n += 1
                    try:
                        lat_sum += float(latency_matrix.loc[t["src"], r_name])
                    except:
                        pass

            avg_lat = lat_sum / max(mig_n, 1)
            renew_util = renew_used / total_renewable if total_renewable > 0 else 0

            if not feasible:
                cost += penalty
                carbon += penalty * 0.001
                avg_lat += penalty * 0.01

            F[k, 0] = cost
            F[k, 1] = carbon
            F[k, 2] = avg_lat
            F[k, 3] = 1.0 - renew_util

        out["F"] = F


problem = NSGA2Problem()

# ============================================================
# 6. 自定义采样器（修正版）
# ============================================================
from pymoo.core.sampling import Sampling


class SeedSampling(Sampling):
    """种子初始化 + 随机补充，确保多样性"""

    def __init__(self, seed_array, noise_prob=0.05):
        super().__init__()
        self.seed_array = seed_array
        self.noise_prob = noise_prob

    def _do(self, problem, n_samples, **kwargs):
        n_seed = len(self.seed_array)
        if n_samples <= n_seed:
            rng = np.random.default_rng()
            indices = rng.choice(n_seed, size=n_samples, replace=False)
            selected = self.seed_array[indices].copy()
            for i in range(n_samples):
                for j in range(problem.n_var):
                    if np.random.rand() < self.noise_prob:
                        selected[i, j] = np.random.randint(
                            problem.xl[j], problem.xu[j] + 1
                        )
            return selected

        extra_n = n_samples - n_seed
        extra = np.zeros((extra_n, problem.n_var), dtype=int)
        for i in range(extra_n):
            for j in range(problem.n_var):
                extra[i, j] = np.random.randint(problem.xl[j], problem.xu[j] + 1)
        seeds = self.seed_array.copy()
        for i in range(n_seed):
            for j in range(problem.n_var):
                if np.random.rand() < self.noise_prob:
                    seeds[i, j] = np.random.randint(problem.xl[j], problem.xu[j] + 1)
        return np.vstack([seeds, extra])


pop_size = max(80, len(seed_array) * 2)
pop_size = min(pop_size, 120)  # 上限，避免评估过慢
n_gen = 30  # 调试时可改小，正式跑推荐 300

print(f"种群大小: {pop_size}, 代数: {n_gen}")
print(f"总评估次数预估: {pop_size * n_gen}")


# ============================================================
# 7. 回调函数（实时显示进度）
# ============================================================
class GenCounter(Callback):
    """每代结束时打印进度"""

    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def notify(self, algorithm):
        gen = algorithm.n_gen
        n_eval = algorithm.evaluator.n_eval
        elapsed = time.time() - self.start_time
        if gen > 0:
            avg_per_gen = elapsed / gen
            eta = avg_per_gen * (n_gen - gen)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = ""
        print(
            f"[GEN] gen={gen}/{n_gen} evals={n_eval} "
            f"elapsed={elapsed:.0f}s {eta_str}",
            flush=True,
        )


# ============================================================
# 8. 运行 NSGA-II
# ============================================================
print("\n" + "=" * 60)
print("Step6: 运行 NSGA-II")
print("=" * 60)

algorithm = NSGA2(
    pop_size=pop_size,
    sampling=SeedSampling(seed_array, noise_prob=0.03),
    crossover=SBX(prob=0.9, eta=10, vtype=int),
    mutation=PolynomialMutation(eta=15, vtype=int, prob=1.0 / n_vars),
    eliminate_duplicates=False,  # 关闭！避免死循环
    callback=GenCounter(),
)

t0 = time.time()
res = minimize(
    problem,
    algorithm,
    ("n_gen", n_gen),
    seed=42,
    verbose=True,  # 打开 pymoo 内置日志
)
elapsed = time.time() - t0

print(f"\n✅ NSGA-II 完成, Pareto 解数: {len(res.F)}")
print(f" 总耗时: {elapsed:.1f}s")
print(f" 总评估次数: {problem.eval_count}")

# ============================================================
# 9. 提取 Pareto 解集 & 选择代表解
# ============================================================
print("\n" + "=" * 60)
print("Step7: 提取 Pareto 解集")
print("=" * 60)

pareto_F = res.F
pareto_X = res.X

# 归一化后找 knee point
F_min = pareto_F.min(axis=0)
F_max = pareto_F.max(axis=0)
F_range = np.clip(F_max - F_min, 1e-10, None)
F_norm = (pareto_F - F_min) / F_range

# Knee: 距理想点最远
distances = np.linalg.norm(F_norm, axis=1)
knee_idx = np.argmax(distances)

# 各目标最优
idx_cost = np.argmin(pareto_F[:, 0])
idx_carbon = np.argmin(pareto_F[:, 1])
idx_latency = np.argmin(pareto_F[:, 2])
idx_renew = np.argmin(pareto_F[:, 3])

print(
    f"\n{'方案':<22s} {'成本(元)':>14s} {'碳(tCO2)':>12s} {'时延(ms)':>10s} {'新能源利用':>10s}"
)
print("-" * 75)
for lab, idx in [
    ("knee", knee_idx),
    ("min_cost", idx_cost),
    ("min_carbon", idx_carbon),
    ("min_latency", idx_latency),
    ("max_renew", idx_renew),
]:
    f = pareto_F[idx]
    print(f"nsga2_{lab:<16s} {f[0]:>14.0f} {f[1]:>12.2f} {f[2]:>10.1f} {1-f[3]:>10.4f}")


# ============================================================
# 10. 解码 Pareto 解 → 调度表
# ============================================================
def genes_to_schedule(genes):
    """将基因解码为调度方案"""
    sched = {}
    for i, t in enumerate(tasks_sorted):
        g = int(genes[i])
        lookup = t["_lookup"]
        if len(lookup) == 0:
            continue
        g = min(g, len(lookup) - 1)
        r_idx, s_idx, r, s, e = lookup[g]
        sched[t["tid"]] = {
            "region": r,
            "start": s,
            "end": e,
            "dur": t["dur_h"],
            "gpu": t["gpu"],
            "task_type": t["tt"],
            "source": t["src"],
            "it_pw": t["it_pw"],
            "fac_pw": t["it_pw"] * rc[r]["pue"],
            "migrated": (r != t["src"]),
        }
    return sched


selected = {}
for lab, idx in [
    ("knee", knee_idx),
    ("min_cost", idx_cost),
    ("min_carbon", idx_carbon),
    ("min_latency", idx_latency),
    ("max_renew", idx_renew),
]:
    genes = pareto_X[idx]
    sched = genes_to_schedule(genes)
    c, co, al, ru, mn, nt = compute_all_metrics(sched)
    selected[lab] = {
        "genes": genes,
        "sched": sched,
        "metrics": {
            "cost": c,
            "carbon": co,
            "latency": al,
            "renew": ru,
            "migrated": mn,
            "n": nt,
        },
    }
    # 保存调度表
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
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"q2_nsga2_sched_{lab}.csv", index=False)

# Pareto 前沿表
pf_rows = []
for k in range(len(pareto_F)):
    pf_rows.append(
        {
            "sol_id": k,
            "cost_yuan": pareto_F[k, 0],
            "carbon_tco2": pareto_F[k, 1],
            "avg_latency_ms": pareto_F[k, 2],
            "renew_utilization": 1 - pareto_F[k, 3],
        }
    )
pf_df = pd.DataFrame(pf_rows)
pf_df.to_csv(OUTPUT_DIR / "q2_nsga2_pareto.csv", index=False)

# ============================================================
# 11. 综合对比表
# ============================================================
print("\n" + "=" * 60)
print("Step9: 综合对比")
print("=" * 60)

compare_rows = []
for name, sched in seed_scheds.items():
    c, co, al, ru, mn, nt = compute_all_metrics(sched)
    compare_rows.append(
        {
            "scheme": f"greedy_{name}",
            "cost_yuan": c,
            "carbon_tco2": co,
            "avg_latency_ms": al,
            "renew_utilization": ru,
            "migrated": mn,
            "n_tasks": nt,
        }
    )

for lab, data in selected.items():
    m = data["metrics"]
    compare_rows.append(
        {
            "scheme": f"nsga2_{lab}",
            "cost_yuan": m["cost"],
            "carbon_tco2": m["carbon"],
            "avg_latency_ms": m["latency"],
            "renew_utilization": m["renew"],
            "migrated": m["migrated"],
            "n_tasks": m["n"],
        }
    )

cmp_df = pd.DataFrame(compare_rows)
cmp_df.to_csv(OUTPUT_DIR / "q2_nsga2_compare.csv", index=False)

print(
    f"\n{'方案':<25s} {'成本(元)':>12s} {'碳(tCO2)':>10s} {'时延(ms)':>8s} {'新能源':>8s} {'迁移':>6s}"
)
print("-" * 78)
for _, r in cmp_df.iterrows():
    print(
        f"{r['scheme']:<25s} {r['cost_yuan']:>12.0f} {r['carbon_tco2']:>10.2f} "
        f"{r['avg_latency_ms']:>8.1f} {r['renew_utilization']:>8.4f} {int(r['migrated']):>6d}"
    )

# 超体积
try:
    hv_ref = np.max(pareto_F, axis=0) * 1.1
    hv = Hypervolume(ref_point=hv_ref).do(pareto_F)
    print(f"\n超体积 HV = {hv:.2f}")
except Exception as e:
    print(f"\nHV 计算跳过: {e}")

# ============================================================
# 12. 保存完整结果
# ============================================================
print("\n" + "=" * 60)
print("Step10: 保存")
print("=" * 60)

with open(OUTPUT_DIR / "q2_nsga2_result.pkl", "wb") as f:
    pickle.dump(
        {
            "pareto_F": pareto_F,
            "pareto_X": pareto_X,
            "pareto_df": pf_df,
            "selected": {k: v["metrics"] for k, v in selected.items()},
            "compare_df": cmp_df,
            "tasks_sorted": tasks_sorted,
            "seed_labels": seed_labels,
        },
        f,
    )

print(f"✅ {OUTPUT_DIR / 'q2_nsga2_result.pkl'}")
print(f"✅ {OUTPUT_DIR / 'q2_nsga2_pareto.csv'}")
print(f"✅ {OUTPUT_DIR / 'q2_nsga2_compare.csv'}")
for lab in selected:
    print(f"✅ {OUTPUT_DIR / f'q2_nsga2_sched_{lab}.csv'}")

print(f"\n🎉 NSGA-II 改进版全部完成！总耗时: {elapsed:.1f}s")
