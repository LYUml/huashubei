"""
问题二：碳感知任务调度模型 —— NSGA-II 多目标优化求解器
=================================================================
与 q2_schedule.py (贪心加权) 的区别：
  本脚本使用 NSGA-II (pymoo) 在真帕累托前沿上搜索，
  不依赖主观权重，能同时给出一组非支配解供决策者选择。

数据接口：复用 q2_data_prep.py 产出的 preprocessed_data.pkl
输出：
  /data/workspace/q2_nsga2_result.pkl   (Pareto 解集 + 指标)
  /data/workspace/q2_nsga2_pareto.csv   (Pareto 前沿表)
  /data/workspace/q2_nsga2_metrics.csv   (每个解的详细指标)
  /data/workspace/q2_nsga2_compare.csv  (与贪心基线的对比)
"""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── pymoo ────────────────────────────────────────────────────────
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import Hypervolume

np.random.seed(42)

# ============================================================
# 1. 加载数据 (与 q2_schedule.py 完全一致)
# ============================================================
print("=" * 60)
print("Step 1: 加载预处理数据")
print("=" * 60)

with open("/data/workspace/preprocessed_data.pkl", 'rb') as f:
    D = pickle.load(f)

workload       = D['workload']
regions        = D['regions']
task_types     = D['task_types']
rtd            = D['region_time']
latency_matrix = D['latency_matrix']
power_dict     = D['power_dict']
region_info    = D['region_info']

N = len(workload)
print(f"任务总数: {N}")
print(f"区域: {regions}")
print(f"类型: {task_types}")

# ── 快速查找表 ────────────────────────────────────────────────
pp = {}  # (region, hour) -> {price, carbon, renew, nonai}
for _, row in rtd.iterrows():
    r, h = row['Region'], int(row['Hour'])
    pp[(r, h)] = {
        'price':  float(row['Price_Yuan_per_MWh']),
        'carbon': float(row['CarbonIntensity_tCO2_per_MWh']),
        'renew':  float(row['AvailableRenewable_MW']),
        'nonai':  float(row['NonAI_IT_Load_MW']),
    }

rc = {}  # region -> {gpu, max_it, max_fac, pue}
for r, info in region_info.items():
    rc[r] = {
        'gpu':     int(info['Available_GPU']),
        'max_it':  float(info['Max_IT_Power_MW']),
        'max_fac': float(info['Max_Facility_Power_MW']),
        'pue':     float(info['PUE']),
    }

R2I = {r: i for i, r in enumerate(regions)}   # region -> index
I2R = {i: r for i, r in enumerate(regions)}   # index -> region
n_reg = len(regions)

# ============================================================
# 2. 任务预处理：构建每个任务的候选区域集合
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 构建候选区域与可行时间窗")
print("=" * 60)

priority = {'RealTimeInference': 0, 'BatchInference': 1, 'AITraining': 2}

tasks = []  # 每个元素是一个 dict，描述一个任务的全部属性
for idx, row in workload.iterrows():
    tid    = int(row['TaskID'])
    src    = row['SourceRegion']
    tt     = row['TaskType']
    gpu    = float(row['GPU_Demand'])
    dur_min= float(row['EstimatedDuration_min'])
    dur_h  = int(np.ceil(dur_min / 60))
    arr    = int(row['ArrivalHour'])
    lfh    = int(row['LatestFinishHour'])
    maxlat = float(row['MaxLatency_ms'])
    it_pw  = gpu * float(power_dict.get(tt, 0.003))

    # 候选区域（满足时延约束）
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
        cand_regions = [src]  # 保底

    # 可行开工时刻
    if tt == 'RealTimeInference':
        start_options = [arr]  # 到达即开工
    else:
        latest_start = min(lfh - dur_h, 2405 - dur_h)
        max_delay = max(0, latest_start - arr)
        max_delay = min(max_delay, 72)  # 最多延迟72小时
        start_options = list(range(arr, arr + max_delay + 1))

    # 过滤掉会跨越2406的开始时刻
    start_options = [s for s in start_options if s + dur_h - 1 <= 2405]

    tasks.append({
        'tid': tid, 'src': src, 'tt': tt, 'gpu': gpu,
        'dur_h': dur_h, 'arr': arr, 'lfh': lfh,
        'maxlat': maxlat, 'it_pw': it_pw,
        'cand_regions': cand_regions,
        'start_options': start_options,
        'pri': priority[tt],
    })

print(f"任务预处理完成: {len(tasks)} 个任务")
print(f"  实时推理: {sum(1 for t in tasks if t['tt']=='RealTimeInference')}")
print(f"  批量推理: {sum(1 for t in tasks if t['tt']=='BatchInference')}")
print(f"  AI训练:   {sum(1 for t in tasks if t['tt']=='AITraining')}")

# 候选列总数（决策变量空间大小）
total_cols = sum(len(t['cand_regions']) * len(t['start_options']) for t in tasks)
print(f"  候选列总数: {total_cols}")

# ============================================================
# 3. 为 NSGA-II 构造整数编码方案
# ============================================================
"""
编码方案：
  每个任务 i 用 1 个整数基因 g_i ∈ [0, n_candidates_i - 1]
  n_candidates_i = |cand_regions| × |start_options|
  解码时：region_idx = g_i // |start_options|
          start_idx  = g_i %  |start_options|
"""
for t in tasks:
    t['n_cand'] = len(t['cand_regions']) * len(t['start_options'])

# 按优先级排序（实时推理最先编码，保证约束优先满足）
tasks_sorted = sorted(tasks, key=lambda t: (t['pri'], t['arr']))

# 决策变量维度
n_vars = len(tasks_sorted)
var_bounds = [(0, t['n_cand'] - 1) for t in tasks_sorted]
xl = np.array([b[0] for b in var_bounds], dtype=int)
xu = np.array([b[1] for b in var_bounds], dtype=int)

print(f"\nNSGA-II 决策变量维度: {n_vars}")
print(f"  每变量取值范围: 0 ~ {max(t['n_cand']-1 for t in tasks_sorted)}")

# ============================================================
# 4. 解码 + 约束检查 + 目标计算
# ============================================================
def decode_and_evaluate(genes):
    """
    输入: genes, shape (n_vars,) 整数数组
    输出: (cost, carbon, avg_latency, renew_util, feasibility_penalty)
    """
    # 初始化逐时占用
    gpu_use = {r: defaultdict(float) for r in regions}
    it_use  = {r: defaultdict(float) for r in regions}

    cost_total = 0.0
    carbon_total = 0.0
    lat_total = 0.0
    mig_count = 0
    renew_total_used = 0.0
    total_renewable = 0.0  # 全时域可用新能源总量
    feasible = True
    penalty = 0.0

    for i, t in enumerate(tasks_sorted):
        g = int(genes[i])
        n_r = len(t['cand_regions'])
        n_s = len(t['start_options'])
        r_idx = g // n_s
        s_idx = g % n_s
        r = t['cand_regions'][r_idx]
        s = t['start_options'][s_idx]
        e = s + t['dur_h'] - 1

        # 检查 GPU 容量 & IT 功率
        cap_gpu = rc[r]['gpu']
        cap_it  = rc[r]['max_it']
        it_pw   = t['it_pw']
        pue_r   = rc[r]['pue']

        for h in range(s, e + 1):
            if gpu_use[r][h] + t['gpu'] > cap_gpu:
                feasible = False
                penalty += (gpu_use[r][h] + t['gpu'] - cap_gpu) * 1000
                break
            if it_use[r][h] + it_pw > cap_it:
                feasible = False
                penalty += (it_use[r][h] + it_pw - cap_it) * 1000
                break

        if not feasible:
            continue

        # 累加占用
        fac_pw = it_pw * pue_r
        for h in range(s, e + 1):
            gpu_use[r][h] += t['gpu']
            it_use[r][h]  += it_pw

            # 累加成本、碳排、新能源消纳
            p = pp.get((r, h))
            if p is None:
                p = {'price': 0.0, 'carbon': 0.0, 'renew': 0.0}
            cost_total   += fac_pw * p['price']
            carbon_total += fac_pw * p['carbon']
            renew_used    = min(p['renew'], fac_pw)
            renew_total_used += renew_used

        # 时延
        if r != t['src']:
            mig_count += 1
            try:
                lat_total += float(latency_matrix.loc[t['src'], r])
            except Exception:
                pass

    # 新能源利用率（全时域总量）
    for (r, h), p in pp.items():
        total_renewable += p['renew']
    if total_renewable > 0:
        renew_util = renew_total_used / total_renewable
    else:
        renew_util = 0.0

    # 平均迁移时延
    n_mig = sum(1 for i, t in enumerate(tasks_sorted)
                if t['cand_regions'][genes[i] // len(t['start_options'])] != t['src'])
    avg_latency = lat_total / max(n_mig, 1)

    return cost_total, carbon_total, avg_latency, renew_util, penalty, feasible

# ============================================================
# 5. 定义 pymoo Problem
# ============================================================
# 先做一次 dummy 调用获取 total_renewable 的参考值
_total_renew = sum(p['renew'] for p in pp.values())
print(f"\n全时域可用新能源总量: {_total_renew:.2f} MW")

class CarbonAwareProblem(Problem):
    """
    4 个目标 (全部最小化):
      F1 = 运行成本 (元)
      F2 = 碳排放量 (tCO2)
      F3 = 平均网络时延 (ms)  [仅迁移任务]
      F4 = 1 - 新能源利用率  (越大=利用率越低, 要最小化)
    无非线性约束 (约束通过罚函数内化到目标中)
    """
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
        """
        X: (n_pop, n_var) 整数矩阵
        out['F']: (n_pop, 4) 目标值
        """
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 4))

        for k in range(n_pop):
            genes = X[k]
            cost, carbon, avg_lat, renew_util, penalty, feasible = \
                decode_and_evaluate(genes)

            # 罚函数：不可行解的 F 值大幅抬高
            if not feasible:
                cost   += penalty
                carbon  += penalty * 0.001
                avg_lat += penalty * 0.1

            F[k, 0] = cost
            F[k, 1] = carbon
            F[k, 2] = avg_lat
            F[k, 3] = 1.0 - renew_util  # 最小化 = 最大化利用率

        out['F'] = F


problem = CarbonAwareProblem()

# ============================================================
# 6. 运行 NSGA-II
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 运行 NSGA-II 多目标优化")
print("=" * 60)

algorithm = NSGA2(
    pop_size=50,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.9, eta=15, vtype=int),
    mutation=PolynomialMutation(eta=20, vtype=int, prob=1.0/n_vars),
    eliminate_duplicates=True,
)

res = minimize(
    problem,
    algorithm,
    ('n_gen', 200),
    seed=42,
    verbose=True,
)

print(f"\n✅ NSGA-II 完成")
print(f"  Pareto 解数量: {len(res.F)}")

# ============================================================
# 7. 提取并保存 Pareto 解集
# ============================================================
print("\n" + "=" * 60)
print("Step 7: 提取 Pareto 解集")
print("=" * 60)

pareto_F = res.F          # (n_pareto, 4) 目标值
pareto_X = res.X          # (n_pareto, n_vars) 决策变量

# 构建结果表
rows = []
for k in range(len(pareto_F)):
    genes = pareto_X[k]
    cost, carbon, avg_lat, renew_util, penalty, feasible = \
        decode_and_evaluate(genes)
    rows.append({
        'sol_id': k,
        'cost_yuan': cost,
        'carbon_tco2': carbon,
        'avg_latency_ms': avg_lat,
        'renew_utilization': renew_util,
        'feasible': feasible,
    })

pf_df = pd.DataFrame(rows)
pf_df.to_csv("/data/workspace/q2_nsga2_pareto.csv", index=False)
print(f"✅ Pareto 前沿表 → q2_nsga2_pareto.csv ({len(pf_df)} 行)")
print(f"\nPareto 前沿概览:")
print(pf_df[['cost_yuan','carbon_tco2','avg_latency_ms','renew_utilization']].describe().round(2))

# ============================================================
# 8. 选择代表解 (Knee Point + 4 种典型策略)
# ============================================================
print("\n" + "=" * 60)
print("Step 8: 选择代表解")
print("=" * 60)

# 归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
F_norm = scaler.fit_transform(pareto_F)

# Knee point: 在归一化空间中找离理想点(0,0,0,0)最远的解
ideal = np.zeros(4)
distances = np.linalg.norm(F_norm, axis=1)
knee_idx = np.argmax(distances)
print(f"  Knee point (离理想点最远): sol_id={knee_idx}")
print(f"    成本={pareto_F[knee_idx,0]:.0f} 碳={pareto_F[knee_idx,1]:.2f} "
      f"时延={pareto_F[knee_idx,2]:.1f} 新能源利用率={1-pareto_F[knee_idx,3]:.4f}")

# 4 种典型策略：各目标单独最优
labels = ['min_cost', 'min_carbon', 'min_latency', 'max_renew']
idxs   = [np.argmin(pareto_F[:, j]) for j in range(4)]
print(f"\n  各目标单独最优解:")
for lab, idx in zip(labels, idxs):
    print(f"    {lab:15s}: sol_id={idx}  "
          f"成本={pareto_F[idx,0]:.0f} 碳={pareto_F[idx,1]:.2f} "
          f"时延={pareto_F[idx,2]:.1f} 新能源利用率={1-pareto_F[idx,3]:.4f}")

# 保存全部 Pareto 解 + 选中解的详细调度
selected = {'knee': int(knee_idx)}
for lab, idx in zip(labels, idxs):
    selected[lab] = int(idx)

# 为每个选中解构建完整调度表
def build_schedule(genes):
    """将基因解码为可读的调度表"""
    sched_rows = []
    for i, t in enumerate(tasks_sorted):
        g = int(genes[i])
        n_s = len(t['start_options'])
        r_idx = g // n_s
        s_idx = g % n_s
        r = t['cand_regions'][r_idx]
        s = t['start_options'][s_idx]
        e = s + t['dur_h'] - 1
        sched_rows.append({
            'TaskID': t['tid'],
            'TaskType': t['tt'],
            'SourceRegion': t['src'],
            'AssignedRegion': r,
            'StartHour': s,
            'EndHour': e,
            'Duration_h': t['dur_h'],
            'GPU_Demand': t['gpu'],
            'Migrated': (r != t['src']),
            'IT_Power_MW': t['it_pw'],
            'Facility_Power_MW': t['it_pw'] * rc[r]['pue'],
        })
    return sched_rows

selected_scheds = {}
for name, idx in selected.items():
    genes = pareto_X[idx]
    rows = build_schedule(genes)
    df = pd.DataFrame(rows)
    selected_scheds[name] = df
    df.to_csv(f"/data/workspace/q2_nsga2_sched_{name}.csv", index=False)
    print(f"  ✅ 调度表 → q2_nsga2_sched_{name}.csv ({len(df)} 行)")

# ============================================================
# 9. 与贪心基线的对比
# ============================================================
print("\n" + "=" * 60)
print("Step 9: 与贪心基线的对比")
print("=" * 60)

# 加载贪心结果
try:
    greedy_df = pd.read_csv("/data/workspace/q2_schedule.csv")
    greedy_cost = greedy_df['Facility_Power_MW'].sum() * 0  # placeholder
    # 重新计算贪心指标
    g_cost = g_carbon = g_renew = 0.0
    g_lat = 0.0; g_mig = 0
    for _, row in greedy_df.iterrows():
        r = row['AssignedRegion']
        s = int(row['StartHour'])
        e = int(row['EndHour'])
        fac_pw = float(row['Facility_Power_MW'])
        for h in range(s, e + 1):
            p = pp.get((r, h))
            if p is None:
                p = {'price':0,'carbon':0,'renew':0}
            g_cost   += fac_pw * p['price']
            g_carbon += fac_pw * p['carbon']
            g_renew  += min(p['renew'], fac_pw)
        if row['Migrated']:
            g_mig += 1
            try:
                g_lat += float(latency_matrix.loc[row['SourceRegion'], r])
            except: pass
    g_total_renew = sum(p['renew'] for p in pp.values())
    g_util = g_renew / g_total_renew if g_total_renew > 0 else 0
    g_avg_lat = g_lat / max(g_mig, 1)

    compare_rows = []
    compare_rows.append({
        'scheme': 'greedy_weighted',
        'cost_yuan': g_cost,
        'carbon_tco2': g_carbon,
        'avg_latency_ms': g_avg_lat,
        'renew_utilization': g_util,
    })
    for name, idx in selected.items():
        compare_rows.append({
            'scheme': f'nsga2_{name}',
            'cost_yuan': pareto_F[idx, 0],
            'carbon_tco2': pareto_F[idx, 1],
            'avg_latency_ms': pareto_F[idx, 2],
            'renew_utilization': 1 - pareto_F[idx, 3],
        })
    cmp_df = pd.DataFrame(compare_rows)
    cmp_df.to_csv("/data/workspace/q2_nsga2_compare.csv", index=False)
    print(f"\n{'方案':<25s} {'成本(元)':>14s} {'碳(tCO2)':>12s} {'时延(ms)':>10s} {'新能源利用':>10s}")
    print("-" * 75)
    for _, r in cmp_df.iterrows():
        print(f"{r['scheme']:<25s} {r['cost_yuan']:>14.1f} {r['carbon_tco2']:>12.2f} "
              f"{r['avg_latency_ms']:>10.1f} {r['renew_utilization']:>10.4f}")

    # 改善率 (knee vs greedy)
    kr = cmp_df[cmp_df['scheme']=='nsga2_knee'].iloc[0]
    gr = cmp_df[cmp_df['scheme']=='greedy_weighted'].iloc[0]
    print(f"\nNSGA-II Knee vs 贪心加权:")
    for col, lab in [('cost_yuan','成本'),('carbon_tco2','碳'),
                     ('avg_latency_ms','时延'),('renew_utilization','新能源利用')]:
        if col == 'renew_utilization':
            imp = (kr[col] - gr[col]) / max(abs(gr[col]), 1e-6) * 100
        else:
            imp = (gr[col] - kr[col]) / max(abs(gr[col]), 1e-6) * 100
        d = "↑" if imp > 0 else "↓"
        print(f"  {lab}: {d}{abs(imp):.1f}%")
except Exception as e:
    print(f"  贪心基线对比跳过: {e}")

# ============================================================
# 10. 保存完整结果
# ============================================================
print("\n" + "=" * 60)
print("Step 10: 保存完整结果")
print("=" * 60)

with open("/data/workspace/q2_nsga2_result.pkl", 'wb') as f:
    pickle.dump({
        'pareto_F': pareto_F,
        'pareto_X': pareto_X,
        'pareto_df': pf_df,
        'selected': selected,
        'selected_scheds': {k: v.to_dict('records') for k, v in selected_scheds.items()},
        'compare_df': cmp_df if 'cmp_df' in locals() else None,
        'tasks_sorted': tasks_sorted,
        'problem_bounds': {'xl': xl, 'xu': xu},
    }, f)
print("✅ 完整结果 → q2_nsga2_result.pkl")

# ============================================================
# 11. 计算超体积 (Hypervolume)
# ============================================================
try:
    hv_ref = np.max(pareto_F, axis=0) * 1.1  # 参考点
    hv = Hypervolume(ref_point=hv_ref).do(pareto_F)
    print(f"\n  超体积 (HV) = {hv:.4f}")
    print(f"  参考点 = {hv_ref}")
except Exception as e:
    print(f"\n  HV 计算跳过: {e}")

print("\n🎉 NSGA-II 碳感知调度全部完成！")
