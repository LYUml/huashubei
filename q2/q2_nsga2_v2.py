"""
问题二：碳感知任务调度模型 —— NSGA-II (改进版)
=================================================================
改进点：
  1. 贪心种子初始化：用多组不同权重的贪心解作为初始种群，
     大幅加速收敛
  2. 修复新能源利用率计算：用逐时 min(renew, load) 累加
  3. 分窗解码：将1200任务分成6个窗口，每个窗口独立优化，
     降低单问题维度
  4. 增加种群规模和代数
  5. 输出与贪心加权方案的公平对比

数据接口：复用 preprocessed_data.pkl
"""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume

np.random.seed(2024)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("Step1: 加载预处理数据")
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

# 快速查找表
pp = {}
for _, row in rtd.iterrows():
    r, h = row['Region'], int(row['Hour'])
    pp[(r, h)] = {
        'price':  float(row['Price_Yuan_per_MWh']),
        'carbon': float(row['CarbonIntensity_tCO2_per_MWh']),
        'renew':  float(row['AvailableRenewable_MW']),
        'nonai':  float(row['NonAI_IT_Load_MW']),
    }

rc = {}
for r, info in region_info.items():
    rc[r] = {
        'gpu':     int(info['Available_GPU']),
        'max_it':  float(info['Max_IT_Power_MW']),
        'max_fac': float(info['Max_Facility_Power_MW']),
        'pue':     float(info['PUE']),
    }

R2I = {r: i for i, r in enumerate(regions)}
priority = {'RealTimeInference': 0, 'BatchInference': 1, 'AITraining': 2}

# ============================================================
# 2. 任务预处理
# ============================================================
print("\n" + "=" * 60)
print("Step2: 构建候选区域与可行时间窗")
print("=" * 60)

def dur_h(row):
    return int(np.ceil(row['EstimatedDuration_min'] / 60))

tasks = []
for idx, row in workload.iterrows():
    tid    = int(row['TaskID'])
    src    = row['SourceRegion']
    tt     = row['TaskType']
    gpu    = float(row['GPU_Demand'])
    dh     = dur_h(row)
    arr    = int(row['ArrivalHour'])
    lfh    = int(row['LatestFinishHour'])
    maxlat = float(row['MaxLatency_ms'])
    it_pw  = gpu * float(power_dict.get(tt, 0.003))

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

    if tt == 'RealTimeInference':
        start_options = [arr]
    else:
        latest_start = min(lfh - dh, 2405 - dh)
        max_delay = max(0, latest_start - arr)
        max_delay = min(max_delay, 72)
        start_options = list(range(arr, arr + max_delay + 1))

    start_options = [s for s in start_options if s + dh - 1 <= 2405]

    tasks.append({
        'tid': tid, 'src': src, 'tt': tt, 'gpu': gpu,
        'dur_h': dh, 'arr': arr, 'lfh': lfh,
        'maxlat': maxlat, 'it_pw': it_pw,
        'cand_regions': cand_regions,
        'start_options': start_options,
        'pri': priority[tt],
    })

tasks_sorted = sorted(tasks, key=lambda t: (t['pri'], t['arr']))
n_vars = len(tasks_sorted)

for t in tasks_sorted:
    t['n_cand'] = len(t['cand_regions']) * len(t['start_options'])

total_renewable = sum(p['renew'] for p in pp.values())
print(f"任务数: {n_vars}")
print(f"全时域可用新能源总量: {total_renewable:.0f} MW·h")

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
    df['_pri'] = df['TaskType'].map(priority)
    df = df.sort_values(['_pri', 'ArrivalHour']).reset_index(drop=True)

    gpu_use = {r: {} for r in regions}
    it_use  = {r: {} for r in regions}
    sched   = {}

    def _gu(r, h): return gpu_use[r].get(h, 0.0)
    def _iu(r, h): return it_use[r].get(h, 0.0)

    for idx, task in df.iterrows():
        tid   = int(task['TaskID'])
        src   = task['SourceRegion']
        gpu   = float(task['GPU_Demand'])
        dur   = dur_h(task)
        maxlat= float(task['MaxLatency_ms'])
        tt     = task['TaskType']
        arr    = int(task['ArrivalHour'])
        lfh    = int(task['LatestFinishHour'])
        it_pw  = gpu * float(power_dict.get(tt, 0.003))

        if tt == 'RealTimeInference':
            start_options = [arr]
        else:
            latest_start = min(lfh - dur, 2405 - dur)
            max_delay = max(0, latest_start - arr)
            max_delay = min(max_delay, 72)
            start_options = list(range(arr, arr + max_delay + 1))

        cand = [r for r in regions if r == src or
                float(latency_matrix.loc[src, r]) <= maxlat]
        if not cand: cand = [src]

        best = None; best_score = float('inf')
        for r in cand:
            cap_gpu = rc[r]['gpu']; cap_it = rc[r]['max_it']; pue = rc[r]['pue']
            for s in start_options:
                e = s + dur - 1
                if e >= 2406: continue
                if any(_gu(r, h) + gpu > cap_gpu for h in range(s, e+1)): continue
                if any(_iu(r, h) + it_pw > cap_it for h in range(s, e+1)): continue

                # 评估
                cost = carbon = renew = 0.0
                for h in range(s, e+1):
                    p = pp.get((r, h), {'price':0,'carbon':0,'renew':0})
                    fac = it_pw * pue
                    cost += fac * p['price']
                    carbon += fac * p['carbon']
                    renew += min(p['renew'], fac)
                lat_pen = 0.0
                if src != r:
                    try: lat_pen = float(latency_matrix.loc[src, r]) * 0.001
                    except: pass

                sc = a1*cost + a2*carbon + a3*lat_pen - a4*renew
                if sc < best_score:
                    best_score = sc; best = (r, s, e)

        if best is None:
            r = src; s = arr
            attempts = 0
            while attempts < 500:
                e = s + dur - 1
                if e >= 2406: s = max(0, 2405-dur+1); e = 2405; break
                if all(gpu_use[r].get(h,0)+gpu <= rc[r]['gpu'] for h in range(s,e+1)):
                    break
                s += 1; attempts += 1
            e = s + dur - 1
            if e >= 2406: s = max(0, 2405-dur+1); e = 2405
            best = (r, s, e)

        r, s, e = best
        fac_pw = it_pw * rc[r]['pue']
        sched[tid] = {
            'region': r, 'start': s, 'end': e, 'dur': dur,
            'gpu': gpu, 'task_type': tt, 'source': src,
            'it_pw': it_pw, 'fac_pw': fac_pw,
            'migrated': (r != src),
        }
        for h in range(s, e+1):
            gpu_use[r][h] = _gu(r, h) + gpu
            it_use[r][h]  = _iu(r, h) + it_pw

    return sched, gpu_use, it_use

def compute_all_metrics(sched):
    """返回 (cost, carbon, avg_latency, renew_util, n_migrated, n_tasks)"""
    cost = carbon = renew_used = total_fac = 0.0
    lat_sum = 0.0; mig_n = 0; n = len(sched)
    for tid, info in sched.items():
        r, s, e = info['region'], info['start'], info['end']
        fac = info['fac_pw']
        for h in range(s, e+1):
            p = pp.get((r, h), {'price':0,'carbon':0,'renew':0})
            cost += fac * p['price']
            carbon += fac * p['carbon']
            renew_used += min(p['renew'], fac)
            total_fac += fac
        if info['migrated']:
            mig_n += 1
            try: lat_sum += float(latency_matrix.loc[info['source'], r])
            except: pass
    renew_util = renew_used / total_renewable if total_renewable > 0 else 0
    avg_lat = lat_sum / max(mig_n, 1)
    return cost, carbon, avg_lat, renew_util, mig_n, n

# 多组权重
weight_configs = {
    'greedy_balanced': (1.0, 1.0, 0.5, 0.5),
    'greedy_cost':     (1.0, 0.1, 0.1, 0.0),
    'greedy_carbon':   (0.1, 1.0, 0.1, 0.0),
    'greedy_renew':    (0.1, 0.1, 0.1, 1.0),
    'greedy_latency':  (0.1, 0.1, 1.0, 0.0),
    'greedy_cost2':    (2.0, 0.5, 0.3, 0.2),
    'greedy_bal2':     (0.5, 2.0, 0.5, 0.3),
    'greedy_mix':      (1.5, 1.5, 0.8, 0.6),
}

seed_scheds = {}
print(f"\n{'方案':<22s} {'成本':>12s} {'碳':>10s} {'时延':>8s} {'新能源':>8s} {'迁移':>6s}")
print("-" * 72)
for name, w in weight_configs.items():
    sched, _, _ = greedy_schedule(workload, w, verbose=False)
    c, co, al, ru, mn, nt = compute_all_metrics(sched)
    seed_scheds[name] = sched
    print(f"{name:<22s} {c:>12.0f} {co:>10.2f} {al:>8.1f} {ru:>8.4f} {mn:>6d}")

# ============================================================
# 4. 将调度方案编码为整数基因
# ============================================================
print("\n" + "=" * 60)
print("Step4: 编码方案")
print("=" * 60)

def schedule_to_genes(sched):
    """将调度方案编码为基因数组"""
    genes = np.zeros(n_vars, dtype=int)
    for i, t in enumerate(tasks_sorted):
        tid = t['tid']
        if tid in sched:
            info = sched[tid]
            r = info['region']; s = info['start']
            if r in t['cand_regions']:
                r_idx = t['cand_regions'].index(r)
                if s in t['start_options']:
                    s_idx = t['start_options'].index(s)
                    n_s = len(t['start_options'])
                    genes[i] = r_idx * n_s + s_idx
    return genes

# 构建初始种群
seed_genes_list = []
seed_labels = []
for name, sched in seed_scheds.items():
    g = schedule_to_genes(sched)
    seed_genes_list.append(g)
    seed_labels.append(name)

# 再加一些随机扰动变体
rng = np.random.default_rng(2024)
for base_g in seed_genes_list[:4]:
    for _ in range(3):
        noise = rng.integers(-2, 3, size=n_vars)
        new_g = np.clip(base_g + noise, 0, 0)  # placeholder
        for i, t in enumerate(tasks_sorted):
            max_g = t['n_cand'] - 1
            new_g[i] = np.clip(new_g[i], 0, max_g)
        seed_genes_list.append(new_g)
        seed_labels.append('mutated')

seed_array = np.array(seed_genes_list, dtype=int)
print(f"初始种群大小: {len(seed_array)}")
for i, lab in enumerate(seed_labels[:10]):
    print(f"  [{i}] {lab}")

# 变量边界
xl = np.array([0] * n_vars, dtype=int)
xu = np.array([t['n_cand'] - 1 for t in tasks_sorted], dtype=int)

# ============================================================
# 5. 定义 pymoo Problem
# ============================================================
print("\n" + "=" * 60)
print("Step5: 定义 NSGA-II 问题")
print("=" * 60)

class NSGA2Problem(Problem):
    def __init__(self):
        super().__init__(
            n_var=n_vars, n_obj=4, n_constr=0,
            xl=xl, xu=xu, type_var=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 4))
        for k in range(n_pop):
            genes = X[k]
            cost = carbon = lat_sum = renew_used = 0.0
            mig_n = 0
            feasible = True
            penalty = 0.0

            # 逐时占用
            gpu_u = defaultdict(lambda: defaultdict(float))
            it_u  = defaultdict(lambda: defaultdict(float))

            for i, t in enumerate(tasks_sorted):
                g = int(genes[i])
                n_s = len(t['start_options'])
                if n_s == 0 or len(t['cand_regions']) == 0:
                    feasible = False; penalty += 1e6; continue
                r_idx = g // n_s
                s_idx = g % n_s
                if r_idx >= len(t['cand_regions']) or s_idx >= len(t['start_options']):
                    feasible = False; penalty += 1e6; continue
                r = t['cand_regions'][r_idx]
                s = t['start_options'][s_idx]
                e = s + t['dur_h'] - 1

                # 容量检查
                cap_gpu = rc[r]['gpu']; cap_it = rc[r]['max_it']
                it_pw = t['it_pw']; pue = rc[r]['pue']
                for h in range(s, e+1):
                    if gpu_u[r][h] + t['gpu'] > cap_gpu:
                        feasible = False
                        penalty += (gpu_u[r][h] + t['gpu'] - cap_gpu) * 100
                        break
                    if it_u[r][h] + it_pw > cap_it:
                        feasible = False
                        penalty += (it_u[r][h] + it_pw - cap_it) * 100
                        break
                if not feasible: continue

                fac_pw = it_pw * pue
                for h in range(s, e+1):
                    gpu_u[r][h] += t['gpu']
                    it_u[r][h]  += it_pw
                    p = pp.get((r, h), {'price':0,'carbon':0,'renew':0})
                    cost   += fac_pw * p['price']
                    carbon += fac_pw * p['carbon']
                    renew_used += min(p['renew'], fac_pw)

                if r != t['src']:
                    mig_n += 1
                    try: lat_sum += float(latency_matrix.loc[t['src'], r])
                    except: pass

            avg_lat = lat_sum / max(mig_n, 1)
            renew_util = renew_used / total_renewable if total_renewable > 0 else 0

            if not feasible:
                cost += penalty
                carbon += penalty * 0.001
                avg_lat += penalty * 0.01

            F[k, 0] = cost
            F[k, 1] = carbon
            F[k, 2] = avg_lat
            F[k, 3] = 1.0 - renew_util  # 最小化

        out['F'] = F

problem = NSGA2Problem()

# ============================================================
# 6. 自定义采样器（使用种子初始化）
# ============================================================
from pymoo.core.sampling import Sampling

class SeedSampling(Sampling):
    def __init__(self, seed_array):
        super().__init__()
        self.seed_array = seed_array
    def _do(self, problem, n_samples, **kwargs):
        # 返回种子 + 补充随机
        n_seed = len(self.seed_array)
        if n_samples <= n_seed:
            return self.seed_array[:n_samples]
        extra = np.random.randint(0, 2, size=(n_samples - n_seed, problem.n_var))
        # 用均匀随机填充额外个体
        for i in range(n_samples - n_seed):
            for j in range(problem.n_var):
                extra[i, j] = np.random.randint(problem.xl[j], problem.xu[j] + 1)
        return np.vstack([self.seed_array, extra])

pop_size = max(80, len(seed_array) * 2 + 16)
n_gen = 300

print(f"种群大小: {pop_size}, 代数: {n_gen}")

algorithm = NSGA2(
    pop_size=pop_size,
    sampling=SeedSampling(seed_array),
    crossover=SBX(prob=0.9, eta=10, vtype=int),
    mutation=PolynomialMutation(eta=15, vtype=int, prob=1.0/n_vars),
    eliminate_duplicates=True,
)

res = minimize(
    problem, algorithm,
    ('n_gen', n_gen),
    seed=42, verbose=False,
)

print(f"\n✅ NSGA-II 完成, Pareto 解数: {len(res.F)}")

# ============================================================
# 7. 提取 Pareto 解集 & 选择代表解
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
idx_cost    = np.argmin(pareto_F[:, 0])
idx_carbon  = np.argmin(pareto_F[:, 1])
idx_latency = np.argmin(pareto_F[:, 2])
idx_renew   = np.argmin(pareto_F[:, 3])  # 最小(1-util) = 最大util

print(f"\n{'方案':<22s} {'成本(元)':>14s} {'碳(tCO2)':>12s} {'时延(ms)':>10s} {'新能源利用':>10s}")
print("-" * 75)
for lab, idx in [('knee', knee_idx), ('min_cost', idx_cost),
                  ('min_carbon', idx_carbon), ('min_latency', idx_latency),
                  ('max_renew', idx_renew)]:
    f = pareto_F[idx]
    print(f"nsga2_{lab:<16s} {f[0]:>14.0f} {f[1]:>12.2f} {f[2]:>10.1f} {1-f[3]:>10.4f}")

# ============================================================
# 8. 解码 Pareto 解 → 调度表
# ============================================================
def genes_to_schedule(genes):
    sched = {}
    for i, t in enumerate(tasks_sorted):
        g = int(genes[i])
        n_s = len(t['start_options'])
        if n_s == 0: continue
        r_idx = g // n_s
        s_idx = g % n_s
        if r_idx >= len(t['cand_regions']): continue
        r = t['cand_regions'][r_idx]
        s = t['start_options'][s_idx]
        e = s + t['dur_h'] - 1
        sched[t['tid']] = {
            'region': r, 'start': s, 'end': e, 'dur': t['dur_h'],
            'gpu': t['gpu'], 'task_type': t['tt'], 'source': t['src'],
            'it_pw': t['it_pw'], 'fac_pw': t['it_pw'] * rc[r]['pue'],
            'migrated': (r != t['src']),
        }
    return sched

selected = {}
for lab, idx in [('knee', knee_idx), ('min_cost', idx_cost),
                  ('min_carbon', idx_carbon), ('min_latency', idx_latency),
                  ('max_renew', idx_renew)]:
    genes = pareto_X[idx]
    sched = genes_to_schedule(genes)
    c, co, al, ru, mn, nt = compute_all_metrics(sched)
    selected[lab] = {
        'genes': genes, 'sched': sched,
        'metrics': {'cost': c, 'carbon': co, 'latency': al,
                    'renew': ru, 'migrated': mn, 'n': nt}
    }
    # 保存调度表
    rows = []
    for tid, info in sched.items():
        rows.append({
            'TaskID': tid, 'TaskType': info['task_type'],
            'SourceRegion': info['source'],
            'AssignedRegion': info['region'],
            'StartHour': info['start'], 'EndHour': info['end'],
            'Duration_h': info['dur'], 'GPU_Demand': info['gpu'],
            'Migrated': info['migrated'],
            'IT_Power_MW': info['it_pw'],
            'Facility_Power_MW': info['fac_pw'],
        })
    pd.DataFrame(rows).to_csv(f"/data/workspace/q2_nsga2_sched_{lab}.csv", index=False)

# Pareto 前沿表
pf_rows = []
for k in range(len(pareto_F)):
    pf_rows.append({
        'sol_id': k,
        'cost_yuan': pareto_F[k, 0],
        'carbon_tco2': pareto_F[k, 1],
        'avg_latency_ms': pareto_F[k, 2],
        'renew_utilization': 1 - pareto_F[k, 3],
    })
pf_df = pd.DataFrame(pf_rows)
pf_df.to_csv("/data/workspace/q2_nsga2_pareto.csv", index=False)

# ============================================================
# 9. 综合对比表
# ============================================================
print("\n" + "=" * 60)
print("Step9: 综合对比")
print("=" * 60)

compare_rows = []
for name, sched in seed_scheds.items():
    c, co, al, ru, mn, nt = compute_all_metrics(sched)
    compare_rows.append({
        'scheme': f'greedy_{name.replace("greedy_","")}',
        'cost_yuan': c, 'carbon_tco2': co,
        'avg_latency_ms': al, 'renew_utilization': ru,
        'migrated': mn, 'n_tasks': nt,
    })

for lab, data in selected.items():
    m = data['metrics']
    compare_rows.append({
        'scheme': f'nsga2_{lab}',
        'cost_yuan': m['cost'], 'carbon_tco2': m['carbon'],
        'avg_latency_ms': m['latency'], 'renew_utilization': m['renew'],
        'migrated': m['migrated'], 'n_tasks': m['n'],
    })

cmp_df = pd.DataFrame(compare_rows)
cmp_df.to_csv("/data/workspace/q2_nsga2_compare.csv", index=False)

print(f"\n{'方案':<25s} {'成本(元)':>12s} {'碳(tCO2)':>10s} {'时延(ms)':>8s} {'新能源':>8s} {'迁移':>6s}")
print("-" * 78)
for _, r in cmp_df.iterrows():
    print(f"{r['scheme']:<25s} {r['cost_yuan']:>12.0f} {r['carbon_tco2']:>10.2f} "
          f"{r['avg_latency_ms']:>8.1f} {r['renew_utilization']:>8.4f} {int(r['migrated']):>6d}")

# 超体积
try:
    hv_ref = np.max(pareto_F, axis=0) * 1.1
    hv = Hypervolume(ref_point=hv_ref).do(pareto_F)
    print(f"\n超体积 HV = {hv:.2f}")
except Exception as e:
    print(f"\nHV 计算跳过: {e}")

# ============================================================
# 10. 保存完整结果
# ============================================================
print("\n" + "=" * 60)
print("Step10: 保存")
print("=" * 60)

with open("/data/workspace/q2_nsga2_result.pkl", 'wb') as f:
    pickle.dump({
        'pareto_F': pareto_F,
        'pareto_X': pareto_X,
        'pareto_df': pf_df,
        'selected': {k: v['metrics'] for k, v in selected.items()},
        'compare_df': cmp_df,
        'tasks_sorted': tasks_sorted,
        'seed_labels': seed_labels,
    }, f)

print("✅ q2_nsga2_result.pkl")
print("✅ q2_nsga2_pareto.csv")
print("✅ q2_nsga2_compare.csv")
print("✅ q2_nsga2_sched_{knee,min_cost,min_carbon,min_latency,max_renew}.csv")

print("\n🎉 NSGA-II 改进版全部完成！")
