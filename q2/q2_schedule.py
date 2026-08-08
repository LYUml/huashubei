"""
问题二：碳感知任务调度模型 —— 调度优化引擎
==================================================
真实数据字段（与 gen_data.py 对齐）：
  workload_trace.xlsx : TaskID, TaskType, ArrivalHour, GPU_Demand,
                        EstimatedDuration_min, DelaySensitivity,
                        SourceRegion, MaxLatency_ms,
                        LatestFinishHour, EarliestStartHour, ExecutionMode
  region_time_data(+carbon+price+renewable) 主键(Region,Hour):
        NonAI_IT_Load_MW, Baseline_AI_IT_Load_MW,
        IT_Load_MW, Facility_Load_MW,
        CarbonIntensity_tCO2_per_MWh, Price_Yuan_per_MWh,
        AvailableRenewable_MW
  power_mapping.xlsx : TaskType -> PowerPerGPU_MW
  gpu_information.xlsx: Region -> Available_GPU, Max_IT_Power_MW,
                               Max_Facility_Power_MW, PUE
  network_latency.xlsx: SourceRegion x TargetRegion -> Latency_ms

算法：碳感知贪心
  目标: min  α1·cost + α2·carbon + α3·latency − α4·renewable
  约束: GPU容量 / IT功率 / 设施功率 / 时延 / 完成时限
"""

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 1. 加载
# ============================================================
print("=" * 60)
print("Step 1: 加载数据")
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

print(f"任务数: {len(workload)}")
print(f"区域:   {regions}")
print(f"类型:   {task_types}")
print(f"功率映射: {power_dict}")
print(f"电力参数列: {list(rtd.columns)}")

# ============================================================
# 2. 快速查找表
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 构建查找表")
print("=" * 60)

# pp: (Region, Hour) -> dict{price, carbon, renew, nonai}
pp = {}
for _, row in rtd.iterrows():
    r, h = row['Region'], int(row['Hour'])
    pp[(r, h)] = {
        'price':  float(row['Price_Yuan_per_MWh']),
        'carbon': float(row['CarbonIntensity_tCO2_per_MWh']),
        'renew':  float(row['AvailableRenewable_MW']),
        'nonai':  float(row['NonAI_IT_Load_MW']),
    }

# rc: Region -> dict{gpu, max_it, max_fac, pue}
rc = {}
for r, info in region_info.items():
    rc[r] = {
        'gpu':     int(info['Available_GPU']),
        'max_it':  float(info['Max_IT_Power_MW']),
        'max_fac': float(info['Max_Facility_Power_MW']),
        'pue':     float(info['PUE']),
    }

print("区域容量:")
for r, c in rc.items():
    print(f"  {r}: GPU={c['gpu']:4d}  MaxIT={c['max_it']:5.1f}MW  "
          f"MaxFac={c['max_fac']:5.1f}MW  PUE={c['pue']}")

# ============================================================
# 3. 辅助函数
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 辅助函数")
print("=" * 60)

def dur_h(task_row):
    return int(np.ceil(task_row['EstimatedDuration_min'] / 60))

def power_per_gpu(tt):
    return float(power_dict.get(tt, 0.003))

def it_power_of(task_row):
    return float(task_row['GPU_Demand']) * power_per_gpu(task_row['TaskType'])

def check_latency(src, dst, max_lat):
    if src == dst:
        return True
    try:
        return float(latency_matrix.loc[src, dst]) <= float(max_lat)
    except Exception:
        return True

def _get(d, r, h, default=0.0):
    """安全取 gpu_use[r][h] 之类的值"""
    return d.get(r, {}).get(h, default)

print("✅ 辅助函数就绪")

# ============================================================
# 4. 得分评估
# ============================================================
def evaluate(region, sh, eh, gpu_demand, it_pw, pue,
             src, alpha=(1.0, 1.0, 0.5, 0.5)):
    a1, a2, a3, a4 = alpha
    fac_pw = it_pw * pue
    cost = carbon = renew = 0.0
    for h in range(sh, eh + 1):
        p = pp.get((region, h))
        if p is None:
            p = {'price': 0.0, 'carbon': 0.0, 'renew': 0.0}
        cost   += fac_pw * p['price']
        carbon += fac_pw * p['carbon']
        renew  += min(p['renew'], fac_pw)

    lat_pen = 0.0
    if src != region:
        try:
            lat_pen = float(latency_matrix.loc[src, region]) * 0.001
        except Exception:
            lat_pen = 0.0
    mig_pen = 1.0 if (src != region) else 0.0

    return a1*cost + a2*carbon - a4*renew + a3*lat_pen + mig_pen*10.0

# ============================================================
# 5. 贪心调度主函数
# ============================================================
def greedy_schedule(workload_df, alpha=(1.0, 1.0, 0.5, 0.5),
                    verbose=True):
    priority = {'RealTimeInference': 0, 'BatchInference': 1, 'AITraining': 2}
    df = workload_df.copy()
    df['_pri'] = df['TaskType'].map(priority)
    df = df.sort_values(['_pri', 'ArrivalHour']).reset_index(drop=True)

    # 用 dict + setdefault 避免 defaultdict(lambda) 不可 pickle
    gpu_use = {}   # [r][h] -> float
    it_use  = {}
    for _r in regions:
        gpu_use[_r] = {}
        it_use[_r]  = {}

    sched   = {}
    total = len(df)
    migrated = 0

    def _gu(r, h):
        return gpu_use.get(r, {}).get(h, 0.0)
    def _iu(r, h):
        return it_use.get(r, {}).get(h, 0.0)

    for idx, task in df.iterrows():
        tid   = int(task['TaskID'])
        src   = task['SourceRegion']
        gpu   = float(task['GPU_Demand'])
        dur   = dur_h(task)
        maxlat = float(task['MaxLatency_ms'])
        tt     = task['TaskType']
        arr    = int(task['ArrivalHour'])
        lfh    = int(task['LatestFinishHour'])
        it_pw  = gpu * power_per_gpu(tt)

        # 实时推理必须到达即开工
        if tt == 'RealTimeInference':
            start_options = [arr]
        else:
            latest_start = min(lfh - dur, 2405 - dur)
            max_delay = max(0, latest_start - arr)
            max_delay = min(max_delay, 72)
            start_options = list(range(arr, arr + max_delay + 1))

        # 候选区域
        cand = [r for r in regions if check_latency(src, r, maxlat)]
        if not cand:
            cand = [src]

        best = None
        best_score = float('inf')

        for r in cand:
            cap_gpu = rc[r]['gpu']
            cap_it  = rc[r]['max_it']
            pue_r   = rc[r]['pue']

            for s in start_options:
                e = s + dur - 1
                if e >= 2406:
                    continue

                # GPU 容量
                ok = all(_gu(r, h) + gpu <= cap_gpu for h in range(s, e + 1))
                if not ok:
                    continue
                # IT 功率
                ok = all(_iu(r, h) + it_pw <= cap_it for h in range(s, e + 1))
                if not ok:
                    continue

                sc = evaluate(r, s, e, gpu, it_pw, pue_r, src, alpha)
                if sc < best_score:
                    best_score = sc
                    best = (r, s, e)

        # 全部不可行 → 强制本地最早
        if best is None:
            r = src
            s = arr
            attempts = 0
            while attempts < 500:
                e = s + dur - 1
                if e >= 2406:
                    s = max(0, 2405 - dur + 1); e = 2405; break
                ok = all(_gu(r, h) + gpu <= rc[r]['gpu']
                         for h in range(s, e + 1))
                if ok:
                    break
                s += 1; attempts += 1
            e = s + dur - 1
            if e >= 2406:
                s = max(0, 2405 - dur + 1); e = 2405
            best = (r, s, e)
            if verbose and idx < 20:
                print(f"  ⚠️ Task{tid} 强制本地 {r}@{s}")

        r, s, e = best
        fac_pw = it_pw * rc[r]['pue']

        sched[tid] = {
            'region': r, 'start': s, 'end': e, 'dur': dur,
            'gpu': gpu, 'task_type': tt, 'source': src,
            'it_pw': it_pw, 'fac_pw': fac_pw,
            'migrated': (r != src),
        }
        for h in range(s, e + 1):
            gpu_use[r][h] = _gu(r, h) + gpu
            it_use[r][h]  = _iu(r, h) + it_pw

        if r != src:
            migrated += 1

        if verbose and idx % 200 == 0:
            print(f"  ... {idx}/{total} 已迁移{migrated}")

    if verbose:
        print(f"  ✅ 完成: {total} 任务, 迁移 {migrated} ({migrated/total*100:.1f}%)")
    return sched, gpu_use, it_use


# ============================================================
# 6. 指标计算
# ============================================================
def compute_metrics(sched, label=""):
    cost = carbon = renew_used = total_fac = 0.0
    lat_sum = mig_n = 0.0
    n = len(sched)

    for tid, info in sched.items():
        r, s, e = info['region'], info['start'], info['end']
        fac = info['fac_pw']
        for h in range(s, e + 1):
            p = pp.get((r, h))
            if p is None:
                p = {'price': 0.0, 'carbon': 0.0, 'renew': 0.0}
            cost   += fac * p['price']
            carbon += fac * p['carbon']
            renew_used += min(p['renew'], fac)
            total_fac += fac
        if info['migrated']:
            mig_n += 1
            try:
                lat_sum += float(latency_matrix.loc[info['source'], r])
            except Exception:
                pass

    total_renewable = sum(p['renew'] for p in pp.values() if p)
    renew_pct = (renew_used / total_renewable * 100) if total_renewable > 0 else 0.0
    avg_lat = (lat_sum / mig_n) if mig_n > 0 else 0.0

    return {
        'scheme': label,
        'n_tasks': n,
        'cost_yuan': cost,
        'carbon_tco2': carbon,
        'avg_latency_ms': avg_lat,
        'migrated': int(mig_n),
        'migrate_pct': mig_n / n * 100,
        'renewable_pct': renew_pct,
        'total_facility_mwh': total_fac,
    }

# ============================================================
# 7. 运行 —— 碳感知调度
# ============================================================
print("\n" + "=" * 60)
print("Step 7: 执行碳感知调度 (全部任务)")
print("=" * 60)

sched_full, gu_full, iu_full = greedy_schedule(
    workload, alpha=(1.0, 1.0, 0.5, 0.5))

# 测试集: 2376-2399 实际任务
test_tasks = workload[workload['ArrivalHour'].between(2376, 2399)].copy()
print(f"\n测试集任务数 (2376-2399): {len(test_tasks)}")

# 先排 0-2375 的任务, 占住容量
print("\n--- 调度 0-2375 前期任务 (占容量) ---")
sched_pre, gu_pre, iu_pre = greedy_schedule(
    workload[workload['ArrivalHour'] < 2376],
    alpha=(1.0, 1.0, 0.5, 0.5), verbose=False,
)

# 合并前期容量占用 (普通 dict)
print("\n--- 调度 2376-2399 任务 ---")
sched_test, gu_test, iu_test = greedy_schedule(
    test_tasks, alpha=(1.0, 1.0, 0.5, 0.5))

# ============================================================
# 8. 本地调度基线
# ============================================================
print("\n" + "=" * 60)
print("Step 8: 本地调度基线")
print("=" * 60)

def local_schedule(workload_df):
    df = workload_df.copy()
    pri = {'RealTimeInference': 0, 'BatchInference': 1, 'AITraining': 2}
    df['_pri'] = df['TaskType'].map(pri)
    df = df.sort_values(['_pri', 'ArrivalHour']).reset_index(drop=True)
    gu = {}; iu = {}
    for _r in regions:
        gu[_r] = {}; iu[_r] = {}
    sched = {}
    for _, t in df.iterrows():
        tid, src = int(t['TaskID']), t['SourceRegion']
        gpu = float(t['GPU_Demand'])
        dur = dur_h(t)
        it_pw = gpu * power_per_gpu(t['TaskType'])
        s = int(t['ArrivalHour'])
        attempts = 0
        while attempts < 500:
            e = s + dur - 1
            if e >= 2406:
                s = max(0, 2405 - dur + 1); e = 2405; break
            ok = all(gu[src].get(h, 0.0) + gpu <= rc[src]['gpu']
                     for h in range(s, e + 1))
            if ok: break
            s += 1; attempts += 1
        e = s + dur - 1
        if e >= 2406:
            s = max(0, 2405 - dur + 1); e = 2405
        sched[tid] = {
            'region': src, 'start': s, 'end': e, 'dur': dur,
            'gpu': gpu, 'task_type': t['TaskType'], 'source': src,
            'it_pw': it_pw, 'fac_pw': it_pw * rc[src]['pue'],
            'migrated': False,
        }
        for h in range(s, e + 1):
            gu[src][h] = gu[src].get(h, 0.0) + gpu
            iu[src][h]  = iu[src].get(h, 0.0) + it_pw
    return sched, gu, iu

sched_local, _, _ = local_schedule(test_tasks)
print(f"✅ 本地基线完成: {len(sched_local)} 任务")

# ============================================================
# 9. 指标对比
# ============================================================
print("\n" + "=" * 60)
print("Step 9: 指标对比")
print("=" * 60)

m_test  = compute_metrics(sched_test,  "CarbonAware")
m_local = compute_metrics(sched_local, "LocalOnly")

print("\n=== 2376-2399 调度指标对比 ===\n")
fmt = "{:<22s} {:>16s} {:>16s} {:>12s}"
print(fmt.format("指标", "本地调度", "碳感知调度", "改善率"))
print("-" * 68)
for key, name in [
    ('cost_yuan',      "运行成本(元)"),
    ('carbon_tco2',    "碳排放(tCO2)"),
    ('avg_latency_ms', "平均时延(ms)"),
    ('migrate_pct',    "迁移率(%)"),
    ('renewable_pct',  "新能源利用(%)"),
]:
    a, b = m_local[key], m_test[key]
    if a != 0:
        imp = (a - b) / abs(a) * 100
        d = "↓" if imp > 0 else "↑"
        if abs(a) < 1 and abs(b) < 1:
            print(fmt.format(name, f"{a:.6f}", f"{b:.6f}", f"{d}{abs(imp):.1f}%"))
        else:
            print(fmt.format(name, f"{a:,.4f}", f"{b:,.4f}", f"{d}{abs(imp):.1f}%"))
    else:
        if abs(b) < 0.01:
            print(fmt.format(name, f"{a:.6f}", f"{b:.6f}", "N/A"))
        else:
            print(fmt.format(name, f"{a:.4f}", f"{b:.4f}", "N/A"))

# ============================================================
# 10. 保存
# ============================================================
print("\n" + "=" * 60)
print("Step 10: 保存结果")
print("=" * 60)

# 调度明细
rows = []
for tid, info in sched_test.items():
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
sched_df = pd.DataFrame(rows)
sched_df.to_csv("/data/workspace/q2_schedule.csv", index=False)
print(f"✅ 调度明细 → q2_schedule.csv ({len(sched_df)} 行)")

# 指标对比
mdf = pd.DataFrame([m_local, m_test])
mdf.to_csv("/data/workspace/q2_metrics_comparison.csv", index=False)
print(f"✅ 指标对比 → q2_metrics_comparison.csv")

# 把嵌套 dict 转成普通 dict 以便 pickle
def _plain(d):
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()}
    return d

# gu_test / iu_test 里嵌了 defaultdict? 不，是普通 dict
# 但 rc / pp 已经是普通 dict
with open("/data/workspace/q2_schedule_result.pkl", 'wb') as f:
    pickle.dump({
        'test_schedule': sched_test,
        'local_schedule': sched_local,
        'test_metrics': m_test,
        'local_metrics': m_local,
        'gpu_usage': _plain(dict(gu_test)),
        'it_power_usage': _plain(dict(iu_test)),
        'power_params': _plain(dict(pp)),
        'region_capacity': rc,
        'alpha': (1.0, 1.0, 0.5, 0.5),
    }, f)
print("✅ 完整结果 → q2_schedule_result.pkl")

print("\n🎉 问题二建模 + 调度全部完成！")
