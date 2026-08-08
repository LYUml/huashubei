"""
========================================================
问题2：碳感知任务调度模型 - 核心求解器 v2
========================================================
多目标加权贪心调度算法
目标：最小化运行成本、碳排放；最大化新能源利用率；控制网络时延
"""

import pandas as pd
import numpy as np
import time
import sys

# ============================================================
# 1. 数据加载
# ============================================================

def load_all_data():
    wt = pd.read_excel('/data/inputs/workload_trace.xlsx')
    rtd = pd.read_excel('/data/inputs/region_time_data.xlsx')
    gpu_info = pd.read_excel('/data/inputs/GPU_information.xlsx')
    nl = pd.read_excel('/data/inputs/network_latency.xlsx')
    pm = pd.read_excel('/data/inputs/power_mapping.xlsx')
    si = pd.read_excel('/data/inputs/storage_information.xlsx')

    wt['Duration_h'] = np.ceil(wt['EstimatedDuration_min'] / 60).astype(int)

    rtd_main = rtd[rtd['DataPeriod'] == 'Main_0_2399'].copy().set_index(['Hour', 'Region'])
    rtd_close = rtd[rtd['DataPeriod'] == 'Closure_2400_2406'].copy().set_index(['Hour', 'Region'])

    pue_map = dict(zip(gpu_info['Region'], gpu_info['PUE']))
    avail_gpu_map = dict(zip(gpu_info['Region'], gpu_info['Available_GPU']))
    max_it_power_map = dict(zip(gpu_info['Region'], gpu_info['Max_IT_Power_MW']))
    max_fac_power_map = dict(zip(gpu_info['Region'], gpu_info['Max_Facility_Power_MW']))
    max_import_map = dict(zip(si['Region'], si['MaxGridImport_MW']))
    max_export_map = dict(zip(si['Region'], si['MaxGridExport_MW']))

    power_map = dict(zip(pm['TaskType'], pm['GPU_Power_MW_per_EquivalentGPU']))

    latency_dict = {}
    for _, row in nl.iterrows():
        latency_dict[(row['FromRegion'], row['ToRegion'])] = row['NetworkLatency_ms']

    regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']

    return {
        'workload': wt,
        'rtd_main': rtd_main,
        'rtd_close': rtd_close,
        'pue_map': pue_map,
        'avail_gpu_map': avail_gpu_map,
        'max_it_power_map': max_it_power_map,
        'max_fac_power_map': max_fac_power_map,
        'max_import_map': max_import_map,
        'max_export_map': max_export_map,
        'power_map': power_map,
        'latency_dict': latency_dict,
        'regions': regions,
    }


# ============================================================
# 2. 区域状态类
# ============================================================

class RegionState:
    def __init__(self, region):
        self.region = region
        self.gpu_usage = np.zeros(2406, dtype=np.float64)
        self.it_power_usage = np.zeros(2406, dtype=np.float64)


# ============================================================
# 3. 可行区域筛选
# ============================================================

def get_feasible_regions(task, latency_dict, regions):
    src = task['SourceRegion']
    max_lat = task['MaxLatency_ms']
    feasible = []
    for r in regions:
        if latency_dict.get((src, r), 999) <= max_lat:
            feasible.append(r)
    return feasible


# ============================================================
# 4. 容量约束检查
# ============================================================

def check_capacity(state, task, start_hour, data):
    duration = int(task['Duration_h'])
    gpu_demand = task['GPU_Demand']
    task_type = task['TaskType']
    power_per_gpu = data['power_map'][task_type]
    region = None  # will be set by caller
    return True  # simplified, full check below


def check_capacity_full(state, task, target_region, start_hour, data):
    duration = int(task['Duration_h'])
    gpu_demand = task['GPU_Demand']
    task_type = task['TaskType']
    power_per_gpu = data['power_map'][task_type]
    pue = data['pue_map'][target_region]
    avail_gpu = data['avail_gpu_map'][target_region]
    max_it = data['max_it_power_map'][target_region]
    max_fac = data['max_fac_power_map'][target_region]

    end_hour = min(start_hour + duration - 1, 2405)

    for h in range(start_hour, end_hour + 1):
        new_gpu = state.gpu_usage[h] + gpu_demand
        if new_gpu > avail_gpu:
            return False

        new_it = state.it_power_usage[h] + gpu_demand * power_per_gpu

        if h <= 2399:
            nonai = data['rtd_main'].loc[(h, target_region)]['NonAI_IT_Load_MW']
        else:
            if (h, target_region) in data['rtd_close'].index:
                nonai = data['rtd_close'].loc[(h, target_region)]['NonAI_IT_Load_MW']
            else:
                nonai = 0

        if new_it + nonai > max_it:
            return False

        new_fac = (new_it + nonai) * pue
        if new_fac > max_fac:
            return False

    return True


# ============================================================
# 5. 增量成本/碳排放计算
# ============================================================

def compute_incremental(task, target_region, start_hour, data):
    duration = int(task['Duration_h'])
    gpu_demand = task['GPU_Demand']
    task_type = task['TaskType']
    power_per_gpu = data['power_map'][task_type]
    pue = data['pue_map'][target_region]

    end_hour = min(start_hour + duration - 1, 2405)

    incr_cost = 0.0
    incr_carbon = 0.0
    incr_renew = 0.0

    for h in range(start_hour, end_hour + 1):
        if h <= 2399:
            row = data['rtd_main'].loc[(h, target_region)]
        else:
            if (h, target_region) in data['rtd_close'].index:
                row = data['rtd_close'].loc[(h, target_region)]
            else:
                continue

        ai_power = gpu_demand * power_per_gpu  # MW
        facility_load = ai_power * pue  # MW

        incr_cost += facility_load * row['ElectricityPrice_CNY_per_MWh']
        incr_carbon += facility_load * row['CarbonIntensity_tCO2_per_MWh']

        avail_r = row['AvailableRenewable_MW']
        if avail_r > 0:
            incr_renew += min(ai_power, avail_r)

    src = task['SourceRegion']
    latency = data['latency_dict'].get((src, target_region), 999)

    return incr_cost, incr_carbon, latency, incr_renew


# ============================================================
# 6. 单任务调度决策
# ============================================================

def schedule_task(task, data, region_states, weights):
    feasible = get_feasible_regions(task, data['latency_dict'], data['regions'])

    best_score = float('inf')
    best_decision = None
    best_details = None

    src = task['SourceRegion']
    arrival = int(task['ArrivalHour'])
    latest_finish = int(task['LatestFinishHour'])
    duration = int(task['Duration_h'])
    task_type = task['TaskType']

    if task_type == 'RealTimeInference':
        start_candidates = [arrival]
    else:
        earliest = int(task.get('EarliestStartHour', arrival))
        latest_start = min(latest_finish - duration + 1, 2405)
        latest_start = min(latest_start, arrival + 300)
        start_candidates = list(range(earliest, latest_start + 1))

    for target_r in feasible:
        for s_h in start_candidates:
            s_h = int(s_h)
            if s_h + duration - 1 > 2405:
                continue
            if not check_capacity_full(region_states[target_r], task, target_r, s_h, data):
                continue

            cost, carbon, latency, renew = compute_incremental(task, target_r, s_h, data)

            # 归一化
            cost_norm = cost / 1000.0

            if target_r != src:
                lat_pen = latency * weights.get('latency_factor', 1.0)
            else:
                lat_pen = 0.0

            score = (
                weights['w_cost'] * cost_norm +
                weights['w_carbon'] * carbon +
                weights['w_latency'] * lat_pen +
                weights['w_renewable'] * (-renew / 100.0)
            )

            if score < best_score:
                best_score = score
                best_decision = (target_r, s_h)
                best_details = {
                    'cost': cost,
                    'carbon': carbon,
                    'latency': latency,
                    'renewable': renew,
                    'is_migrated': (target_r != src),
                }

    if best_decision is None:
        # 兜底：本地执行
        best_decision = (src, arrival)
        best_details = {'cost': 0, 'carbon': 0, 'latency': 0, 'renewable': 0, 'is_migrated': False}

    return best_decision, best_details


# ============================================================
# 7. 应用调度
# ============================================================

def apply_schedule(region_states, task, decision):
    target_r, s_h = decision
    duration = int(task['Duration_h'])
    gpu_demand = task['GPU_Demand']
    task_type = task['TaskType']
    power_per_gpu = data_global['power_map'][task_type]

    end_hour = min(s_h + duration - 1, 2405)
    state = region_states[target_r]

    for h in range(s_h, end_hour + 1):
        state.gpu_usage[h] += gpu_demand
        state.it_power_usage[h] += gpu_demand * power_per_gpu


# ============================================================
# 8. 完整调度流程
# ============================================================

def run_scheduling(data, weights, verbose=True):
    wt = data['workload']
    regions = data['regions']

    region_states = {r: RegionState(r) for r in regions}

    priority = {'RealTimeInference': 0, 'BatchInference': 1, 'AITraining': 2}
    wt_sorted = wt.copy()
    wt_sorted['priority'] = wt_sorted['TaskType'].map(priority)
    wt_sorted = wt_sorted.sort_values(['ArrivalHour', 'priority', 'TaskID']).reset_index(drop=True)

    schedule_results = []
    migration_count = 0
    total_cost = 0.0
    total_carbon = 0.0

    start_time = time.time()

    for idx, task in wt_sorted.iterrows():
        decision, details = schedule_task(task, data, region_states, weights)
        apply_schedule(region_states, task, decision)

        if details['is_migrated']:
            migration_count += 1
        total_cost += details['cost']
        total_carbon += details['carbon']

        schedule_results.append({
            'TaskID': task['TaskID'],
            'TaskType': task['TaskType'],
            'SourceRegion': task['SourceRegion'],
            'AssignedRegion': decision[0],
            'StartHour': decision[1],
            'EndHour': min(decision[1] + int(task['Duration_h']) - 1, 2405),
            'Duration_h': task['Duration_h'],
            'GPU_Demand': task['GPU_Demand'],
            'MaxLatency_ms': task['MaxLatency_ms'],
            'IsMigrated': details['is_migrated'],
            'NetworkLatency_ms': details['latency'],
            'IncrementalCost': details['cost'],
            'IncrementalCarbon': details['carbon'],
            'RenewableUsed': details['renewable'],
        })

        if verbose and idx % 10000 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = migration_count / (idx + 1) * 100
            print(f"  [{elapsed:.0f}s] {idx}/{len(wt_sorted)} | 迁移率={rate:.1f}%", flush=True)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n调度完成! 耗时={elapsed:.1f}s, 任务={len(wt_sorted)}, "
              f"迁移={migration_count}({migration_count/len(wt_sorted)*100:.1f}%)", flush=True)
        print(f"  累计增量成本={total_cost:,.0f}元, 累计增量碳排={total_carbon:,.0f}tCO2", flush=True)

    return pd.DataFrame(schedule_results), region_states


# ============================================================
# 9. 全局指标计算
# ============================================================

def compute_global_metrics(results_df, region_states, data):
    rtd_main = data['rtd_main']
    rtd_close = data['rtd_close']
    regions = data['regions']

    total_cost = 0.0
    total_carbon = 0.0
    total_renew_avail = 0.0
    total_renew_used = 0.0
    peak_net_import = 0.0
    peak_info = ''

    hourly_records = []

    for h in range(2406):
        for r in regions:
            if h <= 2399:
                row = rtd_main.loc[(h, r)]
            else:
                if (h, r) in rtd_close.index:
                    row = rtd_close.loc[(h, r)]
                else:
                    continue

            ai_it = region_states[r].it_power_usage[h]
            nonai = row['NonAI_IT_Load_MW']
            it_load = nonai + ai_it
            pue = data['pue_map'][r]
            total_load = it_load * pue

            avail_r = row['AvailableRenewable_MW']
            used_r = min(ai_it, avail_r) if avail_r > 0 else 0

            grid_purchase = max(0, total_load - used_r)
            grid_sell = 0.0

            price = row['ElectricityPrice_CNY_per_MWh']
            ci = row['CarbonIntensity_tCO2_per_MWh']
            sell_price = row.get('SellPrice_CNY_per_MWh', 0)

            cost = grid_purchase * price - grid_sell * sell_price
            carbon = grid_purchase * ci

            total_cost += cost
            total_carbon += carbon
            total_renew_avail += avail_r
            total_renew_used += used_r

            net_import = grid_purchase - grid_sell
            if net_import > peak_net_import:
                peak_net_import = net_import
                peak_info = f"{r}@h{h}"

            hourly_records.append({
                'Hour': h, 'Region': r,
                'AI_IT': ai_it, 'NonAI_IT': nonai,
                'TotalLoad': total_load, 'GridPurchase': grid_purchase,
                'GridSell': grid_sell, 'NetImport': net_import,
                'Carbon': carbon, 'Cost': cost,
                'RenewUsed': used_r, 'RenewAvail': avail_r,
            })

    migrated = results_df[results_df['IsMigrated'] == True]
    avg_lat = migrated['NetworkLatency_ms'].mean() if len(migrated) > 0 else 0.0
    renew_util = (total_renew_used / total_renew_avail * 100) if total_renew_avail > 0 else 0.0

    metrics = {
        'total_cost': total_cost,
        'total_carbon': total_carbon,
        'avg_latency': avg_lat,
        'renewable_util': renew_util,
        'peak_net_import': peak_net_import,
        'peak_info': peak_info,
        'migration_count': len(migrated),
        'migration_rate': len(migrated) / len(results_df) * 100,
    }

    return metrics, pd.DataFrame(hourly_records)


# ============================================================
# 10. 主入口
# ============================================================

# 全局引用（供apply_schedule使用）
data_global = None

if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("问题2：碳感知任务调度模型 - 多策略对比", flush=True)
    print("=" * 60, flush=True)

    data = load_all_data()
    data_global = data

    strategies = {
        'carbon_focused': {
            'desc': '强碳感知（侧重低碳+新能源利用）',
            'weights': {'w_cost': 0.1, 'w_carbon': 1.0, 'w_latency': 0.5,
                       'w_renewable': 2.0, 'latency_factor': 0.1},
        },
        'cost_focused': {
            'desc': '强成本感知（侧重低电价）',
            'weights': {'w_cost': 1.0, 'w_carbon': 0.3, 'w_latency': 0.5,
                       'w_renewable': 0.5, 'latency_factor': 0.1},
        },
        'balanced': {
            'desc': '均衡策略（推荐）',
            'weights': {'w_cost': 0.5, 'w_carbon': 0.5, 'w_latency': 0.3,
                       'w_renewable': 1.0, 'latency_factor': 0.1},
        },
    }

    baseline_cost = 1_802_343_507
    baseline_carbon = 2_045_359

    summary_rows = []

    for sname, sinfo in strategies.items():
        print(f"\n{'='*60}", flush=True)
        print(f">>> {sinfo['desc']}", flush=True)
        print(f"{'='*60}", flush=True)

        results_df, region_states = run_scheduling(data, sinfo['weights'])
        metrics, hourly_df = compute_global_metrics(results_df, region_states, data)

        results_df.to_csv(f'/data/workspace/schedule_{sname}.csv', index=False)
        hourly_df.to_csv(f'/data/workspace/hourly_{sname}.csv', index=False)

        cost_r = (1 - metrics['total_cost'] / baseline_cost) * 100
        carbon_r = (1 - metrics['total_carbon'] / baseline_carbon) * 100

        print(f"\n--- {sname} 结果 ---", flush=True)
        print(f"  运行成本: {metrics['total_cost']:>15,.0f} 元  (降{cost_r:.1f}%)", flush=True)
        print(f"  碳排放:   {metrics['total_carbon']:>15,.0f} tCO2 (降{carbon_r:.1f}%)", flush=True)
        print(f"  平均时延: {metrics['avg_latency']:>10.1f} ms", flush=True)
        print(f"  新能源利用: {metrics['renewable_util']:>9.1f}%", flush=True)
        print(f"  峰值净购电: {metrics['peak_net_import']:>8.1f} MW @ {metrics['peak_info']}", flush=True)
        print(f"  迁移率:   {metrics['migration_rate']:>10.1f}%", flush=True)

        summary_rows.append({
            'Strategy': sname,
            'TotalCost': metrics['total_cost'],
            'TotalCarbon': metrics['total_carbon'],
            'AvgLatency_ms': metrics['avg_latency'],
            'RenewableUtil_pct': metrics['renewable_util'],
            'PeakNetImport_MW': metrics['peak_net_import'],
            'MigrationRate_pct': metrics['migration_rate'],
            'CostReduction_pct': cost_r,
            'CarbonReduction_pct': carbon_r,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('/data/workspace/strategy_comparison.csv', index=False)
    print(f"\n{'='*60}", flush=True)
    print("策略对比汇总:", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\n结果已保存到 /data/workspace/", flush=True)
