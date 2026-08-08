"""
========================================================
问题2：碳感知任务调度模型 - 核心求解器
========================================================
策略：加权多目标贪心调度
  - 对每个任务，评估所有可行(区域, 开工时间)组合
  - 用加权评分函数选择最优组合
  - 核心：将任务尽量迁移到 低电价+低碳+高新能源 的区域
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time

# ============================================================
# 1. 数据加载与预处理
# ============================================================

def load_all_data():
    """加载所有输入数据，返回结构化字典"""
    wt = pd.read_excel('/data/inputs/workload_trace.xlsx')
    rtd = pd.read_excel('/data/inputs/region_time_data.xlsx')
    gpu_info = pd.read_excel('/data/inputs/GPU_information.xlsx')
    nl = pd.read_excel('/data/inputs/network_latency.xlsx')
    pm = pd.read_excel('/data/inputs/power_mapping.xlsx')
    si = pd.read_excel('/data/inputs/storage_information.xlsx')

    # 任务持续时间（小时，向上取整）
    wt['Duration_h'] = np.ceil(wt['EstimatedDuration_min'] / 60).astype(int)

    # region_time_data 主时域 0-2399
    rtd_main = rtd[rtd['DataPeriod'] == 'Main_0_2399'].copy()
    rtd_main = rtd_main.set_index(['Hour', 'Region'])

    # 收尾时域 2400-2406
    rtd_close = rtd[rtd['DataPeriod'] == 'Closure_2400_2406'].copy()
    rtd_close = rtd_close.set_index(['Hour', 'Region'])

    # 映射字典
    pue_map = dict(zip(gpu_info['Region'], gpu_info['PUE']))
    avail_gpu_map = dict(zip(gpu_info['Region'], gpu_info['Available_GPU']))
    max_it_power_map = dict(zip(gpu_info['Region'], gpu_info['Max_IT_Power_MW']))
    max_fac_power_map = dict(zip(gpu_info['Region'], gpu_info['Max_Facility_Power_MW']))
    max_import_map = dict(zip(si['Region'], si['MaxGridImport_MW']))
    max_export_map = dict(zip(si['Region'], si['MaxGridExport_MW']))

    power_map = dict(zip(pm['TaskType'], pm['GPU_Power_MW_per_EquivalentGPU']))

    # 时延矩阵
    latency_dict = {}
    for _, row in nl.iterrows():
        latency_dict[(row['FromRegion'], row['ToRegion'])] = row['NetworkLatency_ms']

    # 区域列表
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
# 2. 可行区域筛选
# ============================================================

def get_feasible_regions(task, latency_dict, regions):
    """
    根据网络时延约束筛选可行区域
    RealTimeInference: MaxLatency=20ms -> 只能到时延<=20的区域
    BatchInference: MaxLatency=80ms
    AITraining: MaxLatency=150ms
    """
    src = task['SourceRegion']
    max_lat = task['MaxLatency_ms']
    feasible = []
    for r in regions:
        if latency_dict.get((src, r), 999) <= max_lat:
            feasible.append(r)
    return feasible


# ============================================================
# 3. 多目标评分函数
# ============================================================

def compute_cost_carbon_for_task(task, target_region, start_hour, data):
    """
    计算将任务分配到 (target_region, start_hour) 的增量成本和碳排放
    返回: (增量成本元, 增量碳排放tCO2, 网络时延ms, 新能源利用量MW)
    """
    rtd_main = data['rtd_main']
    rtd_close = data['rtd_close']
    duration = task['Duration_h']
    gpu_demand = task['GPU_Demand']
    task_type = task['TaskType']
    power_per_gpu = data['power_map'][task_type]
    pue = data['pue_map'][target_region]

    # 任务运行的小时区间
    end_hour = min(start_hour + duration - 1, 2405)
    hours = list(range(start_hour, end_hour + 1))

    incremental_cost = 0.0
    incremental_carbon = 0.0
    incremental_renewable_use = 0.0

    for h in hours:
        if h <= 2399:
            row = rtd_main.loc[(h, target_region)]
        else:
            if (h, target_region) in rtd_close.index:
                row = rtd_close.loc[(h, target_region)]
            else:
                continue

        # 该小时该任务的AI IT功率增量
        ai_it_power = gpu_demand * power_per_gpu  # MW

        # 增量设施负荷 = ai_it_power * PUE
        incremental_load = ai_it_power * pue  # MW

        carbon_intensity = row['CarbonIntensity_tCO2_per_MWh']
        elec_price = row['ElectricityPrice_CNY_per_MWh']

        # 增量购电（假设增量负荷由电网满足，简化）
        incremental_cost += incremental_load * elec_price
        incremental_carbon += incremental_load * carbon_intensity

        # 新能源利用：如果新能源有富余，增量负荷可以用新能源
        avail_renewable = row['AvailableRenewable_MW']
        if avail_renewable > 0:
            renewable_used = min(ai_it_power, avail_renewable)
            incremental_renewable_use += renewable_used

    # 网络时延
    src = task['SourceRegion']
    latency = data['latency_dict'].get((src, target_region), 999)

    return incremental_cost, incremental_carbon, latency, incremental_renewable_use


# ============================================================
# 4. 区域状态跟踪
# ============================================================

@dataclass
class RegionState:
    """跟踪每个区域的实时状态"""
    region: str
    # gpu_usage[h] = 该区域第h小时的GPU占用数
    gpu_usage: np.ndarray = field(default_factory=lambda: np.zeros(2406, dtype=np.float64))
    # it_power_usage[h] = 该区域第h小时的AI IT功率(MW)
    it_power_usage: np.ndarray = field(default_factory=lambda: np.zeros(2406, dtype=np.float64))


# ============================================================
# 5. 容量约束检查
# ============================================================

def check_capacity(region_state, task, target_region, start_hour, data):
    """
    检查将任务放在 (target_region, start_hour) 是否满足所有容量约束
    """
    duration = task['Duration_h']
    gpu_demand = task['GPU_Demand']
    task_type = task['TaskType']
    power_per_gpu = data['power_map'][task_type]
    pue = data['pue_map'][target_region]
    avail_gpu = data['avail_gpu_map'][target_region]
    max_it = data['max_it_power_map'][target_region]
    max_fac = data['max_fac_power_map'][target_region]

    end_hour = min(start_hour + duration - 1, 2405)

    for h in range(start_hour, end_hour + 1):
        new_gpu = region_state.gpu_usage[h] + gpu_demand
        if new_gpu > avail_gpu:
            return False

        new_it_power = region_state.it_power_usage[h] + gpu_demand * power_per_gpu
        # 加上NonAI负荷检查IT功率上限
        if h <= 2399:
            nonai = data['rtd_main'].loc[(h, target_region)]['NonAI_IT_Load_MW']
        else:
            if (h, target_region) in data['rtd_close'].index:
                nonai = data['rtd_close'].loc[(h, target_region)]['NonAI_IT_Load_MW']
            else:
                nonai = 0

        if new_it_power + nonai > max_it:
            return False

        new_fac_power = (new_it_power + nonai) * pue
        if new_fac_power > max_fac:
            return False

    return True


# ============================================================
# 6. 单个任务调度
# ============================================================

def schedule_task(task, data, region_states, weights, data_ref=None):
    """
    为单个任务选择最优的 (区域, 开工时间) 组合
    使用加权多目标评分
    """
    if data_ref is None:
        data_ref = data

    feasible_regions = get_feasible_regions(task, data['latency_dict'], data['regions'])

    best_score = float('inf')
    best_decision = None
    best_details = None

    src = task['SourceRegion']
    arrival = int(task['ArrivalHour'])
    latest_finish = int(task['LatestFinishHour'])
    duration = int(task['Duration_h'])
    task_type = task['TaskType']

    # 确定开工时间搜索范围
    if task_type == 'RealTimeInference':
        # 到达即开工
        start_candidates = [arrival]
    else:
        # 弹性任务：可以在 [arrival, latest_finish - duration + 1] 范围内选择
        earliest = int(task.get('EarliestStartHour', arrival))
        latest_start = min(latest_finish - duration + 1, 2405)
        # 限制搜索范围：最多往后看300小时
        latest_start = min(latest_start, arrival + 300)
        start_candidates = list(range(earliest, latest_start + 1))

    for target_r in feasible_regions:
        for s_h in start_candidates:
            s_h = int(s_h)
            # 检查时间合法性
            if s_h + duration - 1 > 2405:
                continue

            # 检查容量约束
            if not check_capacity(region_states[target_r], task, target_r, s_h, data):
                continue

            # 计算多目标
            cost, carbon, latency, renew = compute_cost_carbon_for_task(
                task, target_r, s_h, data
            )

            # 时延惩罚
            if target_r != src:
                latency_penalty = latency * weights.get('latency_factor', 1.0)
            else:
                latency_penalty = 0

            # 归一化成本（除以1000）
            cost_norm = cost / 1000.0

            # 综合评分（越低越好）
            score = (
                weights['w_cost'] * cost_norm +
                weights['w_carbon'] * carbon +
                weights['w_latency'] * latency_penalty +
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
        # 兜底：强制放在本地
        target_r = src
        s_h = arrival
        best_decision = (target_r, s_h)
        best_details = {'cost': 0, 'carbon': 0, 'latency': 0, 'renewable': 0, 'is_migrated': False}

    return best_decision, best_details


# ============================================================
# 7. 应用调度结果到区域状态
# ============================================================

def apply_schedule(region_states, task, decision, data_ref):
    """将调度决策应用到区域状态"""
    target_r, s_h = decision
    duration = int(task['Duration_h'])
    gpu_demand = task['GPU_Demand']
    task_type = task['TaskType']
    power_per_gpu = data_ref['power_map'][task_type]

    end_hour = min(s_h + duration - 1, 2405)
    state = region_states[target_r]

    for h in range(s_h, end_hour + 1):
        state.gpu_usage[h] += gpu_demand
        state.it_power_usage[h] += gpu_demand * power_per_gpu


# ============================================================
# 8. 完整调度流程
# ============================================================

def run_scheduling(data, weights, verbose=True):
    """
    运行完整调度流程
    """
    wt = data['workload']
    regions = data['regions']

    # 初始化区域状态
    region_states = {r: RegionState(region=r) for r in regions}

    # 按到达时间排序，同时间按优先级排序
    priority = {'RealTimeInference': 0, 'BatchInference': 1, 'AITraining': 2}
    wt_sorted = wt.copy()
    wt_sorted['priority'] = wt_sorted['TaskType'].map(priority)
    wt_sorted = wt_sorted.sort_values(['ArrivalHour', 'priority', 'TaskID']).reset_index(drop=True)

    # 结果存储
    schedule_results = []
    migration_count = 0
    total_cost = 0
    total_carbon = 0

    start_time = time.time()

    for idx, task in wt_sorted.iterrows():
        decision, details = schedule_task(task, data, region_states, weights)
        target_r, s_h = decision

        # 应用调度
        apply_schedule(region_states, task, decision, data)

        # 统计
        if details['is_migrated']:
            migration_count += 1
        total_cost += details['cost']
        total_carbon += details['carbon']

        schedule_results.append({
            'TaskID': task['TaskID'],
            'TaskType': task['TaskType'],
            'SourceRegion': task['SourceRegion'],
            'AssignedRegion': target_r,
            'StartHour': s_h,
            'EndHour': min(s_h + int(task['Duration_h']) - 1, 2405),
            'Duration_h': task['Duration_h'],
            'GPU_Demand': task['GPU_Demand'],
            'MaxLatency_ms': task['MaxLatency_ms'],
            'IsMigrated': details['is_migrated'],
            'NetworkLatency_ms': details['latency'],
            'IncrementalCost': details['cost'],
            'IncrementalCarbon': details['carbon'],
            'RenewableUsed': details['renewable'],
        })

        if verbose and idx % 5000 == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(f"  已处理 {idx}/{len(wt_sorted)} 任务, 耗时 {elapsed:.1f}s, "
                  f"迁移率={migration_count/(idx+1)*100:.1f}%")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n调度完成! 总耗时: {elapsed:.1f}s")
        print(f"  总任务数: {len(wt_sorted)}")
        print(f"  迁移任务数: {migration_count} ({migration_count/len(wt_sorted)*100:.1f}%)")
        print(f"  总增量成本: {total_cost:.0f} 元")
        print(f"  总增量碳排放: {total_carbon:.0f} tCO2")

    results_df = pd.DataFrame(schedule_results)
    return results_df, region_states


# ============================================================
# 9. 全局指标计算
# ============================================================

def compute_global_metrics(results_df, region_states, data):
    """
    计算全局指标
    """
    rtd_main = data['rtd_main']
    rtd_close = data['rtd_close']
    regions = data['regions']

    # 计算各区域逐时AI IT负荷
    ai_it_load = {}
    for r in regions:
        ai_it_load[r] = region_states[r].it_power_usage.copy()

    # 逐时逐区域计算
    total_cost = 0.0
    total_carbon = 0.0
    total_renewable_available = 0.0
    total_renewable_used = 0.0
    total_net_grid_import = 0.0
    peak_net_import = 0.0
    peak_import_info = ''

    # 存储逐时数据用于绘图
    hourly_data = []

    for h in range(2406):
        for r in regions:
            if h <= 2399:
                row = rtd_main.loc[(h, r)]
            else:
                if (h, r) in rtd_close.index:
                    row = rtd_close.loc[(h, r)]
                else:
                    continue

            nonai = row['NonAI_IT_Load_MW']
            ai_it = ai_it_load[r][h]
            it_load = nonai + ai_it
            pue = data['pue_map'][r]
            total_load = it_load * pue

            avail_renew = row['AvailableRenewable_MW']
            # 新能源优先供给AI负荷
            used_renew = min(ai_it, avail_renew)

            # 购电 = 设施总负荷 - 新能源直接消纳（简化）
            grid_purchase = max(0, total_load - used_renew)
            grid_sell = 0  # 问题2不考虑外送（部分区域MaxGridExport=0）

            elec_price = row['ElectricityPrice_CNY_per_MWh']
            carbon_intensity = row['CarbonIntensity_tCO2_per_MWh']
            sell_price = row.get('SellPrice_CNY_per_MWh', 0)

            cost = grid_purchase * elec_price - grid_sell * sell_price
            carbon = grid_purchase * carbon_intensity

            total_cost += cost
            total_carbon += carbon
            total_renewable_available += avail_renew
            total_renewable_used += used_renew

            net_import = grid_purchase - grid_sell
            total_net_grid_import += net_import
            if net_import > peak_net_import:
                peak_net_import = net_import
                peak_import_info = f"{r}@hour{h}"

            hourly_data.append({
                'Hour': h,
                'Region': r,
                'AI_IT_Load': ai_it,
                'NonAI_IT_Load': nonai,
                'Total_Load': total_load,
                'GridPurchase': grid_purchase,
                'GridSell': grid_sell,
                'NetGridImport': net_import,
                'CarbonEmission': carbon,
                'Cost': cost,
                'RenewableUsed': used_renew,
                'AvailableRenewable': avail_renew,
            })

    # 网络时延
    migrated_tasks = results_df[results_df['IsMigrated'] == True]
    avg_latency = migrated_tasks['NetworkLatency_ms'].mean() if len(migrated_tasks) > 0 else 0

    # 新能源利用率
    renewable_utilization = (total_renewable_used / total_renewable_available * 100) \
        if total_renewable_available > 0 else 0

    metrics = {
        'total_cost_CNY': total_cost,
        'total_carbon_tCO2': total_carbon,
        'avg_network_latency_ms': avg_latency,
        'renewable_utilization_pct': renewable_utilization,
        'peak_net_grid_import_MW': peak_net_import,
        'peak_import_info': peak_import_info,
        'migration_count': len(migrated_tasks),
        'migration_rate_pct': len(migrated_tasks) / len(results_df) * 100,
    }

    hourly_df = pd.DataFrame(hourly_data)
    return metrics, hourly_df


# ============================================================
# 10. 主入口 - 多策略对比
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("问题2：碳感知任务调度模型")
    print("=" * 60)

    # 加载数据
    data = load_all_data()

    # 定义三种策略
    strategies = {
        'carbon_focused': {
            'desc': '强碳感知策略（侧重低碳+新能源）',
            'weights': {
                'w_cost': 0.1,
                'w_carbon': 1.0,
                'w_latency': 0.5,
                'w_renewable': 2.0,
                'latency_factor': 0.1,
            }
        },
        'cost_focused': {
            'desc': '强成本感知策略（侧重低成本）',
            'weights': {
                'w_cost': 1.0,
                'w_carbon': 0.3,
                'w_latency': 0.5,
                'w_renewable': 0.5,
                'latency_factor': 0.1,
            }
        },
        'balanced': {
            'desc': '均衡策略（推荐默认）',
            'weights': {
                'w_cost': 0.5,
                'w_carbon': 0.5,
                'w_latency': 0.3,
                'w_renewable': 1.0,
                'latency_factor': 0.1,
            }
        }
    }

    all_results = {}
    baseline_cost = 1802343507
    baseline_carbon = 2045359

    for strat_name, strat_info in strategies.items():
        print(f"\n{'='*60}")
        print(f">>> 运行策略: {strat_info['desc']}")
        print(f"{'='*60}")

        results_df, region_states = run_scheduling(data, strat_info['weights'])
        metrics, hourly_df = compute_global_metrics(results_df, region_states, data)

        # 保存
        results_df.to_csv(f'/data/workspace/schedule_{strat_name}.csv', index=False)
        hourly_df.to_csv(f'/data/workspace/hourly_{strat_name}.csv', index=False)

        # 打印指标
        print(f"\n--- {strat_name} 策略指标 ---")
        print(f"  总运行成本: {metrics['total_cost_CNY']:,.0f} 元")
        print(f"  总碳排放:   {metrics['total_carbon_tCO2']:,.0f} tCO2")
        print(f"  平均网络时延: {metrics['avg_network_latency_ms']:.1f} ms")
        print(f"  新能源利用率: {metrics['renewable_utilization_pct']:.1f}%")
        print(f"  峰值净购电: {metrics['peak_net_grid_import_MW']:.1f} MW")
        print(f"  迁移率:     {metrics['migration_rate_pct']:.1f}%")

        cost_red = (1 - metrics['total_cost_CNY'] / baseline_cost) * 100
        carbon_red = (1 - metrics['total_carbon_tCO2'] / baseline_carbon) * 100
        print(f"  成本降低: {cost_red:.1f}% (vs 基准)")
        print(f"  碳排降低: {carbon_red:.1f}% (vs 基准)")

        all_results[strat_name] = {
            'results': results_df,
            'hourly': hourly_df,
            'metrics': metrics,
        }

    # 保存汇总对比
    summary = []
    for name, info in all_results.items():
        m = info['metrics']
        summary.append({
            'Strategy': name,
            'TotalCost_CNY': m['total_cost_CNY'],
            'TotalCarbon_tCO2': m['total_carbon_tCO2'],
            'AvgLatency_ms': m['avg_network_latency_ms'],
            'RenewableUtil_pct': m['renewable_utilization_pct'],
            'PeakNetImport_MW': m['peak_net_grid_import_MW'],
            'MigrationRate_pct': m['migration_rate_pct'],
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('/data/workspace/strategy_comparison.csv', index=False)
    print(f"\n策略对比表已保存到 strategy_comparison.csv")
    print(summary_df.to_string(index=False))
