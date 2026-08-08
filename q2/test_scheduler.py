"""最小化测试"""
import pandas as pd
import numpy as np
import time

print("加载数据...")
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

power_map = dict(zip(pm['TaskType'], pm['GPU_Power_MW_per_EquivalentGPU']))

latency_dict = {}
for _, row in nl.iterrows():
    latency_dict[(row['FromRegion'], row['ToRegion'])] = row['NetworkLatency_ms']

regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']

print(f"数据加载完成. 任务数={len(wt)}")
print(f"power_map: {power_map}")
print(f"pue_map: {pue_map}")

# Test latency
print(f"\nRegionA->RegionD latency: {latency_dict.get(('RegionA','RegionD'),'N/A')}")
print(f"RegionD->RegionE latency: {latency_dict.get(('RegionD','RegionE'),'N/A')}")

# Test a few tasks
print("\n=== 前5个任务信息 ===")
for i in range(5):
    t = wt.iloc[i]
    print(f"  Task{t['TaskID']}: type={t['TaskType']}, src={t['SourceRegion']}, "
          f"arrival={t['ArrivalHour']}, GPU={t['GPU_Demand']}, "
          f"dur_h={t['Duration_h']}, maxlat={t['MaxLatency_ms']}")
