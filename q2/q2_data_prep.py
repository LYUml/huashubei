"""
问题二：碳感知任务调度模型 —— 数据预处理与特征工程
====================================================
数据文件 (gen_data.py 生成):
  workload_trace.xlsx      : TaskID, TaskType, ArrivalHour, GPU_Demand,
                            EstimatedDuration_min, DelaySensitivity,
                            SourceRegion, MaxLatency_ms, LatestFinishHour,
                            EarliestStartHour, ExecutionMode
  region_time_data.xlsx   : Region, Hour, NonAI_IT_Load_MW,
                            Baseline_AI_IT_Load_MW, IT_Load_MW, Facility_Load_MW
  network_latency.xlsx    : SourceRegion, TargetRegion, Latency_ms
  power_mapping.xlsx      : TaskType, PowerPerGPU_MW
  gpu_information.xlsx    : Region, Available_GPU, Max_IT_Power_MW,
                            Max_Facility_Power_MW, PUE
  carbon_intensity.xlsx   : Region, Hour, CarbonIntensity_tCO2_per_MWh
  electricity_price.xlsx  : Region, Hour, Price_Yuan_per_MWh
  renewable_generation.xlsx: Region, Hour, AvailableRenewable_MW
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path("/data/workspace/data")

# ============================================================
# 1. workload_trace
# ============================================================
print("=" * 60)
print("Step 1: 读取 workload_trace.xlsx")
print("=" * 60)

df = pd.read_excel(DATA_DIR / "workload_trace.xlsx")
print(f"任务总数: {len(df)}")
print(f"字段: {list(df.columns)}")
print(f"ArrivalHour 范围: {df['ArrivalHour'].min()} ~ {df['ArrivalHour'].max()}")
print(f"\n区域分布:\n{df['SourceRegion'].value_counts().sort_index()}")
print(f"\n任务类型分布:\n{df['TaskType'].value_counts()}")
print(f"\nDelaySensitivity 分布:\n{df['DelaySensitivity'].value_counts()}")

# ============================================================
# 2. GPU 需求序列 (Region × TaskType × Hour)
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 构造 GPU 需求序列")
print("=" * 60)

regions    = sorted(df['SourceRegion'].unique())   # RegionA~F
task_types = sorted(df['TaskType'].unique())      # 3 种
max_hour  = int(df['ArrivalHour'].max())        # 2399

pivot = df.pivot_table(
    index='ArrivalHour',
    columns=['SourceRegion', 'TaskType'],
    values='GPU_Demand',
    aggfunc='sum',
    fill_value=0,
)
pivot = pivot.reindex(range(max_hour + 1), fill_value=0)
pivot.index.name = 'Hour'

print(f"Pivot 形状: {pivot.shape}")
print(f"列(前 6): {list(pivot.columns)[:6]}")
print(f"\n各序列总 GPU 需求 TOP10:")
print(pivot.sum().sort_values(ascending=False).head(10))

pivot.to_csv("/data/workspace/gpu_demand_pivot.csv")

# 长表 (绕开 MultiIndex 列名问题)
long_parts = []
for r in regions:
    for tt in task_types:
        col = (r, tt)
        if col in pivot.columns:
            sub = pivot[col].reset_index()
            sub.columns = ['Hour', 'GPU_Demand']
            sub['Region']   = r
            sub['TaskType'] = tt
            long_parts.append(sub)
long_df = pd.concat(long_parts, ignore_index=True)
long_df = long_df[['Hour', 'Region', 'TaskType', 'GPU_Demand']]
long_df.to_csv("/data/workspace/gpu_demand_long.csv", index=False)
print(f"\n✅ GPU 需求序列已保存 (pivot={pivot.shape}, long={long_df.shape})")

# ============================================================
# 3. 合并逐时电力参数
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 读取并合并逐时电力参数")
print("=" * 60)

rtd = pd.read_excel(DATA_DIR / "region_time_data.xlsx")
ci  = pd.read_excel(DATA_DIR / "carbon_intensity.xlsx")
ep  = pd.read_excel(DATA_DIR / "electricity_price.xlsx")
rnw = pd.read_excel(DATA_DIR / "renewable_generation.xlsx")

print(f"region_time_data: {list(rtd.columns)} | {rtd.shape}")
print(f"carbon_intensity : {list(ci.columns)}  | {ci.shape}")
print(f"electricity_price: {list(ep.columns)}  | {ep.shape}")
print(f"renewable_gen    : {list(rnw.columns)} | {rnw.shape}")

# 全部以 (Region, Hour) 为主键合并
rtd = rtd.merge(ci,  on=['Region', 'Hour'])
rtd = rtd.merge(ep,  on=['Region', 'Hour'])
rtd = rtd.merge(rnw, on=['Region', 'Hour'])

print(f"\n合并后字段: {list(rtd.columns)}")
print(f"前 3 行:\n{rtd.head(3)}")

for col in ['NonAI_IT_Load_MW', 'CarbonIntensity_tCO2_per_MWh',
            'Price_Yuan_per_MWh', 'AvailableRenewable_MW']:
    if col in rtd.columns:
        print(f"\n  {col}:")
        print(f"    总均值 = {rtd[col].mean():.4f}")
        print(f"    按区域均值:\n{rtd.groupby('Region')[col].mean()}")

rtd.to_csv("/data/workspace/region_time_data_merged.csv", index=False)
print("\n✅ 合并电力参数已保存")

# ============================================================
# 4. network_latency -> 矩阵
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 网络时延矩阵")
print("=" * 60)

nl = pd.read_excel(DATA_DIR / "network_latency.xlsx")
latency_matrix = nl.pivot_table(
    index='SourceRegion',
    columns='TargetRegion',
    values='Latency_ms',
    aggfunc='first',
)
print(f"时延矩阵:\n{latency_matrix}")
latency_matrix.to_csv("/data/workspace/latency_matrix.csv")
print("✅ 时延矩阵已保存")

# ============================================================
# 5. power_mapping
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 功率映射")
print("=" * 60)

pm = pd.read_excel(DATA_DIR / "power_mapping.xlsx")
print(pm)
power_dict = dict(zip(pm['TaskType'], pm['PowerPerGPU_MW']))
print(f"\n功率字典: {power_dict}")
pd.DataFrame([power_dict]).to_csv("/data/workspace/power_mapping_dict.csv", index=False)

# ============================================================
# 6. gpu_information
# ============================================================
print("\n" + "=" * 60)
print("Step 6: GPU 区域信息")
print("=" * 60)

gi = pd.read_excel(DATA_DIR / "gpu_information.xlsx")
print(gi)
region_info = gi.set_index('Region').to_dict('index')
print(f"\nregion_info 示例 (RegionA): {region_info.get('RegionA')}")
gi.to_csv("/data/workspace/gpu_information.csv", index=False)

# ============================================================
# 7. storage_information
# ============================================================
print("\n" + "=" * 60)
print("Step 7: 储能信息")
print("=" * 60)

st = pd.read_excel(DATA_DIR / "storage_information.xlsx")
print(st)
storage_info = st.set_index('Region').to_dict('index')
st.to_csv("/data/workspace/storage_information.csv", index=False)

# ============================================================
# 8. 汇总
# ============================================================
print("\n" + "=" * 60)
print("Step 8: 汇总保存")
print("=" * 60)

preprocessed = {
    'workload':       df,
    'gpu_pivot':     pivot,
    'gpu_long':      long_df,
    'region_time':   rtd,
    'latency_matrix': latency_matrix,
    'power_dict':    power_dict,
    'region_info':   region_info,
    'storage_info':  storage_info,
    'regions':       regions,
    'task_types':    task_types,
    'max_hour':     max_hour,
}

with open("/data/workspace/preprocessed_data.pkl", 'wb') as f:
    pickle.dump(preprocessed, f)

print(f"\n✅ 全部预处理完成")
print(f"  任务数:    {len(df)}")
print(f"  时间范围:  0 ~ {max_hour}h")
print(f"  区域:      {regions}")
print(f"  任务类型:  {task_types}")
print(f"  序列形状:  {pivot.shape}")
print(f"  电力参数列: {list(rtd.columns)}")
