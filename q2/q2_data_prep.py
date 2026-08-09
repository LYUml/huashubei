"""
问题二：碳感知任务调度模型 —— 数据预处理与特征工程
====================================================
数据文件（只需以下文件，无需单独的碳强度/电价/新能源文件）：
  workload_trace.xlsx      : 任务基本信息
  region_time_data.xlsx    : 逐时电力参数（已包含碳强度、电价、新能源）
  network_latency.xlsx     : 区域间网络时延矩阵
  power_mapping.xlsx       : 任务类型 → 单GPU功率
  gpu_information.xlsx     : 各区域GPU容量、IT功率上限等
  storage_information.xlsx : 储能信息（可选，此处仅读取）
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "附件数据"  # 存放所有 .xlsx 的文件夹
OUTPUT_DIR = SCRIPT_DIR / "output"  # 中间结果输出文件夹

if not DATA_DIR.exists():
    raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("数据预处理开始")
print(f"数据目录: {DATA_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 60)

# ============================================================
# 1. 读取 workload_trace.xlsx
# ============================================================
print("\n" + "=" * 60)
print("Step 1: 读取 workload_trace.xlsx")
print("=" * 60)

workload_path = DATA_DIR / "workload_trace.xlsx"
if not workload_path.exists():
    raise FileNotFoundError(f"缺少文件: {workload_path}")

df = pd.read_excel(workload_path)
print(f"任务总数: {len(df)}")
print(f"字段: {list(df.columns)}")
print(f"ArrivalHour 范围: {df['ArrivalHour'].min()} ~ {df['ArrivalHour'].max()}")
print(f"区域分布:\n{df['SourceRegion'].value_counts().sort_index()}")
print(f"任务类型分布:\n{df['TaskType'].value_counts()}")

# ============================================================
# 2. 构造 GPU 需求序列 (Region × TaskType × Hour)
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 构造 GPU 需求序列")
print("=" * 60)

regions = sorted(df["SourceRegion"].unique())  # RegionA~F
task_types = sorted(df["TaskType"].unique())  # 3 种
max_hour = int(df["ArrivalHour"].max())  # 2399

pivot = df.pivot_table(
    index="ArrivalHour",
    columns=["SourceRegion", "TaskType"],
    values="GPU_Demand",
    aggfunc="sum",
    fill_value=0,
)
pivot = pivot.reindex(range(max_hour + 1), fill_value=0)
pivot.index.name = "Hour"

print(f"Pivot 形状: {pivot.shape}")
print(f"列(前 6): {list(pivot.columns)[:6]}")

pivot.to_csv(OUTPUT_DIR / "gpu_demand_pivot.csv")

# 长表（便于后续使用）
long_parts = []
for r in regions:
    for tt in task_types:
        col = (r, tt)
        if col in pivot.columns:
            sub = pivot[col].reset_index()
            sub.columns = ["Hour", "GPU_Demand"]
            sub["Region"] = r
            sub["TaskType"] = tt
            long_parts.append(sub)
long_df = pd.concat(long_parts, ignore_index=True)
long_df = long_df[["Hour", "Region", "TaskType", "GPU_Demand"]]
long_df.to_csv(OUTPUT_DIR / "gpu_demand_long.csv", index=False)
print(f"✅ GPU 需求序列已保存 (pivot={pivot.shape}, long={long_df.shape})")

# ============================================================
# 3. 读取 region_time_data.xlsx（已包含碳强度、电价、新能源）
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 读取 region_time_data.xlsx")
print("=" * 60)

rtd_path = DATA_DIR / "region_time_data.xlsx"
if not rtd_path.exists():
    raise FileNotFoundError(f"缺少文件: {rtd_path}")

rtd = pd.read_excel(rtd_path)
print(f"字段: {list(rtd.columns)}")
print(f"形状: {rtd.shape}")

# 提取需要的列（注意列名与文件一致）
needed_cols = [
    "Hour",
    "Region",
    "NonAI_IT_Load_MW",
    "CarbonIntensity_tCO2_per_MWh",
    "ElectricityPrice_CNY_per_MWh",
    "AvailableRenewable_MW",
]
missing = [c for c in needed_cols if c not in rtd.columns]
if missing:
    raise KeyError(f"region_time_data.xlsx 缺少以下列: {missing}")

rtd = rtd[needed_cols].copy()

# 重命名电价列，与后续代码兼容
rtd.rename(columns={"ElectricityPrice_CNY_per_MWh": "Price_Yuan_per_MWh"}, inplace=True)

print(f"提取后字段: {list(rtd.columns)}")
print(f"前 3 行:\n{rtd.head(3)}")

for col in [
    "NonAI_IT_Load_MW",
    "CarbonIntensity_tCO2_per_MWh",
    "Price_Yuan_per_MWh",
    "AvailableRenewable_MW",
]:
    print(f"\n  {col}:")
    print(f"    总均值 = {rtd[col].mean():.4f}")
    print(f"    按区域均值:\n{rtd.groupby('Region')[col].mean()}")

rtd.to_csv(OUTPUT_DIR / "region_time_data_merged.csv", index=False)
print("✅ 合并电力参数已保存")

# ============================================================
# 4. network_latency.xlsx → 矩阵
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 网络时延矩阵")
print("=" * 60)

latency_path = DATA_DIR / "network_latency.xlsx"
if not latency_path.exists():
    raise FileNotFoundError(f"缺少文件: {latency_path}")

nl = pd.read_excel(latency_path)
latency_matrix = nl.pivot_table(
    index="FromRegion",
    columns="ToRegion",
    values="NetworkLatency_ms",
    aggfunc="first",
)
print(f"时延矩阵:\n{latency_matrix}")
latency_matrix.to_csv(OUTPUT_DIR / "latency_matrix.csv")
print("✅ 时延矩阵已保存")

# ============================================================
# 5. power_mapping.xlsx
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 功率映射")
print("=" * 60)

power_path = DATA_DIR / "power_mapping.xlsx"
if not power_path.exists():
    raise FileNotFoundError(f"缺少文件: {power_path}")

pm = pd.read_excel(power_path)
print(pm)
power_dict = dict(zip(pm["TaskType"], pm["GPU_Power_MW_per_EquivalentGPU"]))
print(f"功率字典: {power_dict}")
pd.DataFrame([power_dict]).to_csv(OUTPUT_DIR / "power_mapping_dict.csv", index=False)

# ============================================================
# 6. gpu_information.xlsx
# ============================================================
print("\n" + "=" * 60)
print("Step 6: GPU 区域信息")
print("=" * 60)

gpu_info_path = DATA_DIR / "gpu_information.xlsx"
if not gpu_info_path.exists():
    raise FileNotFoundError(f"缺少文件: {gpu_info_path}")

gi = pd.read_excel(gpu_info_path)
print(gi)
region_info = gi.set_index("Region").to_dict("index")
print(f"region_info 示例 (RegionA): {region_info.get('RegionA')}")
gi.to_csv(OUTPUT_DIR / "gpu_information.csv", index=False)

# ============================================================
# 7. storage_information.xlsx（可选）
# ============================================================
print("\n" + "=" * 60)
print("Step 7: 储能信息")
print("=" * 60)

storage_path = DATA_DIR / "storage_information.xlsx"
if not storage_path.exists():
    print("⚠️  storage_information.xlsx 不存在，跳过")
    storage_info = {}
else:
    st = pd.read_excel(storage_path)
    print(st)
    storage_info = st.set_index("Region").to_dict("index")
    st.to_csv(OUTPUT_DIR / "storage_information.csv", index=False)

# ============================================================
# 8. 汇总保存为 pickle
# ============================================================
print("\n" + "=" * 60)
print("Step 8: 汇总保存为 pickle")
print("=" * 60)

preprocessed = {
    "workload": df,
    "gpu_pivot": pivot,
    "gpu_long": long_df,
    "region_time": rtd,
    "latency_matrix": latency_matrix,
    "power_dict": power_dict,
    "region_info": region_info,
    "storage_info": storage_info,
    "regions": regions,
    "task_types": task_types,
    "max_hour": max_hour,
}

pkl_path = OUTPUT_DIR / "preprocessed_data.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(preprocessed, f)

print(f"\n✅ 全部预处理完成")
print(f"  任务数:    {len(df)}")
print(f"  时间范围:  0 ~ {max_hour}h")
print(f"  区域:      {regions}")
print(f"  任务类型:  {task_types}")
print(f"  序列形状:  {pivot.shape}")
print(f"  电力参数列: {list(rtd.columns)}")
print(f"  汇总文件:  {pkl_path}")
print(f"\n所有中间结果已保存至: {OUTPUT_DIR}")
