import pandas as pd
import numpy as np

# Load workload trace
df = pd.read_excel("/data/workspace/data/workload_trace.xlsx")
print("=== workload_trace ===")
print(f"Shape: {df.shape}")
print(df.head(10))
print()
print("TaskType counts:")
print(df['TaskType'].value_counts())
print()
print("SourceRegion counts:")
print(df['SourceRegion'].value_counts())
print()
print("ArrivalHour range:", df['ArrivalHour'].min(), "-", df['ArrivalHour'].max())
print()

# Check region_time_data
rtd = pd.read_excel("/data/workspace/data/region_time_data.xlsx")
print("=== region_time_data ===")
print(f"Shape: {rtd.shape}")
print(rtd.head(10))
print()

# Check gpu info
gpu = pd.read_excel("/data/workspace/data/gpu_information.xlsx")
print("=== gpu_information ===")
print(gpu)
print()

# Check storage
st = pd.read_excel("/data/workspace/data/storage_information.xlsx")
print("=== storage_information ===")
print(st)
print()

# Check carbon
ci = pd.read_excel("/data/workspace/data/carbon_intensity.xlsx")
print("=== carbon_intensity ===")
print(ci.head(10))
print()

# Check prices
ep = pd.read_excel("/data/workspace/data/electricity_price.xlsx")
print("=== electricity_price ===")
print(ep.head(10))
print()

# Check renewable
rg = pd.read_excel("/data/workspace/data/renewable_generation.xlsx")
print("=== renewable_generation ===")
print(rg.head(10))
print()

# Check network latency
nl = pd.read_excel("/data/workspace/data/network_latency.xlsx")
print("=== network_latency ===")
print(nl.head(10))
print()

# Check power mapping
pm = pd.read_excel("/data/workspace/data/power_mapping.xlsx")
print("=== power_mapping ===")
print(pm)
