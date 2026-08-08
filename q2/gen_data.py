import numpy as np
import pandas as pd
import os
import itertools

np.random.seed(42)

OUT = "/data/workspace/data"
os.makedirs(OUT, exist_ok=True)

# ── 1. workload_trace ──────────────────────────────────────────────
N = 1200
TASK_TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
REGIONS = [f"Region{r}" for r in "ABCDEF"]
SENSITIVITY = {"RealTimeInference": "High", "BatchInference": "Medium", "AITraining": "Low"}
MAX_LATENCY = {"RealTimeInference": 20, "BatchInference": 80, "AITraining": 150}

def gen_tasks():
    rows = []
    for i in range(1, N + 1):
        t = np.random.choice(TASK_TYPES, p=[0.35, 0.35, 0.30])
        src = np.random.choice(REGIONS)
        arr = int(np.random.randint(0, 2400))
        if t == "RealTimeInference":
            dur = int(np.random.randint(1, 8))
            gpu = int(np.random.randint(50, 400))
        elif t == "BatchInference":
            dur = int(np.random.randint(4, 25))
            gpu = int(np.random.randint(30, 400))
        else:
            dur = int(np.random.randint(16, 130))
            gpu = int(np.random.randint(10, 400))
        rows.append([i, t, arr, gpu, dur, SENSITIVITY[t], src, MAX_LATENCY[t], 2406, arr, "NonPreemptive"])
    return rows

cols = ["TaskID", "TaskType", "ArrivalHour", "GPU_Demand", "EstimatedDuration_min",
        "DelaySensitivity", "SourceRegion", "MaxLatency_ms",
        "LatestFinishHour", "EarliestStartHour", "ExecutionMode"]

pd.DataFrame(gen_tasks(), columns=cols).to_excel(f"{OUT}/workload_trace.xlsx", index=False)
print("workload_trace.xlsx done")

# ── 2. region_time_data ────────────────────────────────────────────
def smooth(base, amp, n):
    t = np.arange(n)
    return base + amp * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, amp * 0.15, n)

records = []
for r in REGIONS:
    nonai = smooth(5, 2, 2406)
    ai = smooth(8, 4, 2406)
    it = nonai + ai
    fac = it * 1.25
    for h in range(2406):
        records.append([r, h, round(nonai[h], 3), round(ai[h], 3),
                       round(it[h], 3), round(fac[h], 3)])

pdf = pd.DataFrame(records, columns=["Region", "Hour", "NonAI_IT_Load_MW",
                                      "Baseline_AI_IT_Load_MW", "IT_Load_MW",
                                      "Facility_Load_MW"])
pdf.to_excel(f"{OUT}/region_time_data.xlsx", index=False)
print("region_time_data.xlsx done")

# ── 3. network_latency ────────────────────────────────────────────
lat = []
for a, b in itertools.product(REGIONS, repeat=2):
    v = 0 if a == b else int(np.random.randint(5, 60))
    lat.append([a, b, v])
pd.DataFrame(lat, columns=["SourceRegion", "TargetRegion", "Latency_ms"]).to_excel(
    f"{OUT}/network_latency.xlsx", index=False)
print("network_latency.xlsx done")

# ── 4. power_mapping ───────────────────────────────────────────────
pm = [["RealTimeInference", 0.002], ["BatchInference", 0.003], ["AITraining", 0.005]]
pd.DataFrame(pm, columns=["TaskType", "PowerPerGPU_MW"]).to_excel(
    f"{OUT}/power_mapping.xlsx", index=False)
print("power_mapping.xlsx done")

# ── 5. gpu_information ──────────────────────────────────────────────
gpu = [
    ["RegionA", 800, 25, 32, 1.28],
    ["RegionB", 600, 20, 26, 1.30],
    ["RegionC", 700, 22, 28, 1.27],
    ["RegionD", 900, 28, 36, 1.29],
    ["RegionE", 500, 16, 20, 1.25],
    ["RegionF", 750, 24, 30, 1.26],
]
pd.DataFrame(gpu, columns=["Region", "Available_GPU", "Max_IT_Power_MW",
                            "Max_Facility_Power_MW", "PUE"]).to_excel(
    f"{OUT}/gpu_information.xlsx", index=False)
print("gpu_information.xlsx done")

# ── 6. storage_information ──────────────────────────────────────────
st = [
    ["RegionA", 50, 10, 10, 20, 5, 0.92, 0.92],
    ["RegionB", 40, 8, 8, 16, 4, 0.92, 0.92],
    ["RegionC", 45, 9, 9, 18, 5, 0.92, 0.92],
    ["RegionD", 60, 12, 12, 24, 6, 0.92, 0.92],
    ["RegionE", 30, 6, 6, 12, 3, 0.92, 0.92],
    ["RegionF", 48, 10, 10, 19, 5, 0.92, 0.92],
]
pd.DataFrame(st, columns=["Region", "StorageCapacity_MWh", "MaxChargePower_MW",
                           "MaxDischargePower_MW", "InitialSOC_MWh",
                           "MinSOC_MWh", "ChargeEfficiency",
                           "DischargeEfficiency"]).to_excel(
    f"{OUT}/storage_information.xlsx", index=False)
print("storage_information.xlsx done")

# ── 7. carbon_intensity ─────────────────────────────────────────────
ci = []
bases_c = {"RegionA": 0.45, "RegionB": 0.55, "RegionC": 0.50,
           "RegionD": 0.40, "RegionE": 0.35, "RegionF": 0.60}
for r in REGIONS:
    base = bases_c[r]
    for h in range(2406):
        v = base + 0.15 * np.sin(h / 24 * 2 * np.pi) + np.random.normal(0, 0.03)
        ci.append([r, h, round(max(v, 0.05), 4)])
pd.DataFrame(ci, columns=["Region", "Hour", "CarbonIntensity_tCO2_per_MWh"]).to_excel(
    f"{OUT}/carbon_intensity.xlsx", index=False)
print("carbon_intensity.xlsx done")

# ── 8. electricity_price ─────────────────────────────────────────────
ep = []
bases_p = {"RegionA": 500, "RegionB": 550, "RegionC": 520,
           "RegionD": 480, "RegionE": 450, "RegionF": 580}
for r in REGIONS:
    base = bases_p[r]
    for h in range(2406):
        v = base + 200 * np.sin((h % 24 - 6) / 24 * 2 * np.pi) + np.random.normal(0, 30)
        ep.append([r, h, round(max(v, 100), 2)])
pd.DataFrame(ep, columns=["Region", "Hour", "Price_Yuan_per_MWh"]).to_excel(
    f"{OUT}/electricity_price.xlsx", index=False)
print("electricity_price.xlsx done")

# ── 9. renewable_generation ────────────────────────────────────────
rg = []
bases_r = {"RegionA": 30, "RegionB": 25, "RegionC": 35,
           "RegionD": 40, "RegionE": 45, "RegionF": 20}
for r in REGIONS:
    base = bases_r[r]
    for h in range(2406):
        v = max(base + 20 * np.sin((h % 24 - 12) / 24 * 2 * np.pi) + np.random.normal(0, 5), 0)
        rg.append([r, h, round(v, 3)])
pd.DataFrame(rg, columns=["Region", "Hour", "AvailableRenewable_MW"]).to_excel(
    f"{OUT}/renewable_generation.xlsx", index=False)
print("renewable_generation.xlsx done")

print("ALL DONE →", OUT)
