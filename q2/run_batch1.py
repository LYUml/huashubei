"""
Carbon-aware scheduling - Part 1: Data prep + first 10000 tasks
"""
import pandas as pd
import numpy as np
import time
import pickle

print("Loading...", flush=True)
t0 = time.time()

wt = pd.read_excel('/data/inputs/workload_trace.xlsx')
rtd = pd.read_excel('/data/inputs/region_time_data.xlsx')
gpu_info = pd.read_excel('/data/inputs/GPU_information.xlsx')
nl = pd.read_excel('/data/inputs/network_latency.xlsx')
pm = pd.read_excel('/data/inputs/power_mapping.xlsx')

wt['dh'] = (wt['EstimatedDuration_min'] / 60.0).apply(np.ceil).astype(np.int32)

rtd_main = rtd[rtd['DataPeriod']=='Main_0_2399'].set_index(['Hour','Region'])
rtd_close = rtd[rtd['DataPeriod']=='Closure_2400_2406'].set_index(['Hour','Region'])

pue = dict(zip(gpu_info['Region'], gpu_info['PUE']))
avgpu = dict(zip(gpu_info['Region'], gpu_info['Available_GPU']))
maxit = dict(zip(gpu_info['Region'], gpu_info['Max_IT_Power_MW']))
maxfac = dict(zip(gpu_info['Region'], gpu_info['Max_Facility_Power_MW']))
ppg = dict(zip(pm['TaskType'], pm['GPU_Power_MW_per_EquivalentGPU']))

regions = ['RegionA','RegionB','RegionC','RegionD','RegionE','RegionF']
RID = {r:i for i,r in enumerate(regions)}

lat_mat = np.full((6,6), 999, dtype=np.int32)
for _, row in nl.iterrows():
    lat_mat[RID[row['FromRegion']], RID[row['ToRegion']]] = int(row['NetworkLatency_ms'])

H = 2406
price_arr = np.zeros((H,6))
ci_arr = np.zeros((H,6))
avail_r_arr = np.zeros((H,6))
nonai_arr = np.zeros((H,6))

for i,r in enumerate(regions):
    for h in range(2400):
        price_arr[h,i] = rtd_main.loc[(h,r)]['ElectricityPrice_CNY_per_MWh']
        ci_arr[h,i] = rtd_main.loc[(h,r)]['CarbonIntensity_tCO2_per_MWh']
        avail_r_arr[h,i] = rtd_main.loc[(h,r)]['AvailableRenewable_MW']
        nonai_arr[h,i] = rtd_main.loc[(h,r)]['NonAI_IT_Load_MW']
    for h in range(2400, H):
        if (h,r) in rtd_close.index:
            price_arr[h,i] = rtd_close.loc[(h,r)]['ElectricityPrice_CNY_per_MWh']
            ci_arr[h,i] = rtd_close.loc[(h,r)]['CarbonIntensity_tCO2_per_MWh']
            avail_r_arr[h,i] = rtd_close.loc[(h,r)]['AvailableRenewable_MW']
            nonai_arr[h,i] = rtd_close.loc[(h,r)]['NonAI_IT_Load_MW']

print(f"Data ready in {time.time()-t0:.1f}s", flush=True)

# Sort tasks
pri = {'RealTimeInference':0,'BatchInference':1,'AITraining':2}
wt['p'] = wt['TaskType'].map(pri)
wt.sort_values(['ArrivalHour','p','TaskID'], inplace=True, kind='mergesort')
wt.reset_index(drop=True, inplace=True)

N = len(wt)
print(f"Tasks: {N}", flush=True)

# Extract arrays
ttype = wt['TaskType'].values
tsrc = wt['SourceRegion'].values
tarr = wt['ArrivalHour'].values.astype(np.int32)
tgpu = wt['GPU_Demand'].values.astype(np.float64)
tdur = wt['dh'].values.astype(np.int32)
tmaxlat = wt['MaxLatency_ms'].values.astype(np.int32)
tlf = wt['LatestFinishHour'].values.astype(np.int32)
tes = wt['EarliestStartHour'].values.astype(np.int32)

# Type index
type_idx = np.zeros(N, dtype=np.int8)
for i, t in enumerate(ttype):
    if t == 'RealTimeInference': type_idx[i] = 0
    elif t == 'BatchInference': type_idx[i] = 1
    else: type_idx[i] = 2
ppg_arr = np.array([ppg['RealTimeInference'], ppg['BatchInference'], ppg['AITraining']], dtype=np.float64)

# State
gpu_usg = np.zeros((6,H), dtype=np.float64)
it_pwr = np.zeros((6,H), dtype=np.float64)

# Pre-compute region param arrays
pue_for = np.array([pue[regions[r]] for r in range(6)], dtype=np.float64)
avgpu_for = np.array([avgpu[regions[r]] for r in range(6)], dtype=np.float64)
maxit_for = np.array([maxit[regions[r]] for r in range(6)], dtype=np.float64)
maxfac_for = np.array([maxfac[regions[r]] for r in range(6)], dtype=np.float64)

# Save prepared data for batch processing
np.savez('/data/workspace/prep.npz',
    ttype=ttype, tsrc=tsrc, tarr=tarr, tgpu=tgpu, tdur=tdur,
    tmaxlat=tmaxlat, tlf=tlf, tes=tes, type_idx=type_idx,
    ppg_arr=ppg_arr, lat_mat=lat_mat,
    price_arr=price_arr, ci_arr=ci_arr,
    avail_r_arr=avail_r_arr, nonai_arr=nonai_arr,
    gpu_usg=gpu_usg, it_pwr=it_pwr,
    pue_for=pue_for, avgpu_for=avgpu_for,
    maxit_for=maxit_for, maxfac_for=maxfac_for,
    RID_keys=list(RID.keys()), RID_vals=list(RID.values()),
    regions=np.array(regions),
    N=np.array([N]))

# Also save task IDs separately
tid = wt['TaskID'].values
np.save('/data/workspace/tid.npy', tid)

print(f"Saved prep data. N={N}", flush=True)
print("Part 1 DONE", flush=True)
