"""
碳感知调度 - 优化版（向量化加速）
"""
import pandas as pd
import numpy as np
import time

print("Loading...", flush=True)
t0 = time.time()

wt = pd.read_excel('/data/inputs/workload_trace.xlsx')
rtd = pd.read_excel('/data/inputs/region_time_data.xlsx')
gpu_info = pd.read_excel('/data/inputs/GPU_information.xlsx')
nl = pd.read_excel('/data/inputs/network_latency.xlsx')
pm = pd.read_excel('/data/inputs/power_mapping.xlsx')

wt['dh'] = np.ceil(wt['EstimatedDuration_min']/60).astype(np.int32)

rtd_main = rtd[rtd['DataPeriod']=='Main_0_2399'].set_index(['Hour','Region'])
rtd_close = rtd[rtd['DataPeriod']=='Closure_2400_2406'].set_index(['Hour','Region'])

# Pre-extract arrays for speed
pue = dict(zip(gpu_info['Region'], gpu_info['PUE']))
avgpu = dict(zip(gpu_info['Region'], gpu_info['Available_GPU']))
maxit = dict(zip(gpu_info['Region'], gpu_info['Max_IT_Power_MW']))
maxfac = dict(zip(gpu_info['Region'], gpu_info['Max_Facility_Power_MW']))
ppg = dict(zip(pm['TaskType'], pm['GPU_Power_MW_per_EquivalentGPU']))

# Latency matrix as numpy
regions = ['RegionA','RegionB','RegionC','RegionD','RegionE','RegionF']
RID = {r:i for i,r in enumerate(regions)}
lat_mat = np.full((6,6), 999, dtype=np.int32)
for _, row in nl.iterrows():
    a, b = RID[row['FromRegion']], RID[row['ToRegion']]
    lat_mat[a,b] = int(row['NetworkLatency_ms'])

# Pre-build hourly lookup tables (2406 hours x 6 regions)
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

# State arrays
gpu_usg = np.zeros((6,H), dtype=np.float64)
it_pwr = np.zeros((6,H), dtype=np.float64)

# Sort tasks
pri = {'RealTimeInference':0,'BatchInference':1,'AITraining':2}
wt['p'] = wt['TaskType'].map(pri)
wt.sort_values(['ArrivalHour','p','TaskID'], inplace=True, kind='mergesort')
wt.reset_index(drop=True, inplace=True)

N = len(wt)
print(f"Tasks: {N}", flush=True)

# Pre-extract task arrays
tid = wt['TaskID'].values
ttype = wt['TaskType'].values
tsrc = wt['SourceRegion'].values
tarr = wt['ArrivalHour'].values.astype(np.int32)
tgpu = wt['GPU_Demand'].values.astype(np.float64)
tdur = wt['Duration_h'].values.astype(np.int32)
tmaxlat = wt['MaxLatency_ms'].values.astype(np.int32)
tlf = wt['LatestFinishHour'].values.astype(np.int32)
tes = wt['EarliestStartHour'].values.astype(np.int32)

# Result storage
res_region = np.zeros(N, dtype='U10')
res_sh = np.zeros(N, dtype=np.int32)
res_eh = np.zeros(N, dtype=np.int32)
res_mig = np.zeros(N, dtype=bool)
res_cost = np.zeros(N)
res_carb = np.zeros(N)
res_lat = np.zeros(N)
res_renew = np.zeros(N)

# Weights
wc, wcarb, wlat, wrev = 0.5, 0.5, 0.3, 1.0
LAT_F = 0.1

mig_count = 0
total_cost = 0.0
total_carb = 0.0

t_start = time.time()

for idx in range(N):
    src_r = tsrc[idx]
    si = RID[src_r]
    max_lat = tmaxlat[idx]
    dur = tdur[idx]
    gpu_d = tgpu[idx]
    ttyp = ttype[idx]
    ppg_val = ppg[ttyp]
    pue_val = pue[src_r]  # will be updated per target
    arrival = tarr[idx]
    lf = tlf[idx]
    
    # Feasible target regions
    feas_mask = lat_mat[si, :] <= max_lat
    feas_list = np.where(feas_mask)[0]
    
    # Start candidates
    if ttyp == 'RealTimeInference':
        s_cands = np.array([arrival], dtype=np.int32)
    else:
        es = tes[idx]
        ls = min(lf - dur + 1, 2405)
        ls = min(ls, arrival + 300)
        s_cands = np.arange(es, ls+1, dtype=np.int32)
    
    best_score = 1e18
    best_r = -1
    best_sh = -1
    best_cost = 0.0
    best_carb = 0.0
    best_lat = 0.0
    best_rew = 0.0
    is_mig = False
    
    for tr_idx in feas_list:
        tr_name = regions[tr_idx]
        pue_v = pue[tr_name]
        ppg_v = ppg_val
        avg_v = avgpu[tr_name]
        mit_v = maxit[tr_name]
        mfac_v = maxfac[tr_name]
        
        gu_arr = gpu_usg[tr_idx]
        ip_arr = it_pwr[tr_idx]
        
        for sc_idx in range(len(s_cands)):
            sh = s_cands[sc_idx]
            eh = min(sh + dur - 1, 2405)
            
            # Capacity check (vectorized)
            hours = np.arange(sh, eh+1)
            max_gu = gu_arr[hours].max() + gpu_d
            if max_gu > avg_v:
                continue
            
            # IT power check
            new_ip_peak = (ip_arr[hours] + gpu_d * ppg_v).max()
            # Need to check with nonai for IT power limit
            max_nonai = nonai_arr[hours, tr_idx].max()
            if new_ip_peak + max_nonai > mit_v:
                continue
            
            # Facility power check
            max_fac = ((ip_arr[hours] + gpu_d * ppg_v + nonai_arr[hours, tr_idx]) * pue_v).max()
            if max_fac > mfac_v:
                continue
            
            # Compute incremental cost/carbon (vectorized)
            il = gpu_d * ppg_v * pue_v  # incremental facility load
            hours_arr = hours
            hcost = (il * price_arr[hours_arr, tr_idx]).sum()
            hcarb = (il * ci_arr[hours_arr, tr_idx]).sum()
            
            # Renewable
            hrew = 0.0
            ar = avail_r_arr[hours_arr, tr_idx]
            if ar.sum() > 0:
                ai_power = gpu_d * ppg_v
                hrew = min(ai_power, ar.mean()) * len(hours)  # approximate
            
            # Latency
            lat_v = lat_mat[si, tr_idx]
            
            # Score
            cost_n = hcost / 1000.0
            if tr_idx != si:
                lpen = lat_v * LAT_F
            else:
                lpen = 0.0
            
            score = wc * cost_n + wcarb * hcarb + wlat * lpen + wrev * (-hrew / 100.0)
            
            if score < best_score:
                best_score = score
                best_r = tr_idx
                best_sh = sh
                best_cost = hcost
                best_carb = hcarb
                best_lat = lat_v
                best_rew = hrew
                is_mig = (tr_idx != si)
    
    if best_r == -1:
        best_r = si
        best_sh = arrival
        best_cost = 0.0; best_carb = 0.0; best_lat = 0.0; best_rew = 0.0
        is_mig = False
    
    # Apply
    eh_final = min(best_sh + dur - 1, 2405)
    tr_name = regions[best_r]
    ppg_v = ppg_val
    gu_arr = gpu_usg[best_r]
    ip_arr = it_pwr[best_r]
    for h in range(best_sh, eh_final+1):
        gu_arr[h] += gpu_d
        ip_arr[h] += gpu_d * ppg_v
    
    # Record
    res_region[idx] = tr_name
    res_sh[idx] = best_sh
    res_eh[idx] = eh_final
    res_mig[idx] = is_mig
    res_cost[idx] = best_cost
    res_carb[idx] = best_carb
    res_lat[idx] = best_lat
    res_rew[idx] = best_rew
    
    if is_mig:
        mig_count += 1
    total_cost += best_cost
    total_carb += best_carb
    
    if idx % 10000 == 0 and idx > 0:
        el = time.time() - t_start
        print(f"  {idx}/{N} ({idx/N*100:.0f}%) mig={mig_count} cost={total_cost:,.0f} t={el:.0f}s", flush=True)

elapsed = time.time() - t_start
print(f"\nScheduling DONE in {elapsed:.1f}s", flush=True)
print(f"  Migrated: {mig_count}/{N} ({mig_count/N*100:.1f}%)", flush=True)
print(f"  Total incr cost: {total_cost:,.0f} CNY", flush=True)
print(f"  Total incr carbon: {total_carb:,.0f} tCO2", flush=True)

# Save results
res_df = pd.DataFrame({
    'TaskID': tid,
    'TaskType': ttype,
    'SourceRegion': tsrc,
    'AssignedRegion': res_region,
    'StartHour': res_sh,
    'EndHour': res_eh,
    'GPU_Demand': tgpu,
    'MaxLatency_ms': tmaxlat,
    'IsMigrated': res_mig,
    'NetworkLatency_ms': res_lat,
    'IncrCost': res_cost,
    'IncrCarbon': res_carb,
    'RenewableUsed': res_rew,
})
res_df.to_csv('/data/workspace/schedule_balanced.csv', index=False)
print(f"  Saved schedule_balanced.csv ({len(res_df)} rows)", flush=True)

# ========== Global metrics ==========
print("\nComputing global metrics...", flush=True)
total_c = 0.0; total_co = 0.0
total_ra = 0.0; total_ru = 0.0
peak_ni = 0.0

for h in range(H):
    for ri in range(6):
        rname = regions[ri]
        ai = it_pwr[ri][h]
        nonai = nonai_arr[h,ri]
        itl = nonai + ai
        pue_v = pue[rname]
        tl = itl * pue_v
        
        ar = avail_r_arr[h,ri]
        ur = min(ai, ar) if ar > 0 else 0.0
        
        gp = max(0.0, tl - ur)
        gs = 0.0
        
        price = price_arr[h,ri]
        ci = ci_arr[h,ri]
        
        cost = gp * price - gs * 0.0
        carb = gp * ci
        
        total_c += cost
        total_co += carb
        total_ra += ar
        total_ru += ur
        
        ni = gp - gs
        if ni > peak_ni:
            peak_ni = ni

ru_pct = total_ru / total_ra * 100 if total_ra > 0 else 0
mig_df = res_df[res_df['IsMigrated']==True]
avg_lat = mig_df['NetworkLatency_ms'].mean() if len(mig_df)>0 else 0

print(f"\n{'='*50}", flush=True)
print(f"  TOTAL COST:        {total_c:>15,.0f} CNY", flush=True)
print(f"  TOTAL CARBON:      {total_co:>15,.0f} tCO2", flush=True)
print(f"  AVG MIG LATENCY:   {avg_lat:>10.1f} ms", flush=True)
print(f"  RENEWABLE UTIL:    {ru_pct:>9.1f}%", flush=True)
print(f"  PEAK NET IMPORT:   {peak_ni:>10.1f} MW", flush=True)
print(f"  MIGRATION RATE:    {mig_count/N*100:>9.1f}%", flush=True)
print(f"{'='*50}", flush=True)

bc = 1_802_343_507
bco = 2_045_359
print(f"\n  vs BASELINE:", flush=True)
print(f"    Cost reduction:   {(1-total_c/bc)*100:.1f}%", flush=True)
print(f"    Carbon reduction: {(1-total_co/bco)*100:.1f}%", flush=True)

# Save hourly data for plotting
print("\nSaving hourly data...", flush=True)
hourly_rows = []
for h in range(H):
    for ri in range(6):
        rname = regions[ri]
        ai = it_pwr[ri][h]
        nonai = nonai_arr[h,ri]
        itl = nonai + ai
        tl = itl * pue[rname]
        ar = avail_r_arr[h,ri]
        ur = min(ai, ar) if ar > 0 else 0.0
        gp = max(0.0, tl - ur)
        carb = gp * ci_arr[h,ri]
        cost = gp * price_arr[h,ri]
        hourly_rows.append({
            'Hour':h,'Region':rname,'AI_IT':ai,'NonAI':nonai,
            'TotalLoad':tl,'GridPurchase':gp,'Carbon':carb,'Cost':cost,
            'RenewUsed':ur,'RenewAvail':ar
        })

hdf = pd.DataFrame(hourly_rows)
hdf.to_csv('/data/workspace/hourly_balanced.csv', index=False)
print(f"  Saved hourly_balanced.csv ({len(hdf)} rows)", flush=True)
print("ALL DONE!", flush=True)
