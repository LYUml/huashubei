"""运行均衡策略（最简版，验证可行性）"""
import pandas as pd
import numpy as np
import time, sys

print("Loading data...", flush=True)
t0 = time.time()

wt = pd.read_excel('/data/inputs/workload_trace.xlsx')
rtd = pd.read_excel('/data/inputs/region_time_data.xlsx')
gpu_info = pd.read_excel('/data/inputs/GPU_information.xlsx')
nl = pd.read_excel('/data/inputs/network_latency.xlsx')
pm = pd.read_excel('/data/inputs/power_mapping.xlsx')
si = pd.read_excel('/data/inputs/storage_information.xlsx')

wt['Duration_h'] = np.ceil(wt['EstimatedDuration_min'] / 60).astype(int)
rtd_main = rtd[rtd['DataPeriod']=='Main_0_2399'].copy().set_index(['Hour','Region'])
rtd_close = rtd[rtd['DataPeriod']=='Closure_2400_2406'].copy().set_index(['Hour','Region'])

pue_map = dict(zip(gpu_info['Region'], gpu_info['PUE']))
avail_gpu = dict(zip(gpu_info['Region'], gpu_info['Available_GPU']))
max_it_p = dict(zip(gpu_info['Region'], gpu_info['Max_IT_Power_MW']))
max_fac_p = dict(zip(gpu_info['Region'], gpu_info['Max_Facility_Power_MW']))
power_map = dict(zip(pm['TaskType'], pm['GPU_Power_MW_per_EquivalentGPU']))

latency_dict = {}
for _, row in nl.iterrows():
    latency_dict[(row['FromRegion'], row['ToRegion'])] = row['NetworkLatency_ms']

regions = ['RegionA','RegionB','RegionC','RegionD','RegionE','RegionF']

print(f"Data loaded in {time.time()-t0:.1f}s. Tasks={len(wt)}", flush=True)

# Region states
gpu_usage = {r: np.zeros(2406, dtype=np.float64) for r in regions}
it_power = {r: np.zeros(2406, dtype=np.float64) for r in regions}

# Weights (balanced)
W = {'w_cost':0.5, 'w_carbon':0.5, 'w_latency':0.3, 'w_renewable':1.0, 'lat_f':0.1}

# Sort tasks
pri = {'RealTimeInference':0, 'BatchInference':1, 'AITraining':2}
wt2 = wt.copy()
wt2['pri'] = wt2['TaskType'].map(pri)
wt2 = wt2.sort_values(['ArrivalHour','pri','TaskID']).reset_index(drop=True)

results = []
mig_count = 0
incr_cost_total = 0.0
incr_carbon_total = 0.0

t_start = time.time()
N = len(wt2)

for idx, task in wt2.iterrows():
    src = task['SourceRegion']
    max_lat = task['MaxLatency_ms']
    dur = int(task['Duration_h'])
    gpu_d = task['GPU_Demand']
    ttype = task['TaskType']
    ppg = power_map[ttype]
    arrival = int(task['ArrivalHour'])
    lf = int(task['LatestFinishHour'])
    
    # Feasible regions
    feas_r = [r for r in regions if latency_dict.get((src,r),999) <= max_lat]
    
    # Start candidates
    if ttype == 'RealTimeInference':
        s_cands = [arrival]
    else:
        es = int(task.get('EarliestStartHour', arrival))
        ls = min(lf - dur + 1, 2405)
        ls = min(ls, arrival + 300)
        s_cands = list(range(es, ls+1))
    
    best_score = float('inf')
    best_dec = None
    best_det = None
    
    for tr in feas_r:
        pue = pue_map[tr]
        for sh in s_cands:
            sh = int(sh)
            eh = sh + dur - 1
            if eh > 2405:
                continue
            
            # Capacity check
            ok = True
            new_it_peak = 0
            for h in range(sh, min(eh+1, 2406)):
                gu = gpu_usage[tr][h] + gpu_d
                if gu > avail_gpu[tr]:
                    ok = False; break
                nip = it_power[tr][h] + gpu_d * ppg
                # nonai
                if h <= 2399:
                    nonai = rtd_main.loc[(h,tr)]['NonAI_IT_Load_MW']
                elif (h,tr) in rtd_close.index:
                    nonai = rtd_close.loc[(h,tr)]['NonAI_IT_Load_MW']
                else:
                    nonai = 0
                if nip + nonai > max_it_p[tr]:
                    ok = False; break
                fl = (nip + nonai) * pue
                if fl > max_fac_p[tr]:
                    ok = False; break
                if nip > new_it_peak:
                    new_it_peak = nip
            if not ok:
                continue
            
            # Compute incremental cost/carbon
            icost = 0.0; icarb = 0.0; irenew = 0.0
            for h in range(sh, min(eh+1, 2406)):
                if h <= 2399:
                    row = rtd_main.loc[(h,tr)]
                elif (h,tr) in rtd_close.index:
                    row = rtd_close.loc[(h,tr)]
                else:
                    continue
                il = gpu_d * ppg * pue  # incremental facility load MW
                icost += il * row['ElectricityPrice_CNY_per_MWh']
                icarb += il * row['CarbonIntensity_tCO2_per_MWh']
                ar = row['AvailableRenewable_MW']
                if ar > 0:
                    irenew += min(gpu_d * ppg, ar)
            
            lat = latency_dict.get((src,tr),999)
            cost_n = icost / 1000.0
            if tr != src:
                lpen = lat * W['lat_f']
            else:
                lpen = 0.0
            
            score = W['w_cost']*cost_n + W['w_carbon']*icarb + W['w_latency']*lpen + W['w_renewable']*(-irenew/100.0)
            
            if score < best_score:
                best_score = score
                best_dec = (tr, sh)
                best_det = {'cost':icost,'carbon':icarb,'latency':lat,'renew':irenew,'mig':(tr!=src)}
    
    if best_dec is None:
        best_dec = (src, arrival)
        best_det = {'cost':0,'carbon':0,'latency':0,'renew':0,'mig':False}
    
    tr, sh = best_dec
    eh = min(sh + dur - 1, 2405)
    
    # Apply
    for h in range(sh, eh+1):
        gpu_usage[tr][h] += gpu_d
        it_power[tr][h] += gpu_d * ppg
    
    if best_det['mig']:
        mig_count += 1
    incr_cost_total += best_det['cost']
    incr_carbon_total += best_det['carbon']
    
    results.append({
        'TaskID': task['TaskID'], 'TaskType': ttype,
        'SourceRegion': src, 'AssignedRegion': tr,
        'StartHour': sh, 'EndHour': eh,
        'Duration_h': dur, 'GPU_Demand': gpu_d,
        'MaxLatency_ms': max_lat, 'IsMigrated': best_det['mig'],
        'NetworkLatency_ms': best_det['latency'],
        'IncrCost': best_det['cost'],
        'IncrCarbon': best_det['carbon'],
        'RenewableUsed': best_det['renew'],
    })
    
    if idx % 10000 == 0 and idx > 0:
        el = time.time() - t_start
        print(f"  {idx}/{N} ({idx/N*100:.0f}%) | mig={mig_count}| cost={incr_cost_total:,.0f} | {el:.0f}s", flush=True)

elapsed = time.time() - t_start
print(f"\nDONE in {elapsed:.1f}s!", flush=True)
print(f"  Tasks: {N}, Migrated: {mig_count} ({mig_count/N*100:.1f}%)", flush=True)
print(f"  Total incr cost: {incr_cost_total:,.0f} CNY", flush=True)
print(f"  Total incr carbon: {incr_carbon_total:,.0f} tCO2", flush=True)

# Save
res_df = pd.DataFrame(results)
res_df.to_csv('/data/workspace/schedule_balanced.csv', index=False)
print(f"  Saved to schedule_balanced.csv ({len(res_df)} rows)", flush=True)

# Quick global metrics
print("\nComputing global metrics...", flush=True)
total_cost = 0.0; total_carbon = 0.0
total_ra = 0.0; total_ru = 0.0
peak_ni = 0.0

for h in range(2406):
    for r in regions:
        if h <= 2399:
            row = rtd_main.loc[(h,r)]
        elif (h,r) in rtd_close.index:
            row = rtd_close.loc[(h,r)]
        else:
            continue
        
        ai = it_power[r][h]
        nonai = row['NonAI_IT_Load_MW']
        itl = nonai + ai
        pue = pue_map[r]
        tl = itl * pue
        
        ar = row['AvailableRenewable_MW']
        ur = min(ai, ar) if ar > 0 else 0
        
        gp = max(0, tl - ur)
        gs = 0.0
        
        price = row['ElectricityPrice_CNY_per_MWh']
        ci = row['CarbonIntensity_tCO2_per_MWh']
        
        total_cost += gp * price - gs * row.get('SellPrice_CNY_per_MWh',0)
        total_carbon += gp * ci
        total_ra += ar
        total_ru += ur
        
        ni = gp - gs
        if ni > peak_ni:
            peak_ni = ni

ru_pct = total_ru / total_ra * 100 if total_ra > 0 else 0
migrated_df = res_df[res_df['IsMigrated']==True]
avg_lat = migrated_df['NetworkLatency_ms'].mean() if len(migrated_df)>0 else 0

print(f"\n{'='*50}", flush=True)
print(f"  TOTAL COST:        {total_cost:>15,.0f} CNY", flush=True)
print(f"  TOTAL CARBON:      {total_carbon:>15,.0f} tCO2", flush=True)
print(f"  AVG MIGRATION LAT:  {avg_lat:>10.1f} ms", flush=True)
print(f"  RENEWABLE UTIL:    {ru_pct:>9.1f}%", flush=True)
print(f"  PEAK NET IMPORT:   {peak_ni:>10.1f} MW", flush=True)
print(f"  MIGRATION RATE:    {mig_count/N*100:>9.1f}%", flush=True)
print(f"{'='*50}", flush=True)

base_cost = 1_802_343_507
base_carbon = 2_045_359
print(f"\n  vs BASELINE:", flush=True)
print(f"    Cost reduction:   {(1-total_cost/base_cost)*100:.1f}%", flush=True)
print(f"    Carbon reduction: {(1-total_carbon/base_carbon)*100:.1f}%", flush=True)
