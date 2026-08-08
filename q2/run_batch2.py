"""
Carbon-aware scheduling - Part 2: Process tasks in batches
Processes tasks 0..N-1, saves results incrementally
"""
import numpy as np
import time

print("Loading prep data...", flush=True)
data = np.load('/data/workspace/prep.npz', allow_pickle=True)

ttype = data['ttype']
tsrc = data['tsrc']
tarr = data['tarr'].astype(np.int32)
tgpu = data['tgpu'].astype(np.float64)
tdur = data['tdur'].astype(np.int32)
tmaxlat = data['tmaxlat'].astype(np.int32)
tlf = data['tlf'].astype(np.int32)
tes = data['tes'].astype(np.int32)
type_idx = data['type_idx'].astype(np.int8)
ppg_arr = data['ppg_arr']
lat_mat = data['lat_mat']
price_arr = data['price_arr']
ci_arr = data['ci_arr']
avail_r_arr = data['avail_r_arr']
nonai_arr = data['nonai_arr']
gpu_usg = data['gpu_usg']
it_pwr = data['it_pwr']
pue_for = data['pue_for']
avgpu_for = data['avgpu_for']
maxit_for = data['maxit_for']
maxfac_for = data['maxfac_for']
regions = data['regions'].tolist()
N = int(data['N'][0])

print(f"N={N}, Regions={regions}", flush=True)

# Weights
wc, wcarb, wlat, wrev = 0.5, 0.5, 0.3, 1.0
LAT_F = 0.1

# Result storage (append to file incrementally)
res_region = np.zeros(N, dtype='U10')
res_sh = np.zeros(N, dtype=np.int32)
res_eh = np.zeros(N, dtype=np.int32)
res_mig = np.zeros(N, dtype=bool)
res_cost = np.zeros(N)
res_carb = np.zeros(N)
res_lat = np.zeros(N)
res_rew = np.zeros(N)

mig_count = 0
total_cost = 0.0
total_carb = 0.0

t_start = time.time()
last_save = t_start

BATCH = 5000

for idx in range(N):
    src_r_str = tsrc[idx]
    si = 0
    for ri, rn in enumerate(regions):
        if rn == src_r_str:
            si = ri
            break
    
    max_lat = tmaxlat[idx]
    dur = tdur[idx]
    gpu_d = tgpu[idx]
    ti = type_idx[idx]
    ppg_val = ppg_arr[ti]
    arrival = tarr[idx]
    lf = tlf[idx]
    
    feas_mask = lat_mat[si, :] <= max_lat
    
    if ti == 0:  # RealTime
        n_sc = 1
        s_cands = np.array([arrival], dtype=np.int32)
    else:
        es = tes[idx]
        ls = min(lf - dur + 1, 2405)
        ls = min(ls, arrival + 300)
        if ls < es:
            ls = es
        n_sc = ls - es + 1
        s_cands = np.arange(es, ls+1, dtype=np.int32)
    
    best_score = 1e18
    best_r = -1
    best_sh = -1
    best_cost = 0.0
    best_carb_v = 0.0
    best_lat_v = 0.0
    best_rew = 0.0
    
    for tr_idx in range(6):
        if not feas_mask[tr_idx]:
            continue
        
        pue_v = pue_for[tr_idx]
        avg_v = avgpu_for[tr_idx]
        mit_v = maxit_for[tr_idx]
        mfac_v = maxfac_for[tr_idx]
        
        gu_arr = gpu_usg[tr_idx]
        ip_arr = it_pwr[tr_idx]
        
        for sc_i in range(n_sc):
            sh = int(s_cands[sc_i])
            eh = min(sh + dur - 1, 2405)
            
            hrs = np.arange(sh, eh+1)
            max_gu = gu_arr[hrs].max() + gpu_d
            if max_gu > avg_v:
                continue
            
            new_ip = ip_arr[hrs] + gpu_d * ppg_val
            max_nai = nonai_arr[hrs, tr_idx].max()
            if new_ip.max() + max_nai > mit_v:
                continue
            
            max_fac_val = ((new_ip + max_nai) * pue_v).max()
            if max_fac_val > mfac_v:
                continue
            
            il = gpu_d * ppg_val * pue_v
            hcost = il * price_arr[hrs, tr_idx].sum()
            hcarb = il * ci_arr[hrs, tr_idx].sum()
            
            hrew = 0.0
            ar_sum = avail_r_arr[hrs, tr_idx].sum()
            if ar_sum > 0:
                hrew = min(gpu_d * ppg_val, ar_sum / len(hrs)) * len(hrs)
            
            lat_v = lat_mat[si, tr_idx]
            
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
                best_carb_v = hcarb
                best_lat_v = lat_v
                best_rew = hrew
    
    if best_r == -1:
        best_r = si
        best_sh = arrival
        best_cost = 0.0; best_carb_v = 0.0; best_lat_v = 0.0; best_rew = 0.0
    
    eh_final = min(best_sh + dur - 1, 2405)
    
    # Apply
    gu_a = gpu_usg[best_r]
    ip_a = it_pwr[best_r]
    ppg_v2 = ppg_val
    for h in range(best_sh, eh_final+1):
        gu_a[h] += gpu_d
        ip_a[h] += gpu_d * ppg_v2
    
    is_mig = (best_r != si)
    rname = regions[best_r]
    
    res_region[idx] = rname
    res_sh[idx] = best_sh
    res_eh[idx] = eh_final
    res_mig[idx] = is_mig
    res_cost[idx] = best_cost
    res_carb[idx] = best_carb_v
    res_lat[idx] = best_lat_v
    res_rew[idx] = best_rew
    
    if is_mig:
        mig_count += 1
    total_cost += best_cost
    total_carb += best_carb_v
    
    if idx % BATCH == 0 and idx > 0:
        el = time.time() - t_start
        print(f"  {idx}/{N} ({idx/N*100:.0f}%) mig={mig_count} cost={total_cost:,.0f} t={el:.0f}s", flush=True)
        # Incremental save
        np.savez('/data/workspace/res_partial.npz',
                res_region=res_region, res_sh=res_sh, res_eh=res_eh,
                res_mig=res_mig, res_cost=res_cost, res_carb=res_carb,
                res_lat=res_lat, res_rew=res_rew,
                mig_count=np.array([mig_count]),
                total_cost=np.array([total_cost]),
                total_carb=np.array([total_carb]),
                gpu_usg=gpu_usg, it_pwr=it_pwr,
                last_idx=np.array([idx]))

elapsed = time.time() - t_start
print(f"\nScheduling DONE in {elapsed:.1f}s", flush=True)
print(f"  Migrated: {mig_count}/{N} ({mig_count/N*100:.1f}%)", flush=True)
print(f"  Total incr cost: {total_cost:,.0f} CNY", flush=True)
print(f"  Total incr carbon: {total_carb:,.0f} tCO2", flush=True)

# Save final results
np.savez('/data/workspace/results_final.npz',
        res_region=res_region, res_sh=res_sh, res_eh=res_eh,
        res_mig=res_mig, res_cost=res_cost, res_carb=res_carb,
        res_lat=res_lat, res_rew=res_rew,
        mig_count=np.array([mig_count]),
        total_cost=np.array([total_cost]),
        total_carb=np.array([total_carb]),
        gpu_usg=gpu_usg, it_pwr=it_pwr)
print("Saved results_final.npz", flush=True)
