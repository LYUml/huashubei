"""
问题二：碳感知任务调度 —— 可视化 (10 张论文配图)
==================================================
真实数据列名（与 gen_data.py 对齐）：
  region_time_data(+carbon+price+renewable) 主键(Region,Hour):
    NonAI_IT_Load_MW, Baseline_AI_IT_Load_MW,
    IT_Load_MW, Facility_Load_MW,
    CarbonIntensity_tCO2_per_MWh, Price_Yuan_per_MWh,
    AvailableRenewable_MW
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, Patch as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── 全局设置 ────────────────────────────────────────────────
rcParams['font.family'] = 'WenQuanYi Micro Hei'
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 200
rcParams['savefig.bbox'] = 'tight'

RCOL = dict(RegionA='#e74c3c', RegionB='#3498db', RegionC='#2ecc71',
            RegionD='#f39c12', RegionE='#9b59b6', RegionF='#1abc9c')
TCOL = dict(RealTimeInference='#e74c3c', BatchInference='#3498db',
            AITraining='#2ecc71')

# ── 加载 ──────────────────────────────────────────────────────
print("加载数据…")
with open("/data/workspace/preprocessed_data.pkl", 'rb') as f:
    D = pickle.load(f)
with open("/data/workspace/q2_predictions.pkl", 'rb') as f:
    PR = pickle.load(f)
with open("/data/workspace/q2_schedule_result.pkl", 'rb') as f:
    SC = pickle.load(f)

pivot    = D['gpu_pivot']
regions  = D['regions']
rtd      = D['region_time']      # 已合并
preds    = PR['predictions']
actuals   = PR['actuals']
metrics   = PR['metrics']

sched_t  = SC['test_schedule']
sched_l  = SC['local_schedule']
m_test   = SC['test_metrics']
m_local  = SC['local_metrics']
gu       = SC['gpu_usage']
pp       = SC['power_params']
rc       = SC['region_capacity']

print("✅ 数据就绪")

# ============================================================
# 图1 各区域GPU需求时间序列
# ============================================================
print("\n📊 图1: GPU需求时间序列")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for i, r in enumerate(regions):
    ax = axes.flat[i]
    cols = [c for c in pivot.columns if c[0] == r]
    ts = pivot[cols].sum(axis=1)
    ax.fill_between(ts.index, ts.values, alpha=0.25, color=RCOL[r])
    ax.plot(ts.index, ts.values, color=RCOL[r], lw=0.7)
    ax.axvspan(2376, 2399, alpha=0.12, color='#3498db')
    ax.axvspan(2400, 2405, alpha=0.10, color='#e67e22')
    ax.set_title(f'{r} GPU需求时间序列', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour'); ax.set_ylabel('GPU Demand')
    ax.set_xlim(0, 2405)
fig.suptitle('图1  各区域GPU需求时间序列总览', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('/data/workspace/fig1_gpu_demand_timeseries.png'); plt.close()
print("✅ 图1 已保存")

# ============================================================
# 图2 GPU需求热力图
# ============================================================
print("📊 图2: GPU需求热力图")
fig, ax = plt.subplots(figsize=(18, 7))
# pivot 行索引是 0..2398, 需要扩展到 0..2405 以匹配 2406 列
hm = pd.DataFrame(index=regions, columns=range(0, 2406), dtype=float)
for r in regions:
    cols = [c for c in pivot.columns if c[0] == r]
    s = pivot[cols].sum(axis=1)
    # 把 s 对齐到 0..2405
    s = s.reindex(range(2406), fill_value=0.0)
    hm.loc[r] = s.values
im = ax.imshow(hm.values, aspect='auto', cmap='YlOrRd',
               extent=[0, 2405, 0, len(regions)])
ax.set_yticks(np.arange(len(regions)) + 0.5)
ax.set_yticklabels(regions, fontsize=11)
ax.set_xlabel('Hour'); ax.set_title('图2  各区域GPU需求热力图', fontsize=14, fontweight='bold')
ax.axvline(2376, color='#3498db', ls='--', lw=1.5, alpha=0.7)
ax.axvline(2399, color='#3498db', ls='--', lw=1.5, alpha=0.7)
ax.text(2387, len(regions)-0.3, '测试区间', ha='center', fontsize=9, color='#3498db')
plt.colorbar(im, ax=ax, label='GPU Demand')
plt.tight_layout(); plt.savefig('/data/workspace/fig2_gpu_heat_map.png'); plt.close()
print("✅ 图2 已保存")

# ============================================================
# 图3 电价与碳强度时序
# ============================================================
print("📊 图3: 电价与碳强度")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for i, r in enumerate(regions):
    ax = axes.flat[i]
    rd = rtd[rtd['Region'] == r].sort_values('Hour')
    ax2 = ax.twinx()
    ax.plot(rd['Hour'], rd['Price_Yuan_per_MWh'], color='#e74c3c', lw=0.8, label='电价')
    ax2.plot(rd['Hour'], rd['CarbonIntensity_tCO2_per_MWh'], color='#2c3e50', lw=0.8, ls='--', label='碳强度')
    ax.set_ylabel('电价 (元/MWh)', color='#e74c3c')
    ax2.set_ylabel('碳强度 (tCO₂/MWh)', color='#2c3e50')
    ax.set_title(f'{r} 电价与碳强度', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2405)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper right', fontsize=8)
fig.suptitle('图3  各区域逐时电价与碳强度对比', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('/data/workspace/fig3_price_carbon.png'); plt.close()
print("✅ 图3 已保存")

# ============================================================
# 图4 预测 vs 实际
# ============================================================
print("📊 图4: 预测vs实际")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for i, r in enumerate(regions):
    ax = axes.flat[i]
    hs, ps, as_ = [], [], []
    for tt in ['RealTimeInference','BatchInference','AITraining']:
        key = (r, tt)
        if key in preds:
            for h in sorted(preds[key].keys()):
                hs.append(h); ps.append(preds[key][h])
                as_.append(actuals.get(key, {}).get(h, 0))
    if not hs:
        ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
        continue
    ax.scatter(hs, as_, c='#2c3e50', s=20, alpha=0.6, label='实际', zorder=3)
    ax.scatter(hs, ps, c='#e74c3c', s=15, alpha=0.5, marker='x', label='预测', zorder=2)
    ax.set_title(f'{r} 预测vs实际 (2376-2399)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour'); ax.set_ylabel('GPU Demand')
    ax.legend(fontsize=9); ax.set_xlim(2375, 2400)
fig.suptitle('图4  第2376-2399小时 GPU需求预测vs实际', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('/data/workspace/fig4_pred_vs_actual.png'); plt.close()
print("✅ 图4 已保存")

# ============================================================
# 图5 调度甘特图（核心）
# ============================================================
print("📊 图5: 调度甘特图")
fig, ax = plt.subplots(figsize=(20, 9))
y_pos = {r: i for i, r in enumerate(regions)}
handles = {}
for tid, info in sorted(sched_t.items()):
    r, s, e = info['region'], info['start'], info['end']
    dur = e - s + 1
    tt = info['task_type']
    col = TCOL.get(tt, '#999')
    box = FancyBboxPatch((s, y_pos[r]-0.35), dur, 0.7,
                          boxstyle="round,pad=0.05",
                          facecolor=col, edgecolor='black',
                          lw=0.4, alpha=0.8,
                          ls='--' if info['migrated'] else '-')
    ax.add_patch(box)
    if tt not in handles:
        handles[tt] = mpatches.Patch(color=col, label=f'{tt}')
ax.legend(handles=list(handles.values()), loc='upper right', fontsize=10)
ax.set_yticks(list(y_pos.values()))
ax.set_yticklabels(regions, fontsize=12, fontweight='bold')
ax.set_xlabel('Hour'); ax.set_title('图5  碳感知任务调度甘特图 (2376-2405)', fontsize=15, fontweight='bold')
ax.axvspan(2376, 2399, alpha=0.05, color='#3498db')
ax.axvspan(2400, 2405, alpha=0.08, color='#e67e22')
ax.text(2387, -0.8, '← 调度区间 2376-2399 →', ha='center', fontsize=9, color='#3498db')
ax.set_xlim(2374, 2406); ax.set_ylim(-1.5, len(regions)+0.5)
plt.tight_layout(); plt.savefig('/data/workspace/fig5_gantt_chart.png'); plt.close()
print("✅ 图5 已保存")

# ============================================================
# 图6 各区域GPU利用率
# ============================================================
print("📊 图6: GPU利用率")
fig, ax = plt.subplots(figsize=(16, 8))
for r in regions:
    hours = list(range(2376, 2406))
    cap = rc[r]['gpu']
    util = [gu[r].get(h, 0) / cap * 100 for h in hours]
    ax.plot(hours, util, color=RCOL[r], lw=1.5, marker='o', ms=3, label=r)
ax.axvline(2399.5, color='gray', ls=':', lw=1, alpha=0.7)
ax.set_xlabel('Hour'); ax.set_ylabel('GPU利用率 (%)')
ax.set_title('图6  各区域GPU利用率曲线 (2376-2405)', fontsize=15, fontweight='bold')
ax.legend(ncol=3, fontsize=11); ax.set_xlim(2375, 2406); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('/data/workspace/fig6_gpu_utilization.png'); plt.close()
print("✅ 图6 已保存")

# ============================================================
# 图7 任务迁移分析
# ============================================================
print("📊 图7: 迁移分析")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
src_cnt = {r: 0 for r in regions}
mig_out = {r: 0 for r in regions}
mig_in  = {r: 0 for r in regions}
for info in sched_t.values():
    s, d = info['source'], info['region']
    src_cnt[s] += 1
    if info['migrated']:
        mig_out[s] += 1; mig_in[d] += 1

x = np.arange(len(regions)); w = 0.35
axes[0].bar(x-w/2, [src_cnt[r] for r in regions], w, label='总任务', color='#3498db', alpha=0.7)
axes[0].bar(x+w/2, [mig_out[r] for r in regions], w, label='迁出数', color='#e74c3c', alpha=0.7)
axes[0].set_xticks(x); axes[0].set_xticklabels(regions)
axes[0].set_title('各区域任务迁出统计', fontsize=13, fontweight='bold')
axes[0].legend()
axes[1].bar(regions, [mig_in[r] for r in regions],
              color=[RCOL[r] for r in regions], alpha=0.7, edgecolor='black', lw=0.5)
axes[1].set_title('任务迁移目的地分布', fontsize=13, fontweight='bold')
fig.suptitle('图7  任务迁移流向分析', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('/data/workspace/fig7_migration_analysis.png'); plt.close()
print("✅ 图7 已保存")

# ============================================================
# 图8 雷达图对比
# ============================================================
print("📊 图8: 雷达图")
labels_cn = ['运行成本', '碳排放', '平均时延', '迁移率', '新能源利用率']
keys = ['cost_yuan', 'carbon_tco2', 'avg_latency_ms', 'migrate_pct', 'renewable_pct']
L = [m_local[k] for k in keys]
C = [m_test[k]  for k in keys]
mx = [max(a, b) if k != 'renewable_pct' else 100 for a, b, k in zip(L, C, keys)]
norm_L = [v/m if k != 'renewable_pct' else 1-v/100 for v, m, k in zip(L, mx, keys)]
norm_C = [v/m if k != 'renewable_pct' else 1-v/100 for v, m, k in zip(C, mx, keys)]
angles = np.linspace(0, 2*np.pi, len(labels_cn), endpoint=False).tolist()
norm_L += norm_L[:1]; norm_C += norm_C[:1]; angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, norm_L, 'o-', lw=2, color='#3498db', label='本地调度')
ax.fill(angles, norm_L, alpha=0.15, color='#3498db')
ax.plot(angles, norm_C, 's-', lw=2, color='#e74c3c', label='碳感知调度')
ax.fill(angles, norm_C, alpha=0.15, color='#e74c3c')
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_cn, fontsize=12)
ax.set_title('图8  调度方案多指标对比雷达图\n(值越大 = 表现越差)', fontsize=13, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
plt.tight_layout(); plt.savefig('/data/workspace/fig8_radar_chart.png'); plt.close()
print("✅ 图8 已保存")

# ============================================================
# 图9 逐时负荷曲线（含PUE）
# ============================================================
print("📊 图9: 逐时负荷")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
avg_pw = {'RealTimeInference': 0.002, 'BatchInference': 0.003, 'AITraining': 0.005}
for i, r in enumerate(regions):
    ax = axes.flat[i]
    hours = list(range(2376, 2406))
    it_load, nonai, fac_load = [], [], []
    for h in hours:
        gpu = gu[r].get(h, 0)
        # 用平均功率估算
        avg = 0.003  # 全类型平均
        it = gpu * avg
        non = pp.get((r, h), {}).get('nonai', 0.0)
        it_load.append(it); nonai.append(non)
        fac_load.append((it + non) * rc[r]['pue'])
    ax.plot(hours, it_load, color='#3498db', lw=1.5, label='AI IT负荷', marker='o', ms=3)
    ax.plot(hours, nonai, color='#95a5a6', lw=1.5, ls='--', label='NonAI IT负荷')
    ax.plot(hours, fac_load, color='#e74c3c', lw=2, label=f'设施负荷 (×PUE={rc[r]["pue"]})', marker='s', ms=3)
    ax.set_title(f'{r} 逐时负荷曲线', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour'); ax.set_ylabel('功率 (MW)')
    ax.legend(fontsize=8); ax.set_xlim(2375, 2406); ax.grid(alpha=0.3)
fig.suptitle('图9  各区域逐时负荷曲线 (2376-2405, 含PUE换算)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('/data/workspace/fig9_load_curves.png'); plt.close()
print("✅ 图9 已保存")

# ============================================================
# 图10 预测误差分析
# ============================================================
print("📊 图10: 预测误差")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 左：MAPE柱状图
if 'Test_MAPE' in metrics.columns:
    mdf = metrics.copy()
    mdf['label'] = mdf['Region'] + '-' + mdf['TaskType'].str[:2]
    axes[0].barh(mdf['label'], mdf['Test_MAPE'],
                  color=[RCOL.get(r, '#999') for r in mdf['Region']],
                  alpha=0.7, edgecolor='black', lw=0.4)
    axes[0].axvline(mdf['Test_MAPE'].mean(), color='red', ls='--', lw=1.5,
                     label=f'均值={mdf["Test_MAPE"].mean():.1f}%')
    axes[0].set_xlabel('MAPE (%)'); axes[0].set_title('各序列预测MAPE', fontsize=13, fontweight='bold')
    axes[0].legend()
# 右：误差直方图
errs = []
for (r, tt), pd_ in preds.items():
    ad = actuals.get((r, tt), {})
    for h in pd_:
        a = ad.get(h, 0)
        errs.append((pd_[h] - a) / max(a, 1) * 100)
axes[1].hist(errs, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black', lw=0.4)
axes[1].axvline(0, color='black', lw=1)
axes[1].set_xlabel('相对误差 (%)'); axes[1].set_ylabel('频次')
axes[1].set_title('预测相对误差分布', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)
fig.suptitle('图10  预测误差分析', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('/data/workspace/fig10_error_analysis.png'); plt.close()
print("✅ 图10 已保存")

# ── 汇总 ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("🎉 全部10张图绘制完成！")
print("="*60)
for f in ['fig1_gpu_demand_timeseries.png','fig2_gpu_heat_map.png',
         'fig3_price_carbon.png','fig4_pred_vs_actual.png',
         'fig5_gantt_chart.png','fig6_gpu_utilization.png',
         'fig7_migration_analysis.png','fig8_radar_chart.png',
         'fig9_load_curves.png','fig10_error_analysis.png']:
    print(f"  ✅ {f}")
