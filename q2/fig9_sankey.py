"""图9: 任务迁移桑基图 (matplotlib 版)"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

rcParams = plt.rcParams
rcParams['font.family'] = 'WenQuanYi Micro Hei'

df = pd.read_csv("/data/workspace/q2_nsga2_sched_knee.csv")
regions = ['RegionA','RegionB','RegionC','RegionD','RegionE','RegionF']

# 统计流向
flows = defaultdict(float)
for _, row in df.iterrows():
    s, d = row['SourceRegion'], row['AssignedRegion']
    flows[(s, d)] += 1

# 只保留有迁移的流 (s != d)
mig_flows = {k:v for k,v in flows.items() if k[0] != k[1]}

fig, ax = plt.subplots(figsize=(10, 6))

# 左列：源区域 (y 位置)
n = len(regions)
y_left  = {r: i for i, r in enumerate(regions)}
y_right = {r: i for i, r in enumerate(regions)}

# 流宽度归一化
max_flow = max(mig_flows.values()) if mig_flows else 1
width_scale = 4.0 / max_flow

# 颜色
cmap = plt.cm.Set2(np.linspace(0, 1, len(regions)))
color_map = {r: cmap[i] for i, r in enumerate(regions)}

# 画箭头
for (s, d), v in sorted(mig_flows.items(), key=lambda x: -x[1]):
    ys = y_left[s] + 0.5
    yd = y_right[d] + 0.5
    w = v * width_scale
    ax.annotate('', xy=(1.0, yd), xytext=(0.0, ys),
                arrowprops=dict(
                    arrowstyle='-',
                    color=color_map[s],
                    alpha=0.6,
                    lw=w*10,
                    connectionstyle='arc3,rad=0.15',
                ))
    # 标注流量
    mid_y = (ys + yd) / 2
    ax.text(0.5, mid_y, f'{v:.0f}', ha='center', va='center',
            fontsize=8, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# 左右标签
for r, yi in y_left.items():
    ax.text(-0.05, yi+0.5, r, ha='right', va='center', fontsize=11, fontweight='bold')
for r, yi in y_right.items():
    ax.text(1.05, yi+0.5, r, ha='left', va='center', fontsize=11, fontweight='bold')

ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.5, n)
ax.axis('off')
ax.set_title('Knee 方案任务迁移流向 (桑基图)', fontsize=14, fontweight='bold')

# 图例
legend_el = [plt.Line2D([0],[0],color=cmap[i],lw=4,label=regions[i]) for i in range(n)]
ax.legend(handles=legend_el, loc='lower center', ncol=6, fontsize=9,
          bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig("/data/workspace/fig9_migration_sankey.png", bbox_inches='tight', dpi=150)
plt.close()
print("✅ fig9_migration_sankey.png")
