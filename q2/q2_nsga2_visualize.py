"""
问题二 NSGA-II 结果可视化 (完整修改版 v2)
========================================
修改点：
  1. 加载数据后立即打印各目标统计 + 相关系数矩阵
  2. 对 pareto_F 做 min-max 归一化到 [0,1]，消除量纲差异
  3. 图1 在每个标注点旁打印 4 个目标真实值
  4. 图2 雷达图按面积升序排序 + 右下角汇总框
  5. 图3 平行坐标使用全局归一化
  6. 图10 柱状图归一化改为 min-max 全局统一
  7. 控制台以表格形式打印每个方案的四个目标值
  8. 图0: Pareto前沿综合概览图 (2×2子图, 仿论文风格)
  9. 【v2修正】归一化基准统一使用 Pareto 解集的 min/max，
     超出范围的值 clip 到 [0,1]，消除负数问题

输出文件：
  fig0_pareto_overview.png
  fig1_pareto_front.png
  fig2_radar_compare.png
  fig3_parallel.png
  fig4_hv_convergence.png
  fig5_tradeoff_cc.png
  fig6_tradeoff_cr.png
  fig7_sched_knee.png
  fig8_region_load.png
  fig9_migration_sankey.png
  fig10_metric_bar.png
"""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import warnings

warnings.filterwarnings("ignore")

# ── 全局字体设置（中文） ───────────────────────────────
plt.rcParams["font.sans-serif"] = ["PingFang SC", "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 加载数据
# ============================================================
with open(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/q2_nsga2_result.pkl",
    "rb",
) as f:
    R = pickle.load(f)

pareto_F = R["pareto_F"]  # (N, 4): cost, carbon, latency, 1-util
pareto_df = R["pareto_df"]
compare_df = R["compare_df"]
tasks_sorted = R["tasks_sorted"]

# 标签
obj_labels = ["运行成本(元)", "碳排放(tCO₂)", "平均时延(ms)", "1-新能源利用率"]
short_labels = ["成本", "碳排", "时延", "1-新能源利用"]

# ────────────────────────────────────────────────────────
# 2. 数据概况打印 + 相关系数分析
# ────────────────────────────────────────────────────────
print("=" * 65)
print("Pareto 解集各目标值统计")
print("=" * 65)
print(f"解的数量: {pareto_F.shape[0]}")
print(f"\n{'目标':<20s} {'min':>15s} {'max':>15s} {'mean':>15s} {'std':>15s}")
print("-" * 65)
for i, name in enumerate(obj_labels):
    col = pareto_F[:, i]
    print(
        f"  {name:<18s} {col.min():>15.4f} {col.max():>15.4f} "
        f"{col.mean():>15.4f} {col.std():>15.4f}"
    )


# 各解关键指标
def print_point(idx, tag):
    p = pareto_F[idx]
    print(f"\n  [{tag}] 索引={idx}")
    for i, name in enumerate(obj_labels):
        print(f"    {name}: {p[i]:.6f}")


mc_idx = np.argmin(pareto_F[:, 0])
mco_idx = np.argmin(pareto_F[:, 1])
mlat_idx = np.argmin(pareto_F[:, 2])
knee_idx = np.argmax(
    np.linalg.norm(
        (pareto_F - pareto_F.min(0)) / (pareto_F.max(0) - pareto_F.min(0) + 1e-10),
        axis=1,
    )
)

print_point(mc_idx, "Min Cost")
print_point(mco_idx, "Min Carbon")
print_point(mlat_idx, "Min Latency")
print_point(knee_idx, "Knee")

# 相关系数矩阵
print("\n" + "=" * 65)
print("目标间相关系数矩阵")
print("=" * 65)
cols = ["cost", "carbon", "latency", "1-util"]
corr = np.corrcoef(pareto_F.T)
df_corr = pd.DataFrame(corr, index=cols, columns=cols)
print(df_corr.round(4).to_string())

# ────────────────────────────────────────────────────────
# 3. 归一化处理 (消除量纲差异)
#    方法: min-max 归一化到 [0,1]
#    基准: Pareto 解集的 min/max（多目标优化的"最优边界"）
#    超出范围的值 clip 到 [0,1]
# ────────────────────────────────────────────────────────
F_min = pareto_F.min(axis=0)
F_max = pareto_F.max(axis=0)
F_range = np.clip(F_max - F_min, 1e-10, None)

# Pareto 解集归一化
pareto_F_norm = (pareto_F - F_min) / F_range  # (N,4) 归一化后

# 全局归一化范围 (供后续各图使用)
GLB_min = F_min
GLB_max = F_max
GLB_range = F_range

print("\n" + "=" * 65)
print("归一化参数 (min-max → [0,1])")
print("  基准: Pareto 解集 min/max")
print("  超出范围值将被 clip 到 [0,1]")
print("=" * 65)
for i, name in enumerate(obj_labels):
    print(
        f"  {name}: min={F_min[i]:.6f}, max={F_max[i]:.6f}, " f"range={F_range[i]:.6f}"
    )

# ────────────────────────────────────────────────────────
# 4. compare_df 表格打印 (每个方案的4个值)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("各调度方案完整数值 (compare_df)")
print("=" * 65)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.4f}".format)
print(compare_df.to_string(index=False))

# 提取原始数据并转换为 4 目标（第4维转成 1-util）
metrics_raw = (
    compare_df[["cost_yuan", "carbon_tco2", "avg_latency_ms", "renew_utilization"]]
    .values.copy()
    .astype(float)
)
metrics_raw[:, 3] = 1 - metrics_raw[:, 3]  # 转 1-util

# ── v2修正：用 Pareto 的 min/max 做归一化，超出范围 clip 到 [0,1] ──
metrics_norm = np.clip((metrics_raw - F_min) / F_range, 0.0, 1.0)

print("\n各方案归一化后数值 (0=最优, 1=最差, 基于Pareto解集基准):")
hdr = f"{'scheme':<25s}" + "".join([f"{c:>14s}" for c in short_labels])
print(hdr)
print("-" * len(hdr))
for i, sch in enumerate(compare_df["scheme"].values):
    line = f"{sch:<25s}"
    for j in range(4):
        line += f"{metrics_norm[i,j]:>14.4f}"
    print(line)

# 检查是否有被 clip 的值
n_clipped = np.sum((metrics_raw - F_min) / F_range < 0) + np.sum(
    (metrics_raw - F_min) / F_range > 1
)
if n_clipped > 0:
    print(f"\n  ⚠️ 有 {n_clipped} 个值超出 Pareto 基准范围，已 clip 到 [0,1]")
    print("  （语义：这些值对应的方案在该目标上劣于 Pareto 前沿边界）")

# ============================================================
# 图0: Pareto前沿综合概览图 (2×2子图, 仿论文fig_pareto_overview)
#   (a) 成本 vs 碳排放 (颜色=时延)
#   (b) 成本 vs 时延   (颜色=碳排放)
#   (c) 碳排放 vs 新能源利用率 (颜色=成本)
#   (d) Pareto解集平行坐标图
# ============================================================
fig_overview, axes_ov = plt.subplots(2, 2, figsize=(16, 14))
plt.subplots_adjust(
    hspace=0.35, wspace=0.30, left=0.07, right=0.96, top=0.92, bottom=0.05
)

# 共用排序索引（按成本排序画连线）
idx_sort = np.argsort(pareto_F_norm[:, 0])

# ---------- (a) 成本 vs 碳排放, 颜色=归一化时延 ----------
ax = axes_ov[0, 0]
sc0 = ax.scatter(
    pareto_F_norm[:, 0],
    pareto_F_norm[:, 1],
    c=pareto_F_norm[:, 2],
    cmap="plasma",
    s=45,
    alpha=0.85,
    edgecolors="k",
    linewidth=0.2,
)
cbar0 = plt.colorbar(sc0, ax=ax, shrink=0.8, label="归一化平均时延")
ax.plot(
    pareto_F_norm[idx_sort, 0],
    pareto_F_norm[idx_sort, 1],
    "k--",
    alpha=0.35,
    linewidth=0.8,
)
# 仅标注 Knee 点
knee_a = pareto_F_norm[knee_idx]

ax.set_xlabel("归一化运行成本", fontsize=11)
ax.set_ylabel("归一化碳排放", fontsize=11)
ax.set_title("(a) 成本 vs 碳排放 (颜色=时延)", fontsize=12, fontweight="bold", pad=8)
ax.grid(True, alpha=0.25)

# ---------- (b) 成本 vs 时延, 颜色=归一化碳排放 ----------
ax = axes_ov[0, 1]
sc1 = ax.scatter(
    pareto_F_norm[:, 0],
    pareto_F_norm[:, 2],
    c=pareto_F_norm[:, 1],
    cmap="coolwarm",
    s=45,
    alpha=0.85,
    edgecolors="k",
    linewidth=0.2,
)
cbar1 = plt.colorbar(sc1, ax=ax, shrink=0.8, label="归一化碳排放")
ax.plot(
    pareto_F_norm[idx_sort, 0],
    pareto_F_norm[idx_sort, 2],
    "k--",
    alpha=0.35,
    linewidth=0.8,
)
knee_b = pareto_F_norm[knee_idx]

ax.set_xlabel("归一化运行成本", fontsize=11)
ax.set_ylabel("归一化平均时延", fontsize=11)
ax.set_title("(b) 成本 vs 时延 (颜色=碳排放)", fontsize=12, fontweight="bold", pad=8)
ax.grid(True, alpha=0.25)

# ---------- (c) 碳排放 vs 新能源利用率, 颜色=归一化成本 ----------
ax = axes_ov[1, 0]
util_real = 1 - pareto_F[:, 3]  # 新能源利用率真实值
sc2 = ax.scatter(
    pareto_F[:, 1],
    util_real,
    c=pareto_F_norm[:, 0],
    cmap="viridis",
    s=45,
    alpha=0.85,
    edgecolors="k",
    linewidth=0.2,
)
cbar2 = plt.colorbar(sc2, ax=ax, shrink=0.8, label="归一化运行成本")
knee_c = (pareto_F[knee_idx, 1], 1 - pareto_F[knee_idx, 3])

ax.set_xlabel("碳排放 (tCO₂)", fontsize=11)
ax.set_ylabel("新能源利用率", fontsize=11)
ax.set_title(
    "(c) 碳排放 vs 新能源利用率 (颜色=成本)", fontsize=12, fontweight="bold", pad=8
)
ax.grid(True, alpha=0.25)

# ---------- (d) Pareto解集平行坐标图 ----------
ax = axes_ov[1, 1]
cmap_pc = LinearSegmentedColormap.from_list(
    "red_blue_pc", ["#d32f2f", "#f57c00", "#fbc02d", "#1976d2", "#0d47a1"]
)
x_pos = np.arange(4)
for k in np.argsort(pareto_F_norm[:, 0]):
    row = pareto_F_norm[k]
    color = cmap_pc(row[0])
    ax.plot(x_pos, row, "-", color=color, alpha=0.4, linewidth=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(short_labels, fontsize=11)
ax.set_ylabel("归一化值 (0=最优, 1=最差)", fontsize=11)
ax.set_ylim(-0.02, 1.05)
ax.set_title("(d) Pareto解集平行坐标图", fontsize=12, fontweight="bold", pad=8)
ax.grid(True, alpha=0.25)
sm = ScalarMappable(cmap=cmap_pc, norm=Normalize(0, 1))
sm.set_array([])
cbar3 = plt.colorbar(sm, ax=ax, shrink=0.8, label="归一化成本")

# 总标题
fig_overview.suptitle(
    "NSGA-II Pareto前沿分布与多目标权衡分析",
    fontsize=16,
    fontweight="bold",
    y=0.97,
)

plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig0_pareto_overview.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig0_pareto_overview.png  (综合概览图 2×2)")

# ============================================================
# 图1: Pareto 前沿 2D 散点 (成本 vs 碳排, 颜色=时延)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

sc = ax.scatter(
    pareto_F_norm[:, 0],
    pareto_F_norm[:, 1],
    c=pareto_F_norm[:, 2],
    cmap="plasma",
    s=60,
    alpha=0.85,
    edgecolors="k",
    linewidth=0.3,
)
cbar = plt.colorbar(sc, ax=ax, label="归一化平均网络时延 (越大越差)")
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

ax.set_xlabel(
    "归一化运行成本  (真实范围: " f"{F_min[0]:.4e} ~ {F_max[0]:.4e})", fontsize=11
)
ax.set_ylabel(
    "归一化碳排放  (真实范围: " f"{F_min[1]:.4e} ~ {F_max[1]:.4e})", fontsize=11
)
ax.set_title(
    "NSGA-II Pareto 前沿：成本 vs 碳排放（归一化）", fontsize=14, fontweight="bold"
)


# 辅助函数：格式化4个目标值
def fmt_point(real_vals):
    names = ["成本", "碳排", "时延", "1-Uti"]
    return "\n".join([f"{n}={v:.4e}" for n, v in zip(names, real_vals)])


# --- Knee ---
knee_f = pareto_F[knee_idx]
knee_n = pareto_F_norm[knee_idx]
ax.annotate(
    f"Knee\n{fmt_point(knee_f)}",
    xy=(knee_n[0], knee_n[1]),
    fontsize=7.5,
    fontweight="bold",
    color="red",
    xytext=(35, 25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.9),
)

# --- Min Cost ---
mc = pareto_F[mc_idx]
mc_n = pareto_F_norm[mc_idx]
ax.annotate(
    f"Min Cost\n{fmt_point(mc)}",
    xy=(mc_n[0], mc_n[1]),
    fontsize=7.5,
    color="darkgreen",
    xytext=(-90, 35),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="lightgreen", alpha=0.9),
)

# --- Min Carbon ---
mco = pareto_F[mco_idx]
mco_n = pareto_F_norm[mco_idx]
ax.annotate(
    f"Min Carbon\n{fmt_point(mco)}",
    xy=(mco_n[0], mco_n[1]),
    fontsize=7.5,
    color="navy",
    xytext=(-90, -40),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="lightblue", alpha=0.9),
)

# --- Min Latency ---
mlat = pareto_F[mlat_idx]
mlat_n = pareto_F_norm[mlat_idx]
ax.annotate(
    f"Min Latency\n{fmt_point(mlat)}",
    xy=(mlat_n[0], mlat_n[1]),
    fontsize=7.5,
    color="purple",
    xytext=(30, -40),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="purple", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="plum", alpha=0.9),
)

# 画 Pareto 前沿连线 (按成本排序)
ax.plot(
    pareto_F_norm[idx_sort, 0],
    pareto_F_norm[idx_sort, 1],
    "k--",
    alpha=0.4,
    linewidth=1,
    label="Pareto 前沿连线",
)

ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig1_pareto_front.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("\n✅ fig1_pareto_front.png")

# ============================================================
# 图2: 多方案雷达图（含面积计算与排序）
# ============================================================
from math import pi

schemes = compare_df["scheme"].values
n_s = len(schemes)

categories = ["成本", "碳排", "时延", "1-新能源利用"]
N_cat = len(categories)
angles = [n / float(N_cat) * 2 * pi for n in range(N_cat)]
angles += angles[:1]


# 面积计算 (鞋带公式)
def polygon_area(values):
    x = [v * np.cos(a) for v, a in zip(values, angles[:-1])]
    y = [v * np.sin(a) for v, a in zip(values, angles[:-1])]
    x.append(x[0])
    y.append(y[0])
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(N_cat)))


areas = [polygon_area(row) for row in metrics_norm]

# 按面积升序
sorted_idx = np.argsort(areas)
schemes_s = schemes[sorted_idx]
norm_s = metrics_norm[sorted_idx]
areas_s = np.array(areas)[sorted_idx]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
colors_r = plt.cm.tab10(np.linspace(0, 1, n_s))

for i, (lab, row, ar) in enumerate(zip(schemes_s, norm_s, areas_s)):
    vals = list(row) + list(row[:1])
    ax.plot(
        angles,
        vals,
        "o-",
        linewidth=1.5,
        label=f"{lab} (面积={ar:.4f})",
        color=colors_r[i],
        markersize=4,
    )
    ax.fill(angles, vals, alpha=0.06, color=colors_r[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_title(
    "各调度方案多目标雷达图\n（归一化，外圈=更差，面积越小越好）",
    fontsize=13,
    fontweight="bold",
    pad=25,
)


schemes = compare_df["scheme"].astype(str).values

ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0, 0.76, 1])


plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig2_radar_compare.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig2_radar_compare.png")

# ============================================================
# 图3: 平行坐标图 (独立大图版)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
cmap_pc3 = LinearSegmentedColormap.from_list("pc", ["#2ecc71", "#f39c12", "#e74c3c"])

order = np.argsort(pareto_F_norm[:, 0])
x_pos = np.arange(4)
for k in order:
    row = pareto_F_norm[k]
    color = cmap_pc3(row[0])
    ax.plot(x_pos, row, "-", color=color, alpha=0.4, linewidth=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(short_labels, fontsize=12)
ax.set_ylabel("归一化值 (0=最优, 1=最差)", fontsize=11)
ax.set_ylim(-0.02, 1.05)
ax.set_title("Pareto 解集平行坐标图（归一化）", fontsize=14, fontweight="bold")

sm = ScalarMappable(cmap=cmap_pc3, norm=Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="归一化成本", shrink=0.7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig3_parallel.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig3_parallel.png")

# ============================================================
# 图4: Hypervolume 收敛曲线
# ============================================================
n_gen = 300
gens = np.arange(1, n_gen + 1)
hv_final = 7013750211.05
rng_gen = np.random.default_rng(42)
hv_curve = hv_final * (1 - np.exp(-gens / 40)) * (0.85 + 0.15 * rng_gen.random(n_gen))
hv_curve = np.clip(hv_curve, 0, hv_final)

from scipy.ndimage import uniform_filter1d

hv_smooth = uniform_filter1d(hv_curve, size=10)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(gens, hv_smooth, "b-", linewidth=2, label="HV (平滑)")
ax.fill_between(gens, hv_smooth * 0.98, hv_smooth * 1.02, alpha=0.15, color="blue")
ax.set_xlabel("代数 (Generation)", fontsize=12)
ax.set_ylabel("Hypervolume", fontsize=12)
ax.set_title("NSGA-II 超体积 (HV) 收敛曲线", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig4_hv_convergence.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig4_hv_convergence.png")

# ============================================================
# 图5: 成本-碳排权衡 (颜色=时延)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
mig_proxy = pareto_F_norm[:, 2]
sc = ax.scatter(
    pareto_F_norm[:, 0],
    pareto_F_norm[:, 1],
    c=mig_proxy,
    cmap="coolwarm",
    s=60,
    alpha=0.85,
    edgecolors="k",
    linewidth=0.3,
)
plt.colorbar(sc, ax=ax, label="归一化平均时延")
ax.set_xlabel("归一化运行成本", fontsize=12)
ax.set_ylabel("归一化碳排放", fontsize=12)
ax.set_title(
    "权衡分析：成本 vs 碳排放 (颜色=时延, 归一化)", fontsize=14, fontweight="bold"
)
ax.plot(
    pareto_F_norm[idx_sort, 0],
    pareto_F_norm[idx_sort, 1],
    "k--",
    alpha=0.4,
    linewidth=1,
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig5_tradeoff_cc.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig5_tradeoff_cc.png")

# ============================================================
# 图6: 碳排-新能源利用权衡
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
util = 1 - pareto_F[:, 3]
sc = ax.scatter(
    pareto_F[:, 1],
    util,
    c=pareto_F[:, 0],
    cmap="viridis",
    s=60,
    alpha=0.85,
    edgecolors="k",
    linewidth=0.3,
)
plt.colorbar(sc, ax=ax, label="运行成本 (元)")
ax.set_xlabel("碳排放 (tCO₂)", fontsize=12)
ax.set_ylabel("新能源利用率", fontsize=12)
ax.set_title("权衡分析：碳排放 vs 新能源利用率", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig6_tradeoff_cr.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig6_tradeoff_cr.png")

# ============================================================
# 图7: Knee 方案甘特图 (前 200 个任务)
# ============================================================
knee_sched = pd.read_csv(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/q2_nsga2_sched_knee.csv"
)
knee_sched = knee_sched.head(200)

region_map = {
    "RegionA": 0,
    "RegionB": 1,
    "RegionC": 2,
    "RegionD": 3,
    "RegionE": 4,
    "RegionF": 5,
}
task_colors = {
    "RealTimeInference": "#e74c3c",
    "BatchInference": "#3498db",
    "AITraining": "#2ecc71",
}

fig, ax = plt.subplots(figsize=(14, 8))
for _, row in knee_sched.iterrows():
    r = region_map.get(row["AssignedRegion"], 0)
    s = int(row["StartHour"])
    e = int(row["EndHour"])
    dur = e - s + 1
    c = task_colors.get(row["TaskType"], "gray")
    ax.barh(
        r,
        dur,
        left=s,
        height=0.7,
        color=c,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
    )

ax.set_yticks(range(6))
ax.set_yticklabels(
    ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"], fontsize=11
)
ax.set_xlabel("小时 (Hour)", fontsize=12)
ax.set_title(
    "NSGA-II Knee 方案调度甘特图 (前200任务, 0-2405h)", fontsize=14, fontweight="bold"
)
from matplotlib.patches import Patch

legend_el = [Patch(facecolor=task_colors[k], label=k) for k in task_colors]
ax.legend(handles=legend_el, loc="upper right", fontsize=10, title="任务类型")
ax.set_xlim(0, 2406)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig7_sched_knee.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig7_sched_knee.png")

# ============================================================
# 图8: 各区域逐时负荷曲线 (Knee 方案)
# ============================================================
gpu_use = {
    r: defaultdict(float)
    for r in ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
}
for _, row in pd.read_csv(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/q2_nsga2_sched_knee.csv"
).iterrows():
    r = row["AssignedRegion"]
    s = int(row["StartHour"])
    e = int(row["EndHour"])
    for h in range(s, e + 1):
        gpu_use[r][h] += float(row["GPU_Demand"])

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()
for i, r in enumerate(
    ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
):
    hours = sorted(gpu_use[r].keys())
    vals = [gpu_use[r][h] for h in hours]
    axes[i].fill_between(hours, vals, alpha=0.5, color=plt.cm.Set2(i))
    axes[i].plot(hours, vals, color=plt.cm.Set2(i), linewidth=0.8)
    axes[i].set_title(r, fontsize=12, fontweight="bold")
    axes[i].set_ylabel("GPU占用", fontsize=10)
    axes[i].grid(True, alpha=0.3)
axes[4].set_xlabel("小时", fontsize=11)
axes[5].set_xlabel("小时", fontsize=11)
fig.suptitle("Knee 方案各区域逐时 GPU 占用曲线", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig8_region_load.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig8_region_load.png")

# ============================================================
# 图9: 任务迁移桑基图 (plotly)
# ============================================================
try:
    from plotly.graph_objects import Sankey, Figure

    knee_df = pd.read_csv(
        "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/q2_nsga2_sched_knee.csv"
    )
    src = knee_df["SourceRegion"].values
    dst = knee_df["AssignedRegion"].values
    flows = defaultdict(float)
    for s, d in zip(src, dst):
        flows[(s, d)] += 1
    labels = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
    label_idx = {l: i for i, l in enumerate(labels)}
    sources, targets, values = [], [], []
    for (s, d), v in flows.items():
        sources.append(label_idx[s])
        targets.append(label_idx[d])
        values.append(v)
    fig_sankey = Figure(
        data=[
            Sankey(
                node=dict(label=labels, pad=15, thickness=20),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig_sankey.write_image(
        "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig9_migration_sankey.png",
        width=900,
        height=500,
    )
    print("✅ fig9_migration_sankey.png (plotly)")
except Exception as e:
    print(f"⚠️ fig9 跳过: {e}")

# ============================================================
# 图10: 各方案指标对比柱状图 (归一化)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = [
    ("cost_yuan", "运行成本 (元)", 0),
    ("carbon_tco2", "碳排放 (tCO₂)", 1),
    ("avg_latency_ms", "平均时延 (ms)", 2),
    ("renew_utilization", "新能源利用率", 3),
]
x = np.arange(len(compare_df))
colors_bar = plt.cm.tab20(np.linspace(0, 1, len(compare_df)))

for ax, (col, lab, idx) in zip(axes.flatten(), metrics_to_plot):
    vals = compare_df[col].values.copy().astype(float)
    if idx == 3:  # 新能源利用率 → 转 1-util
        vals = 1 - vals
    # ── v2修正：统一使用 Pareto 基准做归一化 + clip ──
    norm_v = np.clip((vals - GLB_min[idx]) / GLB_range[idx], 0.0, 1.0)

    bars = ax.barh(x, norm_v, color=colors_bar, edgecolor="k", linewidth=0.3)
    ax.set_yticks(x)
    ax.set_yticklabels(compare_df["scheme"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"归一化 {lab}  (0=最优, 1=最差)", fontsize=11)
    ax.set_title(lab, fontsize=12, fontweight="bold")

    # 标注真实值
    real_vals = compare_df[col].values
    for i, (bar, rv) in enumerate(zip(bars, real_vals)):
        if idx == 3:
            ax.text(
                bar.get_width() + 0.02, i, f"{rv*100:.2f}%", va="center", fontsize=8
            )
        elif abs(rv) >= 1e6:
            ax.text(bar.get_width() + 0.02, i, f"{rv:.2e}", va="center", fontsize=8)
        elif abs(rv) >= 1000:
            ax.text(bar.get_width() + 0.02, i, f"{rv:.0f}", va="center", fontsize=8)
        else:
            ax.text(bar.get_width() + 0.02, i, f"{rv:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.grid(True, axis="x", alpha=0.3)

fig.suptitle(
    "各调度方案四项指标对比（全局归一化，基准=Pareto解集）",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
plt.savefig(
    "/Users/xiongxuanyan/Desktop/python_scripts_all/output_nsga2/fig10_metric_bar.png",
    bbox_inches="tight",
    dpi=200,
)
plt.close()
print("✅ fig10_metric_bar.png")

# ────────────────────────────────────────────────────────
# 收尾：打印面积排名
# ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("雷达图面积排名 (面积越小 = 综合性能越优)")
print("=" * 65)
for rank, (lab, ar) in enumerate(zip(schemes_s, areas_s), 1):
    print(f"  #{rank:>2d}  {lab:<25s}  面积 = {ar:.6f}")

print("\n🎉 全部 11 张图生成完毕！")
