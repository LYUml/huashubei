import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============ 全局设置 ============
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 22))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')

# ============ 颜色方案 ============
C_START   = '#4CAF50'   # 绿色 - 开始/结束
C_PROCESS = '#2196F3'   # 蓝色 - 处理步骤
C_DECISION= '#FF9800'   # 橙色 - 判断
C_DATA    = '#9C27B0'   # 紫色 - 数据/输入
C_OUTPUT  = '#F44336'   # 红色 - 输出
C_LOOP    = '#00BCD4'   # 青色 - 循环
C_TEXT    = '#FFFFFF'    # 白色文字
C_BG      = '#F5F5F5'   # 背景色

fig.patch.set_facecolor('#FAFAFA')

# ============ 辅助函数 ============
def draw_rounded_box(ax, x, y, w, h, text, color, fontsize=10, text_color='white', alpha=1.0, bold=False):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='#333333',
                          linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4,
            linespacing=1.5)

def draw_diamond(ax, x, y, w, h, text, color, fontsize=9):
    """绘制菱形判断框"""
    diamond = plt.Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)],
                           facecolor=color, edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color='white', fontweight='bold', zorder=4, linespacing=1.4)

def draw_parallelogram(ax, x, y, w, h, text, color, fontsize=9):
    """绘制平行四边形（数据输入/输出）"""
    offset = 0.3
    para = plt.Polygon([(x - w/2 + offset, y + h/2), (x + w/2 + offset, y + h/2),
                         (x + w/2 - offset, y - h/2), (x - w/2 - offset, y - h/2)],
                        facecolor=color, edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(para)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color='white', fontweight='bold', zorder=4, linespacing=1.4)

def draw_arrow(ax, x1, y1, x2, y2, color='#333333', lw=2, style='->', label=''):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, 
                              connectionstyle='arc3,rad=0'),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=8, color=color, fontstyle='italic', zorder=5)

def draw_curved_arrow(ax, x1, y1, x2, y2, color='#333333', lw=2, rad=0.3, label=''):
    """绘制弯曲箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                              connectionstyle=f'arc3,rad={rad}'),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.3, my, label, fontsize=8, color=color, fontstyle='italic', zorder=5)

# ============ 标题 ============
ax.text(7, 21.3, 'NSGA-II 碳感知任务调度算法流程图', ha='center', va='center',
        fontsize=18, fontweight='bold', color='#212121', zorder=5)
ax.text(7, 20.85, 'Algorithm 1: NSGA-II for Carbon-Aware Task Scheduling', ha='center', va='center',
        fontsize=11, color='#757575', fontstyle='italic', zorder=5)

# ============ 主流程（居中 x=7）============
cx = 7.0

# Step 1: 开始
draw_rounded_box(ax, cx, 20.2, 3.0, 0.7, '开  始', C_START, fontsize=13, bold=True)

# Step 2: 输入数据
draw_parallelogram(ax, cx, 19.1, 5.5, 0.8,
                   '输入：任务集合 $\\mathcal{I}$、区域参数、电力参数、时延矩阵', 
                   C_DATA, fontsize=9)

# Step 3: 候选集生成
draw_rounded_box(ax, cx, 17.9, 5.8, 0.9,
                   '为每个任务生成候选区域集 $\\mathcal{R}_i$\n和候选开工时刻集 $\\mathcal{S}_i$',
                   C_PROCESS, fontsize=10, bold=False)

# Step 4: 贪心种子
draw_rounded_box(ax, cx, 16.6, 5.8, 0.9,
                   '贪心启发式生成 8 组初始解\n（不同权重组合 $\\omega_1,\\omega_2,\\omega_3,\\omega_4$）',
                   C_PROCESS, fontsize=10)

# Step 5: 扰动变体
draw_rounded_box(ax, cx, 15.4, 5.8, 0.8,
                   '对贪心解施加扰动 → 产生约 40 个个体',
                   C_PROCESS, fontsize=10)

# Step 6: 随机填充
draw_rounded_box(ax, cx, 14.3, 5.8, 0.8,
                   '随机采样候选集 → 填充至种群规模 $N = 80$',
                   C_PROCESS, fontsize=10)

# Step 7: 非支配排序 + 拥挤度
draw_rounded_box(ax, cx, 13.1, 5.8, 0.9,
                   '对初始种群 $\\mathcal{P}_0$ 进行非支配排序\n计算拥挤度距离',
                   C_PROCESS, fontsize=10)

# Step 8: 进化循环入口 (菱形)
draw_diamond(ax, cx, 11.8, 3.5, 1.2,
              '$gen \\leq 300$ ?', C_DECISION, fontsize=11)

# ============ 循环体（向右偏移 x=10.5）============
rx = 10.8

# 循环体背景
loop_bg = FancyBboxPatch((8.3, 6.3), 4.8, 5.0,
                          boxstyle="round,pad=0.2",
                          facecolor='#E0F7FA', edgecolor=C_LOOP,
                          linewidth=2, linestyle='--', alpha=0.4, zorder=1)
ax.add_patch(loop_bg)
ax.text(10.7, 11.05, '进化循环体', ha='center', va='center',
        fontsize=10, color=C_LOOP, fontweight='bold', alpha=0.8, zorder=2)

# Step 9: 锦标赛选择
draw_rounded_box(ax, rx, 10.3, 3.8, 0.8,
                   '锦标赛选择（size=2）\n基于 rank + 拥挤度',
                   C_PROCESS, fontsize=9)

# Step 10: SBX交叉
draw_rounded_box(ax, rx, 9.1, 3.8, 0.8,
                   'SBX 交叉 ($p_c=0.9, \\eta_c=20$)\n生成子代 $\\mathcal{Q}_{gen}$',
                   C_PROCESS, fontsize=9)

# Step 11: 多项式变异
draw_rounded_box(ax, rx, 7.9, 3.8, 0.8,
                   '多项式变异 ($p_m=1/|\\mathcal{I}|, \\eta_m=20$)\n基因位重置为合法候选值',
                   C_PROCESS, fontsize=9)

# Step 12: 合并
draw_rounded_box(ax, rx, 6.8, 3.8, 0.7,
                   '合并：$\\mathcal{R} = \\mathcal{P} \\cup \\mathcal{Q}$ (规模 $2N$)',
                   '#673AB7', fontsize=9)

# 回到左侧 - 非支配排序
draw_rounded_box(ax, cx, 10.4, 5.8, 0.9,
                   '对 $\\mathcal{R}$ 进行非支配排序 → 分层 $F_1, F_2, \\dots$',
                   '#3F51B5', fontsize=10)

# 截断填充
draw_rounded_box(ax, cx, 9.2, 5.8, 0.9,
                   '逐层填充新种群 $\\mathcal{P}_{gen}$ 至 $N$ 个个体\n最后一层按拥挤度距离降序截断',
                   '#3F51B5', fontsize=10)

# gen++
draw_rounded_box(ax, cx, 8.0, 3.5, 0.7,
                   '$gen \\leftarrow gen + 1$',
                   C_LOOP, fontsize=11, bold=True)

# ============ 循环结束 → 输出 ============
# 否 → 输出
draw_parallelogram(ax, cx, 6.5, 5.5, 0.8,
                   '输出：Pareto 最优解集 $\\mathcal{P}_{300}$（80个非支配解）',
                   C_OUTPUT, fontsize=10)

# 结束
draw_rounded_box(ax, cx, 5.4, 3.0, 0.7, '结  束', C_START, fontsize=13, bold=True)

# ============ 代表方案选取（底部）============
draw_rounded_box(ax, cx, 4.2, 6.2, 0.9,
                   '从 Pareto 前沿选取代表方案：Knee 点 / 成本最优 / 碳排最优 / 时延最优 / 新能源最优',
                   '#E65100', fontsize=9)

# ============ 箭头连接 ============
# 开始 → 输入
draw_arrow(ax, cx, 19.85, cx, 19.55)

# 输入 → 候选集
draw_arrow(ax, cx, 18.7, cx, 18.4)

# 候选集 → 贪心
draw_arrow(ax, cx, 17.45, cx, 17.08)

# 贪心 → 扰动
draw_arrow(ax, cx, 16.15, cx, 15.85)

# 扰动 → 随机填充
draw_arrow(ax, cx, 14.95, cx, 14.75)

# 随机填充 → 非支配排序
draw_arrow(ax, cx, 13.9, cx, 13.6)

# 非支配排序 → 判断
draw_arrow(ax, cx, 12.65, cx, 12.45)

# 判断 YES → 锦标赛选择 (向右)
draw_arrow(ax, 8.75, 11.8, 8.9, 11.8, color=C_DECISION, lw=2.5)
ax.text(8.85, 12.05, '是', fontsize=11, color='#2E7D32', fontweight='bold')

# 锦标赛 → 交叉
draw_arrow(ax, rx, 9.9, rx, 9.55)

# 交叉 → 变异
draw_arrow(ax, rx, 8.7, rx, 8.35)

# 变异 → 合并
draw_arrow(ax, rx, 7.5, rx, 7.18)

# 合并 → 非支配排序 (回到左边)
draw_curved_arrow(ax, 8.9, 6.8, 8.9, 10.85, color='#3F51B5', lw=2, rad=-0.4)
ax.text(8.55, 8.85, '子代+父代', fontsize=8, color='#3F51B5', fontstyle='italic')

# 非支配排序 → 截断填充
draw_arrow(ax, cx, 9.95, cx, 9.7)

# 截断填充 → gen++
draw_arrow(ax, cx, 8.75, cx, 8.4)

# gen++ → 判断 (回到菱形)
draw_curved_arrow(ax, 5.25, 8.0, 5.25, 11.8, color=C_DECISION, lw=2, rad=-0.5)
ax.text(4.6, 9.9, '更新种群', fontsize=8, color=C_DECISION, fontstyle='italic')

# 判断 NO → 输出
draw_arrow(ax, cx, 11.2, cx, 6.95, color='#C62828', lw=2.5)
ax.text(7.35, 8.9, '否', fontsize=11, color='#C62828', fontweight='bold')

# 输出 → 结束
draw_arrow(ax, cx, 6.1, cx, 5.78)

# 结束 → 代表方案
draw_arrow(ax, cx, 5.05, cx, 4.7)

# ============ 图例 ============
legend_items = [
    (C_START, '开始 / 结束'),
    (C_DATA, '数据输入'),
    (C_PROCESS, '处理步骤'),
    (C_DECISION, '判断条件'),
    ('#3F51B5', '环境选择'),
    (C_LOOP, '循环更新'),
    (C_OUTPUT, '数据输出'),
    ('#E65100', '后处理分析'),
]

for i, (color, label) in enumerate(legend_items):
    lx = 0.5 + (i % 4) * 3.4
    ly = 3.0 - (i // 4) * 0.6
    rect = FancyBboxPatch((lx, ly - 0.2), 0.5, 0.4,
                           boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor='#333', linewidth=1, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.7, ly, label, fontsize=9, va='center', color='#333', zorder=4)

# ============ 保存 ============
plt.tight_layout()
plt.savefig('/data/workspace/fig_algorithm_flowchart.png', dpi=200, bbox_inches='tight',
            facecolor='#FAFAFA', edgecolor='none')
plt.savefig('/data/workspace/fig_algorithm_flowchart.pdf', bbox_inches='tight',
            facecolor='#FAFAFA', edgecolor='none')
print("流程图已保存！")
