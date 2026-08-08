import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
import numpy as np
import os

# ============ 字体设置 ============
# 直接用字体文件路径，避免字体名查找失败
font_path = '/usr/share/fonts/truetype/noto/NotoSansSC-Light.ttf'
if not os.path.exists(font_path):
    # 备选路径
    for p in [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/nix/store/smgbvz9sqcikc18vf3i9ry2p5zlnb7h5-noto-fonts-cjk-sans-2.004/share/fonts/opentype/noto-cjk/NotoSansCJK-VF.otf.ttc',
    ]:
        if os.path.exists(p):
            font_path = p
            break

fp = FontProperties(fname=font_path, size=10)
fp_bold = FontProperties(fname=font_path, size=10, weight='bold')

def get_fp(fontsize=10, bold=False):
    """返回指定大小的字体属性"""
    return FontProperties(fname=font_path, size=fontsize, weight='bold' if bold else 'normal')

fig, ax = plt.subplots(figsize=(16, 26))
ax.set_xlim(0, 16)
ax.set_ylim(0, 26)
ax.axis('off')
fig.patch.set_facecolor('white')

# ============ 颜色方案 ============
C_START    = '#43A047'
C_PROCESS  = '#1E88E5'
C_DECISION = '#FB8C00'
C_IO       = '#8E24AA'
C_LOOP     = '#E53935'
C_ARROW    = '#37474F'
C_TITLE    = '#1A237E'
C_BG       = '#FAFAFA'

# 浅色版本（用于框背景）
C_START_L   = '#E8F5E9'
C_PROCESS_L = '#E3F2FD'
C_DECISION_L= '#FFF3E0'
C_IO_L      = '#F3E5F5'

# ============ 辅助函数 ============
def draw_box(ax, x, y, w, h, text, fc, ec='#263238', fontsize=10, bold=False,
             text_color='white', alpha=1.0, lw=1.8, pad=0.2):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={pad}",
                          facecolor=fc, edgecolor=ec,
                          linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontproperties=get_fp(fontsize, bold), color=text_color, zorder=4,
            linespacing=1.45)

def draw_stadium(ax, x, y, w, h, text, color, fontsize=12, text_color='white'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.35",
                          facecolor=color, edgecolor='#1B5E20',
                          linewidth=2.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontproperties=get_fp(fontsize, True), color=text_color, zorder=4)

def draw_diamond(ax, x, y, w, h, text, color, fontsize=10, text_color='white'):
    diamond = plt.Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)],
                           facecolor=color, edgecolor='#E65100', linewidth=2, zorder=3)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center',
            fontproperties=get_fp(fontsize, True), color=text_color, zorder=4,
            linespacing=1.3)

def draw_parallelogram(ax, x, y, w, h, text, color, fontsize=9, text_color='white'):
    offset = 0.35
    para = plt.Polygon([(x - w/2 + offset, y + h/2), (x + w/2 + offset, y + h/2),
                         (x + w/2 - offset, y - h/2), (x - w/2 - offset, y - h/2)],
                        facecolor=color, edgecolor='#4A148C', linewidth=1.8, zorder=3)
    ax.add_patch(para)
    ax.text(x, y, text, ha='center', va='center',
            fontproperties=get_fp(fontsize, True), color=text_color, zorder=4,
            linespacing=1.35)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=2)

def arrow_label(ax, x1, y1, x2, y2, label, color='#B71C1C', offset_x=0.2, offset_y=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.0), zorder=2)
    mid_x = (x1 + x2) / 2 + offset_x
    mid_y = (y1 + y2) / 2 + offset_y
    ax.text(mid_x, mid_y, label,
            fontproperties=get_fp(9, True), color=color,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF9C4',
                      edgecolor='#F9A825', alpha=0.95), zorder=5)

def side_note(ax, x, y, text, fontsize=8.5, color='#37474F'):
    ax.text(x, y, text,
            fontproperties=get_fp(fontsize, False), color=color,
            ha='left', va='center', style='italic', zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ECEFF1',
                      edgecolor='#90A4AE', alpha=0.9))

# ============ 标题 ============
ax.text(8, 25.3, 'NSGA-II 求解碳感知任务调度 — 算法流程图',
        ha='center', va='center',
        fontproperties=get_fp(18, True), color=C_TITLE)
ax.plot([1.5, 14.5], [25.0, 25.0], color=C_TITLE, linewidth=2.5, zorder=2)

# ============ 主轴位置 ============
cx = 7.5

# --- 1. 开始 ---
y = 24.2
draw_stadium(ax, cx, y, 2.6, 0.8, '开  始', C_START, fontsize=14)

# --- 2. 输入数据 ---
y_prev = y
y = 22.8
draw_parallelogram(ax, cx, y, 5.5, 1.0,
                   '输入：任务集合 $\\mathcal{I}$、区域参数\n电力参数、网络时延矩阵',
                   C_IO, fontsize=10)
arrow(ax, cx, y_prev - 0.4, cx, y + 0.5)

# --- 3. 预处理：候选集生成 ---
y_prev = y
y = 21.2
draw_box(ax, cx, y, 6.2, 1.4,
          '预处理：为每个任务生成候选集\n'
          '$\\mathcal{R}_i$（满足时延约束的区域集）\n'
          '$\\mathcal{S}_i$（满足完成时限的开工时刻集）',
          C_PROCESS, fontsize=10, bold=False, text_color='white')
arrow(ax, cx, y_prev - 0.5, cx, y + 0.7)
side_note(ax, cx + 3.8, y, '约束预剪枝\n缩小搜索空间', fontsize=8.5)

# --- 4. 初始化种群 ---
y_prev = y
y = 19.2
draw_box(ax, cx, y, 6.8, 1.6,
          '初始化种群 $\\mathcal{P}_0$（规模 $N=80$）\n'
          '① 贪心启发式个体（8组权重 × 扰动 ≈ 40个）\n'
          '② 随机个体（均匀采样 ≈ 40个）',
          C_PROCESS, fontsize=10, bold=False, text_color='white')
arrow(ax, cx, y_prev - 0.7, cx, y + 0.8)
side_note(ax, cx + 4.2, y, '高质量种子 + 随机探索\n平衡开发与探索', fontsize=8.5)

# --- 5. 非支配排序 + 拥挤度 ---
y_prev = y
y = 17.4
draw_box(ax, cx, y, 6.2, 1.1,
          '非支配排序 + 拥挤度距离计算\n'
          '→ 划分等级 $F_1, F_2, \\dots$\n'
          '→ 计算各层拥挤度距离',
          C_PROCESS, fontsize=10, bold=False, text_color='white')
arrow(ax, cx, y_prev - 0.8, cx, y + 0.55)

# ============ 循环区域标记 ============
loop_x = 1.5
loop_top = 16.6
loop_bot = 5.0
# 背景色
loop_bg = FancyBboxPatch((loop_x - 0.3, loop_bot), 1.6, loop_top - loop_bot,
                           boxstyle="round,pad=0.15",
                           facecolor='#FFEBEE', edgecolor=C_LOOP,
                           linewidth=2, linestyle='--', alpha=0.3, zorder=0)
ax.add_patch(loop_bg)
ax.text(loop_x + 0.5, (loop_top + loop_bot)/2, '$gen = 1 \\to 300$',
        fontproperties=get_fp(12, True), color=C_LOOP,
        ha='center', va='center', rotation=90, zorder=5)

# --- 6. 锦标赛选择 ---
y_prev = y
y = 15.5
draw_box(ax, cx, y, 5.8, 1.0,
          '锦标赛选择（size = 2）\n'
          '按非支配等级优先 → 同等级按拥挤度距离',
          '#1976D2', fontsize=10, text_color='white')
arrow(ax, cx, y_prev - 0.55, cx, y + 0.5)

# --- 7. SBX 交叉 ---
y_prev = y
y = 14.0
draw_box(ax, cx, y, 5.8, 1.0,
          'SBX 模拟二进制交叉（$p_c = 0.9,\\ \\eta_c = 20$）\n'
          '父代基因重组 → 产生子代种群 $\\mathcal{Q}_{gen}$',
          '#1565C0', fontsize=10, text_color='white')
arrow(ax, cx, y_prev - 0.5, cx, y + 0.5)

# --- 8. 多项式变异 ---
y_prev = y
y = 12.5
draw_box(ax, cx, y, 5.8, 1.0,
          '多项式变异（$p_m = 1/|\\mathcal{I}|,\\ \\eta_m = 20$）\n'
          '随机重置任务基因位 → 注入多样性',
          '#0D47A1', fontsize=10, text_color='white')
arrow(ax, cx, y_prev - 0.5, cx, y + 0.5)

# --- 9. 合并种群 ---
y_prev = y
y = 11.1
draw_box(ax, cx, y, 5.8, 0.9,
          '合并：$\\mathcal{R}_{gen} = \\mathcal{P}_{gen-1} \\cup \\mathcal{Q}_{gen}$\n'
          '（规模 $2N = 160$）',
          C_PROCESS, fontsize=10, text_color='white')
arrow(ax, cx, y_prev - 0.5, cx, y + 0.45)

# --- 10. 再次非支配排序 + 精英选择 ---
y_prev = y
y = 9.7
draw_box(ax, cx, y, 5.8, 1.0,
          '对 $\\mathcal{R}_{gen}$ 非支配排序\n'
          '逐层填充至 $N$ 个精英个体\n'
          '最后一层按拥挤度距离降序选择',
          C_PROCESS, fontsize=10, text_color='white')
arrow(ax, cx, y_prev - 0.45, cx, y + 0.5)

# --- 11. 判断：是否达到最大代数 ---
y_prev = y
y = 8.0
draw_diamond(ax, cx, y, 3.6, 1.6,
              '$gen \\geq 300$ ?',
              C_DECISION, fontsize=12)
arrow(ax, cx, y_prev - 0.5, cx, y + 0.8)

# 否 → 回到选择（左侧回路）
loop_back_x = 3.5
# 竖线
ax.plot([loop_back_x, loop_back_x], [y, 15.5], color=C_LOOP, lw=2.5, zorder=1)
# 箭头回到选择框
ax.annotate('', xy=(cx - 2.9, 15.5), xytext=(loop_back_x, 15.5),
            arrowprops=dict(arrowstyle='->', color=C_LOOP, lw=2.5), zorder=2)
# 回到选择框顶部
ax.annotate('', xy=(cx - 2.9, 16.0), xytext=(cx - 2.9, 15.5),
            arrowprops=dict(arrowstyle='->', color=C_LOOP, lw=2.5), zorder=2)
arrow_label(ax, loop_back_x, y, loop_back_x, 15.5, '否', color='#C62828', offset_x=-0.25, offset_y=0)

# 是 → 输出（右侧）
arrow_label(ax, cx + 1.8, y, 11.5, y, '是', color='#2E7D32', offset_x=0.2, offset_y=0)

# --- 12. 输出 Pareto 解集 ---
y_out = 6.8
draw_parallelogram(ax, 11.5, y_out, 4.2, 1.0,
                   '输出 Pareto 最优解集\n$\\mathcal{P}_{300}$（80个非支配解）',
                   C_IO, fontsize=10)
arrow(ax, 11.5, y, 11.5, y_out + 0.5)

# --- 13. 结束 ---
y_end = 5.3
draw_stadium(ax, 11.5, y_end, 2.6, 0.8, '结  束', C_START, fontsize=14)
arrow(ax, 11.5, y_out - 0.5, 11.5, y_end + 0.4)

# ============ 右侧图例和说明栏 ============
rx = 14.5
legend_bg = FancyBboxPatch((13.2, 5.0), 2.8, 19.5,
                            boxstyle="round,pad=0.2",
                            facecolor='#FAFAFA', edgecolor='#B0BEC5',
                            linewidth=1.2, zorder=1, alpha=0.8)
ax.add_patch(legend_bg)

ax.text(rx, 23.8, '图  例', ha='center',
        fontproperties=get_fp(12, True), color='#263238')

legend_items = [
    (C_START,   '开始 / 结束'),
    (C_PROCESS, '处理步骤'),
    (C_DECISION,'判断条件'),
    (C_IO,      '输入 / 输出'),
    (C_LOOP,    '循环体'),
]

for i, (color, label) in enumerate(legend_items):
    ly = 23.0 - i * 0.85
    box = FancyBboxPatch((rx - 0.6, ly - 0.22), 1.2, 0.44,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='#263238', linewidth=1, zorder=3)
    ax.add_patch(box)
    ax.text(rx + 0.85, ly, label, ha='left', va='center',
            fontproperties=get_fp(9.5, False), color='#263238')

# 算法要点
ax.text(rx, 18.5, '核心要点', ha='center',
        fontproperties=get_fp(11, True), color='#1A237E')

key_points = [
    '• 候选集预剪枝\n  缩小搜索空间',
    '• 混合初始化\n  贪心种子 + 随机探索',
    '• 精英保留策略\n  父子代合并选优',
    '• 拥挤度维持\n  保证解的分布均匀',
    '• 罚函数处理\n  约束违反 → 目标劣化',
    '• 非支配排序\n  多层前沿逐层筛选',
]
for i, text in enumerate(key_points):
    ky = 17.6 - i * 1.35
    ax.text(rx, ky, text, ha='center', va='center',
            fontproperties=get_fp(8.5, False), color='#37474F', style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8EAF6',
                      edgecolor='#7986CB', alpha=0.9))

# ============ 底部标注 ============
ax.text(8, 4.0, '图：NSGA-II 碳感知任务调度算法流程图',
        ha='center', fontproperties=get_fp(10, False), color='#757575',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#E0E0E0'))

plt.tight_layout()
output_path = '/data/workspace/fig_nsga2_flowchart.png'
plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"流程图已保存: {output_path}")
