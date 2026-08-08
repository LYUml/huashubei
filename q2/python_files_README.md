# Python 文件清单与说明

## 核心文件（论文直接依赖）

| 文件 | 作用 | 被谁调用 |
|------|------|----------|
| `q2_data_prep.py` | 数据预处理：读取所有xlsx，构建GPU需求序列、合并电力参数、构建时延矩阵、功率映射、区域信息，输出 `preprocessed_data.pkl` | `q2_nsga2.py`, `q2_nsga2_v2.py` |
| `q2_nsga2.py` | NSGA-II 主求解器（基础版）：种群50，200代，随机初始化 | 独立运行 |
| `q2_nsga2_v2.py` | NSGA-II 主求解器（改进版）：贪心种子初始化 + 种群80 + 300代 + 分窗优化 | 独立运行 |
| `q2_nsga2_visualize.py` | 结果可视化：生成 fig1~fig10 共10张论文配图 | 依赖 `q2_nsga2_result.pkl` |

## 辅助文件（备选/调试用）

| 文件 | 作用 |
|------|------|
| `q2_predict.py` | 问题一的工作量预测（Tweedie-GBDT） |
| `q2_schedule.py` | 加权贪心调度（生成基线方案） |
| `q2_visualize.py` | 问题一结果可视化 |
| `scheduler.py` / `scheduler_v2.py` | 早期调度器原型 |
| `run_fast.py` / `run_fast2.py` | 快速测试脚本 |
| `run_balanced.py` | 均衡权重贪心测试 |
| `run_batch1.py` / `run_batch2.py` | 批量实验脚本 |
| `gen_data.py` | 生成模拟数据 |
| `test_scheduler.py` | 调度器单元测试 |
| `check_data.py` | 数据完整性检查 |
| `debug_col.py` | 列名调试工具 |
| `fig9_sankey.py` | 桑基图独立脚本 |
| `fig_algorithm_flowchart.py` | 算法流程图绘制 |
| `fig_nsga2_flowchart.py` | NSGA-II流程图绘制 |
| `task_data_full.py` | 任务数据汇总 |
| `generate_data.py` | 数据生成占位文件（空） |

## 推荐运行顺序

```bash
# 1. 数据预处理（必须先运行）
python q2_data_prep.py

# 2. NSGA-II 求解（二选一，推荐v2）
python q2_nsga2.py        # 基础版
python q2_nsga2_v2.py    # 改进版（贪心种子+更大种群）

# 3. 可视化（生成10张图）
python q2_nsga2_visualize.py
```

## 关键依赖

```
pymoo >= 0.6.0    # NSGA-II 核心库
pandas, numpy       # 数据处理
matplotlib         # 绘图
scipy              # 平滑处理
scikit-learn       # 归一化（MinMaxScaler）
plotly             # 桑基图（可选，缺失则跳过fig9）
```
