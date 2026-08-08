# Q2 核心文件说明

> 2026-08-09 清理：删除全部开发期脚手架（gen_data.py 假数据生成器、scheduler 原型、run_* 批量脚本等），仅保留正式链路三件套。

## 正式运行链路（按顺序）

```bash
# 1. 数据预处理：读取官方附件数据 → 输出 preprocessed_data.pkl
python q2_data_prep.py

# 2. NSGA-II 求解：读 preprocessed_data.pkl → 输出 Pareto 前沿、方案表
python q2_nsga2_v2.py

# 3. 可视化：读求解结果 → 生成论文配图 fig1~fig10
python q2_nsga2_visualize.py
```

## 关键依赖

```
pymoo >= 0.6.0    # NSGA-II 核心库
pandas, numpy       # 数据处理
matplotlib         # 绘图
scipy              # 平滑处理
scikit-learn       # 归一化（MinMaxScaler）
plotly             # 桑基图（可选）
```

## 注意事项（评审待修 P0）

- 数据接口当前为 `/data/workspace/data`（Linux 容器路径），本地运行需改为官方附件路径
- 成本/碳排放目标未扣除新能源消纳（应为 `(Load−RenewUsed)×Price`），见评审结论
- 设施功率约束缺失、IT 功率未含 NonAI 负荷、约束为软罚函数（n_constr=0）
