# 第三问：面向算电协同的多目标储能调度优化

本目录包含第三问的可复现交付：优化代码、可视化、计算结果与图表。

## 主要结论

- 建立了以购售电成本、碳排放、区域峰值净购电功率和净购电波动为目标的线性规划模型。
- 等权折中方案相对严格可行的无储能基准：购售电成本改善 `9.27%`，碳排放降低 `100%`，区域峰值净购电功率降低 `100%`，区域净购电标准差平均降低 `97.21%`。
- A、B、C 区域因可再生能源始终覆盖设施负荷且不允许外送，最优方案不调用储能；系统收益主要来自允许外送的 D、E、F 区域。
- 赛题附件中的参考策略存在期末 SOC 不满足和同一时段购售电并存等问题，因此正文采用严格可行的“无储能反事实”作为主基准，并将附件策略仅保留作数据核验参考。

金额为负表示净售电收入超过购电支出；“成本改善”表示该净收益进一步增加。

## 文件结构

```text
q3/
├── src/q3_optimize.py      # 四类单目标锚点 + 等权折中
├── src/q3_visualize.py     # 生成 fig01–fig06
├── data/                   # 本问 Excel 副本（亦可改用仓库根 data/）
├── results/                # 系统/区域指标、可行性校验、逐时调度
├── figures/                # 6 幅 PNG
├── requirements.txt
└── README.md
```

论文插图已汇总到仓库 `paper/figures/`；完整竞赛正文见 `paper/main.tex`。

## 依赖

```bash
# 在仓库根目录
python3 -m venv q3/.venv
q3/.venv/bin/pip install -r q3/requirements.txt
```

## 一键复现

```bash
# 使用本目录 data/ 副本（默认）
q3/.venv/bin/python q3/src/q3_optimize.py

# 或使用仓库根目录附件
q3/.venv/bin/python q3/src/q3_optimize.py --data-dir data

MPLCONFIGDIR=/tmp/mpl q3/.venv/bin/python q3/src/q3_visualize.py
```

Windows（PowerShell）示例：

```powershell
python -m venv q3\.venv
q3\.venv\Scripts\pip install -r q3\requirements.txt
q3\.venv\Scripts\python q3\src\q3_optimize.py
$env:MPLCONFIGDIR='tmp\mpl'; q3\.venv\Scripts\python q3\src\q3_visualize.py
```

## 核验说明

- 优化结果逐区域检查了功率平衡、SOC 递推、期末 SOC 和设备功率边界。
- 等权折中方案的最大功率平衡残差约为 `1.42e-13 MW`，最大 SOC 递推残差约为 `2.84e-13 MWh`。
- 结果中不存在同时充放电或同时购售电。
