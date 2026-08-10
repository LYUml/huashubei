# 第三问：面向算电协同的多目标储能调度优化

本文件夹包含第三问的完整可复现交付：优化代码、可视化代码、计算结果、图表和 LaTeX 论文。

## 主要结论

- 建立了以购售电成本、碳排放、区域峰值净购电功率和净购电波动为目标的线性规划模型。
- 等权折中方案相对严格可行的无储能基准：购售电成本改善 `9.27%`，碳排放降低 `100%`，区域峰值净购电功率降低 `100%`，区域净购电标准差平均降低 `97.21%`。
- A、B、C 区域因可再生能源始终覆盖设施负荷且不允许外送，最优方案不调用储能；系统收益主要来自允许外送的 D、E、F 区域。
- 赛题附件中的参考策略存在期末 SOC 不满足和同一时段购售电并存等问题，因此正文采用严格可行的“无储能反事实”作为主基准，并将附件策略仅保留作数据核验参考。

金额为负表示净售电收入超过购电支出；“成本改善”表示该净收益进一步增加。

## 文件结构

- `src/q3_optimize.py`：读取附件数据、构建并求解四类单目标锚点和等权多目标折中方案。
- `src/q3_visualize.py`：根据结果文件生成 6 幅论文图表。
- `data/`：运行所需的三份 Excel 数据副本。
- `results/`：系统指标、区域指标、可行性校验、收益矩阵和逐时调度结果。
- `figures/`：6 幅 PNG 图表。
- `paper/q3_paper.tex`：可独立编译的中文 LaTeX 正文。
- `output/pdf/q3_paper.pdf`：最终论文 PDF。

## 一键复现顺序

在本文件夹中运行：

```bash
/Users/xiongxuanyan/anaconda3/bin/python3 src/q3_optimize.py
MPLCONFIGDIR=tmp/mpl /Users/xiongxuanyan/anaconda3/bin/python3 src/q3_visualize.py
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=../output/pdf q3_paper.tex
```

若在其他电脑运行，可将上面的 Python 路径替换为已安装 `requirements.txt` 中依赖的 Python。LaTeX 正文使用 macOS 自带中文字体；其他操作系统需在导言区替换对应字体设置。

## 核验说明

- 优化结果逐区域检查了功率平衡、SOC 递推、期末 SOC 和设备功率边界。
- 等权折中方案的最大功率平衡残差约为 `1.42e-13 MW`，最大 SOC 递推残差约为 `2.84e-13 MWh`。
- 结果中不存在同时充放电或同时购售电。
- 最终 PDF 共 9 页，已完成逐页渲染检查。
