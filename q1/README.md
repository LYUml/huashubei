# Q1 运行说明

## 一键复现

在项目根目录运行：

```powershell
& '.\q1\.venv\Scripts\python.exe' '.\q1\run_q1.py'
```

如果需要重新创建环境：

```powershell
python -m venv .\q1\.venv
& '.\q1\.venv\Scripts\python.exe' -m pip install -r '.\q1\requirements.txt'
```

## 主要文件

- `run_q1.py`：完整的数据审计、统计、预测、carry-in、贪心、MILP、验证与绘图流水线；
- `q1flow.md`：整体技术路径；
- `q1tex.md`：根据实际运行结果生成的简版正文；
- `outputs/tables/`：预测、调度和独立校验结果；
- `outputs/figures/`：统计、预测、甘特图与GPU利用率图。

## 求解说明

- MILP求解器为开源HiGHS；
- 默认时间上限180秒，目标MIP Gap为1%；
- 若达到时间上限但已有可行解，脚本保留最好可行解并记录实际MIP Gap；
- 所有硬约束由独立Validator重新计算。
