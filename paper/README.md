# 竞赛论文

主文件：`main.tex`（XeLaTeX + `ctex`）。

```bash
cd paper
xelatex main.tex
xelatex main.tex   # 更新交叉引用
```

- 插图目录：`figures/`（`\graphicspath{{figures/}}`）
- 历史草稿：`archive/`

缺图时编译会报错；当前需本地补齐的文件见仓库根 `README.md`。
