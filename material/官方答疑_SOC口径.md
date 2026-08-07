# 官方答疑记录：C题数据错误回复（SOC 口径）

> 来源：华数杯组委会官方回复（2026-08-07 于官方平台刷到，截图 OCR 转写）
> 状态：已确认，作为 Q3/Q4 建模与论文口径依据

## 回复原文（转写）

经核查，RegionE 第 0 小时储能状态确实存在局部数值差异。按照附件给出的 SOC 递推口径：

SOC(0) = InitialSOC + ChargePower × ChargeEfficiency − DischargePower / DischargeEfficiency

RegionE 的 InitialSOC 为 370 MWh，ChargePower 为 116.3126 MW，ChargeEfficiency 为 0.94，DischargePower 为 0，因此：

SOC(0) = 370 + 116.3126 × 0.94 = 479.333844 MWh

而数据表中给出的 SOC_MWh 为 478.3339 MWh，两者相差约 1 MWh。该处可视为基准运行状态数据中的局部录入误差或口径残差。

参赛队伍在建模与优化计算时，应**优先依据附件说明中的 SOC 递推公式、充放电效率、SOC 上下限和终端 SOC 约束进行计算**。对于附件中基准状态数据与递推公式存在小范围不一致的情况，可以在论文中说明处理口径，例如：

- "以储能递推公式重新计算 SOC 状态"
- "对异常基准值进行一致性修正"

## 对我们的影响与处理口径

1. **Q1 / Q2：无影响**（不涉及储能/SOC）。
2. **Q3 / Q4**：
   - 初始 SOC 用 `storage_information.xlsx` 的 `InitialSOC_MWh`（RegionE = 370 MWh），自行按递推公式计算，不依赖 region_time_data 的 `SOC_MWh` 基准列作初始值或校验基准。
   - SOC 递推公式（与官方一致）：`SOC(t) = SOC(t−1) + ηc·ChargePower(t) − DischargePower(t)/ηd`
   - 终端约束：`SOC(2406) ≥ InitialSOC_MWh`。
   - 论文 Q3 建议写明："由于附件基准 SOC_MWh 与递推公式在 RegionE 第 0 小时存在约 1 MWh 的口径残差，本文以储能递推公式重新计算 SOC 状态。"
