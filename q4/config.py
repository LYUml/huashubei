from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q4 = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data",
    ROOT / "task_c" / "附件数据",
]
OUT = Q4 / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

REGIONS = [f"Region{x}" for x in "ABCDEF"]
TASK_TYPES = ["AITraining", "BatchInference", "RealTimeInference"]

ARRIVAL_END = 2399
EXECUTION_END = 2406  # half-open: no task activity at hour 2406
POWER_HOURS = EXECUTION_END + 1  # 0..2406 inclusive => 2407 hours

SEED = 20260809
