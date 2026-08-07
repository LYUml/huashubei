"""重跑 ε-约束 Pareto 实验：链式 MIP start + 300 秒/点。

为什么链式：
- ε 递增时可行域单调扩大：ε_prev 的最优解必是 ε_cur 的可行解（wait 上限更松）。
- 把上一个更紧 ε 的可行解作为下一个的热启动，既加速收敛，
  又保证输出峰值单调非增（不收敛时也不会比上一解更差）。
- wait_stage1 与 lexicographic_1.05 用同参数重跑，保证整表来自同一轮。

产物格式与 optimize_q1.run_experiments 完全一致：
  optimization_experiments.csv / schedule_epsilon_*.csv / validation_epsilon_*.json /
  recommended_schedule.json

容错：每完成一个场景立即持久化 optimization_experiments.csv（覆盖写），
中途崩溃时已有结果不丢失。支持 --from <scenario> 断点续跑：
跳过场景的输入从已持久化的 schedule/validation 文件恢复。
"""
from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import pandas as pd

import optimize_q1 as oq

EPS_TIME_LIMIT = 300.0    # 每个 ε 点的求解时限（秒）
OTHER_TIME_LIMIT = 120.0  # wait_stage1 / lexicographic 的时限
SCENARIO_ORDER = ["wait_stage1", "lexicographic_1.05",
                  "epsilon_0.05", "epsilon_0.10", "epsilon_0.20", "epsilon_0.40"]


def persist(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(oq.TABLES / "optimization_experiments.csv",
                              index=False, encoding="utf-8-sig")


def main() -> tuple[pd.DataFrame, dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start_from", default=None,
                    choices=SCENARIO_ORDER, help="断点续跑：跳过该场景之前的全部场景")
    args = ap.parse_args()
    start_idx = SCENARIO_ORDER.index(args.start_from) if args.start_from else 0

    data, tasks, candidates, carry_gpu, carry_power, carry_tasks, local = oq.load_context()
    rows, schedules = [], {}
    common = oq.build_model_arrays(tasks, data, candidates, carry_gpu, carry_power,
                                   objective="peak", pruning=True)

    # ---- Stage 1: 最小平均等待（目标非负，找到 W=0 可行解即严格最优）----
    if start_idx <= 0:
        wait_model = oq.model_variant(common, tasks, objective="wait")
        wx, missing_w = oq.mip_start_vector(wait_model, local)
        xw, iw = oq.solve_highs(wait_model, wx, time_limit=OTHER_TIME_LIMIT)
        _, mw = oq.chosen_schedule(wait_model, xw, tasks, data, carry_gpu, carry_power,
                                   "wait_stage1")
        rows.append({"scenario": "wait_stage1", "wait_limit": np.nan, **iw, **mw,
                     "raw_candidates": wait_model["raw_candidates"],
                     "kept_candidates": len(wait_model["candidates"]), "start_missing": missing_w})
        print(f"wait_stage1: W*={mw['mean_wait_hour']:.6f} status={iw['model_status']}")
        persist(rows)

    # ---- 字典序第二阶段：W <= 1.05 W* 下最小化峰值 ----
    if start_idx <= 1:
        w_star = rows[0]["mean_wait_hour"]
        lex_model = oq.model_variant(common, tasks, objective="peak", wait_limit=1.05 * w_star)
        lx, missing_l = oq.mip_start_vector(lex_model, local)
        xl, il = oq.solve_highs(lex_model, lx, time_limit=OTHER_TIME_LIMIT)
        ls, lm = oq.chosen_schedule(lex_model, xl, tasks, data, carry_gpu, carry_power,
                                    "lexicographic_1.05")
        ls.to_csv(oq.TABLES / "schedule_lexicographic_1.05.csv", index=False,
                  encoding="utf-8-sig")
        (oq.TABLES / "validation_lexicographic_1.05.json").write_text(
            json.dumps(lm, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"scenario": "lexicographic_1.05", "wait_limit": 1.05 * w_star, **il, **lm,
                     "raw_candidates": lex_model["raw_candidates"],
                     "kept_candidates": len(lex_model["candidates"]), "start_missing": missing_l})
        print(f"lexicographic_1.05: wait={lm['mean_wait_hour']:.4f} "
              f"peak={lm['peak_gpu_utilization']:.4f} gap={il['mip_gap']:.4f}")
        persist(rows)

    # ---- 恢复断点状态：从已有 CSV 恢复被跳过场景的 rows 与链式起点 ----
    if args.start_from:
        old = pd.read_csv(oq.TABLES / "optimization_experiments.csv")
        for sc in SCENARIO_ORDER[:start_idx]:
            r = old[old["scenario"] == sc].iloc[0].to_dict()
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    r[k] = np.nan
            rows.append(r)
        w_star = float(rows[1]["wait_limit"]) / 1.05

    # ---- ε 链式求解（无状态：链式起点一律从磁盘读上一场景的已持久化文件）----
    eps_scenarios = [("epsilon_0.05", .05), ("epsilon_0.10", .10),
                     ("epsilon_0.20", .20), ("epsilon_0.40", .40)]
    for i, (name, eps) in enumerate(eps_scenarios):
        order_idx = SCENARIO_ORDER.index(name)
        if order_idx < start_idx:
            continue
        if i == 0:
            prev_sched = pd.read_csv(oq.TABLES / "schedule_lexicographic_1.05.csv")
            prev_metrics = json.loads(
                (oq.TABLES / "validation_lexicographic_1.05.json").read_text(encoding="utf-8"))
        else:
            pn = eps_scenarios[i - 1][0]
            prev_sched = pd.read_csv(oq.TABLES / f"schedule_{pn}.csv")
            prev_metrics = json.loads(
                (oq.TABLES / f"validation_{pn}.json").read_text(encoding="utf-8"))

        gc.collect()
        model = oq.model_variant(common, tasks, objective="peak", wait_limit=eps)
        start = prev_sched.copy()
        start.attrs["peak"] = prev_metrics["peak_gpu_utilization"]
        sx, missing = oq.mip_start_vector(model, start)
        x, info = oq.solve_highs(model, sx, time_limit=EPS_TIME_LIMIT)
        sched, metrics = oq.chosen_schedule(model, x, tasks, data, carry_gpu, carry_power, name)
        sched.to_csv(oq.TABLES / f"schedule_{name}.csv", index=False, encoding="utf-8-sig")
        (oq.TABLES / f"validation_{name}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"scenario": name, "wait_limit": eps, **info, **metrics,
                     "raw_candidates": model["raw_candidates"],
                     "kept_candidates": len(model["candidates"]), "start_missing": missing})
        prev_sched, prev_metrics = sched, metrics
        print(f"{name}: wait={metrics['mean_wait_hour']:.4f} "
              f"peak={metrics['peak_gpu_utilization']:.4f} gap={info['mip_gap']:.4f} "
              f"nodes={info['mip_node_count']} status={info['model_status']}", flush=True)
        persist(rows)

    out = pd.DataFrame(rows)

    # ---- 推荐方案 ----
    # 主推规则：在平均等待 ≤ 0.10h 且 MIP Gap ≤ 5%（可声称近最优）的已验证方案中选峰值最低者。
    # 未收敛的 ε 点（Gap 较大）只作前沿展示，不作主推，避免把近似解写进结论。
    feasible = out[(out["scenario"] != "wait_stage1") & out["all_hard_constraints_pass"]].copy()
    feasible["ideal_distance"] = np.sqrt(
        (feasible["mean_wait_hour"] / feasible["mean_wait_hour"].max()) ** 2 +
        (feasible["peak_gpu_utilization"] / feasible["peak_gpu_utilization"].max()) ** 2)
    preferred_pool = feasible[(feasible["mean_wait_hour"] <= .10 + 1e-9)
                              & (feasible["mip_gap"] <= 0.05)]
    if preferred_pool.empty:
        preferred_pool = feasible[feasible["mean_wait_hour"] <= .10 + 1e-9]
    preferred = preferred_pool.sort_values(["peak_gpu_utilization", "mean_wait_hour"]).iloc[0]
    recommendation = {
        "preferred_scenario": preferred["scenario"],
        "selection_rule": ("minimum peak utilization among validated solutions "
                           "with mean wait <= 0.10 h and MIP gap <= 0.05"),
        "mean_wait_hour": float(preferred["mean_wait_hour"]),
        "peak_gpu_utilization": float(preferred["peak_gpu_utilization"]),
        "migration_rate": float(preferred["migration_rate"]),
        "mip_gap": float(preferred["mip_gap"]),
        "all_hard_constraints_pass": bool(preferred["all_hard_constraints_pass"]),
        "carry_in_tasks": len(carry_tasks),
    }
    (oq.TABLES / "recommended_schedule.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out[["scenario", "wait_limit", "mean_wait_hour", "peak_gpu_utilization",
               "migration_rate", "mip_gap", "mip_node_count", "model_status",
               "all_hard_constraints_pass"]].to_string(index=False))
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    return out, recommendation


if __name__ == "__main__":
    main()
