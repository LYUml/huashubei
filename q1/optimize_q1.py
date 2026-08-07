from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import highspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


Q1 = Path(__file__).resolve().parent
ROOT = Q1.parent
TABLES = Q1 / "outputs" / "tables"
FIGURES = Q1 / "outputs" / "figures"
sys.path.insert(0, str(Q1))
import run_q1 as base  # noqa: E402


def build_model_arrays(tasks, data, candidates, carry_gpu, carry_power, wait_limit=None,
                       objective="peak", pruning=True):
    candidates = candidates.copy().reset_index(drop=True)
    raw_n = len(candidates)
    # Safe pruning only: remove exact duplicate decision columns. Cross-time/region
    # economic dominance is deliberately not used because capacity coupling can
    # make an apparently inferior column globally useful.
    if pruning:
        candidates = candidates.drop_duplicates(
            ["TaskID", "Region", "StartHour"], keep="first"
        ).reset_index(drop=True)
    task_lookup = tasks.set_index("TaskID")
    info = data.gpu_info.set_index("Region")
    gpu_cap = info.loc[base.REGIONS, "Available_GPU"].to_numpy(float)
    max_it = info.loc[base.REGIONS, "Max_IT_Power_MW"].to_numpy(float)
    pue = info.loc[base.REGIONS, "PUE"].to_numpy(float)
    max_fac = info.loc[base.REGIONS, "Max_Facility_Power_MW"].to_numpy(float)
    nonai = data.region_hour.pivot(index="Hour", columns="Region", values="NonAI_IT_Load_MW") \
        .loc[base.SCHEDULE_START:base.EXECUTION_END - 1, base.REGIONS].to_numpy(float).T
    n_cand, n_tasks = len(candidates), len(tasks)
    peak_var, n_var = n_cand, n_cand + 1

    wait = candidates["Wait"].to_numpy(float)
    migration = candidates["Migration"].to_numpy(float)
    latency = candidates["LatencyNorm"].to_numpy(float)
    tail = candidates["TailGPUh"].to_numpy(float)
    c = np.zeros(n_var)
    if objective == "wait":
        c[:n_cand] = wait / n_tasks
    elif objective == "peak":
        # Peak is primary; tiny tie-breakers make equal-peak solutions operationally cleaner.
        c[peak_var] = 1.0
        c[:n_cand] = (1e-5 * migration + 2e-6 * latency +
                      2e-6 * tail / max(tail.max(), 1.0)) / n_tasks
    else:
        raise KeyError(objective)

    ri, ci, vv, lo, up = [], [], [], [], []
    row = 0
    for _, group in candidates.groupby("TaskID", sort=False):
        for j in group.index:
            ri.append(row); ci.append(int(j)); vv.append(1.0)
        lo.append(1.0); up.append(1.0); row += 1

    for r, region in enumerate(base.REGIONS):
        rc = candidates[candidates["Region"] == region]
        for t_idx, hour in enumerate(base.HOURS):
            ge, pe = [], []
            for j, cand in rc.iterrows():
                task = task_lookup.loc[int(cand["TaskID"])]
                q = base.overlap_fraction(int(cand["StartHour"]),
                                          int(task["EstimatedDuration_min"]), int(hour))
                if q <= 0:
                    continue
                ge.append((int(j), float(task["GPU_Demand"] * q)))
                pe.append((int(j), float(task["GPU_Demand"] * task["PowerPerGPU"] * q)))
            for j, val in ge:
                ri.append(row); ci.append(j); vv.append(val)
            lo.append(-np.inf); up.append(float(gpu_cap[r] - carry_gpu[r, t_idx])); row += 1
            headroom = min(max_it[r], max_fac[r] / pue[r]) - nonai[r, t_idx] - carry_power[r, t_idx]
            for j, val in pe:
                ri.append(row); ci.append(j); vv.append(val)
            lo.append(-np.inf); up.append(float(headroom)); row += 1
            for j, val in ge:
                ri.append(row); ci.append(j); vv.append(val / gpu_cap[r])
            ri.append(row); ci.append(peak_var); vv.append(-1.0)
            lo.append(-np.inf); up.append(float(-carry_gpu[r, t_idx] / gpu_cap[r])); row += 1

    if wait_limit is not None:
        for j, val in enumerate(wait / n_tasks):
            if val:
                ri.append(row); ci.append(j); vv.append(float(val))
        lo.append(-np.inf); up.append(float(wait_limit)); row += 1

    A = sparse.coo_matrix((vv, (ri, ci)), shape=(row, n_var)).tocsr()
    return {
        "candidates": candidates, "raw_candidates": raw_n, "A": A,
        "lower": np.asarray(lo), "upper": np.asarray(up), "cost": c,
        "col_lower": np.zeros(n_var), "col_upper": np.r_[np.ones(n_cand), 1.0],
        "peak_var": peak_var,
    }


def mip_start_vector(model, schedule):
    candidates = model["candidates"]
    lookup = {(int(r.TaskID), str(r.Region), int(r.StartHour)): i
              for i, r in candidates.iterrows()}
    x = np.zeros(len(candidates) + 1)
    missing = 0
    for r in schedule.itertuples():
        key = (int(r.TaskID), str(r.ExecutionRegion), int(r.StartHour))
        if key in lookup:
            x[lookup[key]] = 1.0
        else:
            missing += 1
    x[-1] = float(schedule.attrs.get("peak", 1.0))
    return x, missing


def model_variant(common, tasks, objective="peak", wait_limit=None):
    """Reuse the expensive common resource matrix across all Pareto solves."""
    model = dict(common)
    cand = common["candidates"]
    c = np.zeros(len(cand) + 1)
    if objective == "wait":
        c[:len(cand)] = cand["Wait"].to_numpy(float) / len(tasks)
    else:
        migration = cand["Migration"].to_numpy(float)
        latency = cand["LatencyNorm"].to_numpy(float)
        tail = cand["TailGPUh"].to_numpy(float)
        c[-1] = 1.0
        c[:len(cand)] = (1e-5 * migration + 2e-6 * latency +
                         2e-6 * tail / max(tail.max(), 1.0)) / len(tasks)
    model["cost"] = c
    if wait_limit is not None:
        wr = sparse.csr_matrix((cand["Wait"].to_numpy(float) / len(tasks),
                                (np.zeros(len(cand), dtype=int), np.arange(len(cand)))),
                               shape=(1, len(cand) + 1))
        model["A"] = sparse.vstack([common["A"], wr], format="csr")
        model["lower"] = np.r_[common["lower"], -np.inf]
        model["upper"] = np.r_[common["upper"], float(wait_limit)]
    return model


def solve_highs(model, start_x=None, time_limit=60.0, gap=0.01):
    A = model["A"]
    lp = highspy.HighsLp()
    lp.num_col_ = A.shape[1]; lp.num_row_ = A.shape[0]
    lp.col_cost_ = model["cost"]
    lp.col_lower_ = model["col_lower"]; lp.col_upper_ = model["col_upper"]
    lp.row_lower_ = model["lower"]; lp.row_upper_ = model["upper"]
    lp.integrality_ = [highspy.HighsVarType.kInteger] * (A.shape[1] - 1) + [highspy.HighsVarType.kContinuous]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.num_col_ = A.shape[1]; lp.a_matrix_.num_row_ = A.shape[0]
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    h = highspy.Highs()
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(gap))
    h.setOptionValue("presolve", "on")
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    start_status = "not_used"
    if start_x is not None:
        idx = np.arange(len(start_x), dtype=np.int32)
        start_status = str(h.setSolution(len(idx), idx, np.asarray(start_x, dtype=float)))
    tic = time.time(); h.run(); elapsed = time.time() - tic
    sol = np.asarray(h.getSolution().col_value)
    info = h.getInfo()
    status = h.getModelStatus()
    result = {
        "model_status": str(h.modelStatusToString(status)),
        "objective": float(h.getObjectiveValue()),
        "mip_gap": float(info.mip_gap), "mip_node_count": int(info.mip_node_count),
        "dual_bound": float(info.mip_dual_bound), "solve_seconds": elapsed,
        "mip_start_status": start_status,
    }
    return sol, result


def chosen_schedule(model, x, tasks, data, carry_gpu, carry_power, name):
    cand = model["candidates"]
    chosen = cand.iloc[np.where(x[:len(cand)] > .5)[0]].copy()
    chosen = chosen.rename(columns={"Region": "ExecutionRegion"})
    lookup = tasks.set_index("TaskID")
    chosen["FinishHour"] = chosen.apply(
        lambda r: r.StartHour + lookup.loc[int(r.TaskID), "EstimatedDuration_min"] / 60, axis=1)
    schedule, gpu, power = base.reconstruct_schedule(tasks, chosen, data, carry_gpu, carry_power)
    metrics = base.validate_schedule(tasks, schedule, gpu, power, data, name)
    return schedule, metrics


def load_context():
    data = base.load_data()
    latency, eligible = base.latency_maps(data)
    carry_gpu, carry_power, carry_tasks = base.build_carry_in(data, eligible, latency)
    tasks = data.tasks[data.tasks["ArrivalHour"].between(2376, 2399)].copy()
    tasks["PowerPerGPU"] = tasks["TaskType"].map(data.power)
    candidates = base.build_milp_candidates(tasks, data, eligible, latency)
    local = pd.read_csv(TABLES / "schedule_local_first.csv")
    local.attrs["peak"] = float(json.loads((TABLES / "validation_local_first.json").read_text(encoding="utf-8"))["peak_gpu_utilization"])
    return data, tasks, candidates, carry_gpu, carry_power, carry_tasks, local


def run_experiments(time_limit=60.0):
    data, tasks, candidates, carry_gpu, carry_power, carry_tasks, local = load_context()
    rows, schedules = [], {}

    # Stage 1 of lexicographic optimization: minimum mean waiting time.
    common = build_model_arrays(tasks, data, candidates, carry_gpu, carry_power,
                                objective="peak", pruning=True)
    wait_model = model_variant(common, tasks, objective="wait")
    start_x, missing = mip_start_vector(wait_model, local)
    xw, iw = solve_highs(wait_model, start_x, time_limit=time_limit)
    sw, mw = chosen_schedule(wait_model, xw, tasks, data, carry_gpu, carry_power, "wait_stage1")
    w_star = mw["mean_wait_hour"]
    rows.append({"scenario": "wait_stage1", "wait_limit": np.nan, **iw, **mw,
                 "raw_candidates": wait_model["raw_candidates"],
                 "kept_candidates": len(wait_model["candidates"]), "start_missing": missing})

    scenarios = [("lexicographic_1.05", 1.05 * w_star),
                 ("epsilon_0.05", .05), ("epsilon_0.10", .10),
                 ("epsilon_0.20", .20), ("epsilon_0.40", .40)]
    for name, eps in scenarios:
        model = model_variant(common, tasks, objective="peak", wait_limit=eps)
        sx, missing = mip_start_vector(model, local)
        x, info = solve_highs(model, sx, time_limit=time_limit)
        sched, metrics = chosen_schedule(model, x, tasks, data, carry_gpu, carry_power, name)
        sched.to_csv(TABLES / f"schedule_{name}.csv", index=False, encoding="utf-8-sig")
        (TABLES / f"validation_{name}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"scenario": name, "wait_limit": eps, **info, **metrics,
                     "raw_candidates": model["raw_candidates"],
                     "kept_candidates": len(model["candidates"]), "start_missing": missing})
        schedules[name] = sched

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "optimization_experiments.csv", index=False, encoding="utf-8-sig")
    feasible = out[(out["scenario"] != "wait_stage1") & out["all_hard_constraints_pass"]].copy()
    feasible["ideal_distance"] = np.sqrt(
        (feasible["mean_wait_hour"] / feasible["mean_wait_hour"].max()) ** 2 +
        (feasible["peak_gpu_utilization"] / feasible["peak_gpu_utilization"].max()) ** 2)
    preferred_pool = feasible[feasible["mean_wait_hour"] <= .10 + 1e-9]
    preferred = preferred_pool.sort_values(["peak_gpu_utilization", "mean_wait_hour"]).iloc[0]
    recommendation = {
        "preferred_scenario": preferred["scenario"],
        "selection_rule": "minimum peak utilization among validated solutions with mean wait <= 0.10 h",
        "mean_wait_hour": float(preferred["mean_wait_hour"]),
        "peak_gpu_utilization": float(preferred["peak_gpu_utilization"]),
        "migration_rate": float(preferred["migration_rate"]),
        "mip_gap": float(preferred["mip_gap"]),
        "all_hard_constraints_pass": bool(preferred["all_hard_constraints_pass"]),
        "carry_in_tasks": len(carry_tasks),
    }
    (TABLES / "recommended_schedule.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")

    old = pd.read_csv(TABLES / "schedule_method_comparison.csv")
    plot = pd.concat([
        old[["name", "mean_wait_hour", "peak_gpu_utilization"]].rename(columns={"name": "scenario"}),
        feasible[["scenario", "mean_wait_hour", "peak_gpu_utilization"]]
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(plot.mean_wait_hour, plot.peak_gpu_utilization, s=65)
    for r in plot.itertuples():
        ax.annotate(r.scenario, (r.mean_wait_hour, r.peak_gpu_utilization),
                    xytext=(4, 5), textcoords="offset points", fontsize=8)
    pr = plot[plot.scenario == recommendation["preferred_scenario"]].iloc[0]
    ax.scatter([pr.mean_wait_hour], [pr.peak_gpu_utilization], marker="*", s=230,
               color="tab:red", label="recommended (wait <= 0.10 h)")
    ax.set_xlabel("Mean waiting time (hour)"); ax.set_ylabel("Peak GPU utilization")
    ax.set_title("Q1 service–peak trade-off under validated schedules")
    ax.grid(alpha=.25); ax.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "09_full_pareto.png", dpi=180); plt.close()
    return out, recommendation


def run_p2_ablation(time_limit=30.0):
    data, tasks, candidates, carry_gpu, carry_power, _, local = load_context()
    common = build_model_arrays(tasks, data, candidates, carry_gpu, carry_power,
                                objective="peak", pruning=True)
    model = model_variant(common, tasks, objective="peak", wait_limit=0.0)
    sx, missing = mip_start_vector(model, local)
    # A valid global lower bound follows from total occupied GPU-hours divided by
    # total region GPU-hour capacity; carry-in is included and only strengthens it.
    info = data.gpu_info.set_index("Region")
    total_capacity = float(info.loc[base.REGIONS, "Available_GPU"].sum() * len(base.HOURS))
    task_gpuh = float((tasks["GPU_Demand"] * tasks["EstimatedDuration_min"] / 60).sum())
    carry_gpuh = float(carry_gpu.sum())
    peak_lb = (task_gpuh + carry_gpuh) / total_capacity
    local_peak = float(local.attrs["peak"])
    rows = []
    for name, start, bounds in [
        ("no_mip_start", None, False),
        ("mip_start", sx, False),
        ("mip_start_plus_peak_bounds", sx, True),
    ]:
        variant = dict(model)
        variant["col_lower"] = model["col_lower"].copy()
        variant["col_upper"] = model["col_upper"].copy()
        if bounds:
            variant["col_lower"][-1] = peak_lb
            variant["col_upper"][-1] = local_peak
        x, solve = solve_highs(variant, start, time_limit=time_limit)
        sched, metrics = chosen_schedule(variant, x, tasks, data, carry_gpu, carry_power,
                                         f"ablation_{name}")
        rows.append({"experiment": name, "time_limit": time_limit,
                     "peak_lower_bound": peak_lb if bounds else 0.0,
                     "peak_upper_bound": local_peak if bounds else 1.0,
                     "start_missing": missing if start is not None else np.nan,
                     **solve, **metrics})
    rows.extend([
        {"experiment": "exact_duplicate_pruning", "time_limit": 0,
         "note": f"no effect: {common['raw_candidates']} -> {len(common['candidates'])} candidates"},
        {"experiment": "symmetry_breaking", "time_limit": 0,
         "note": "not applicable: zero fully identical task groups in final window"},
    ])
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "p2_solver_ablation.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    result, rec = run_experiments(60.0)
    print(result[["scenario", "mean_wait_hour", "peak_gpu_utilization", "mip_gap",
                  "mip_node_count", "all_hard_constraints_pass"]].to_string(index=False))
    print(json.dumps(rec, ensure_ascii=False, indent=2))
