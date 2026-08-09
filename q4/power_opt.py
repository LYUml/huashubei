from __future__ import annotations

import numpy as np

from config import POWER_HOURS, REGIONS
from data_loader import Q4Data


def _region_lp(
    data: Q4Data,
    region_idx: int,
    total_load: np.ndarray,
    available_re: np.ndarray,
    *,
    objective: str = "cost",
    carbon_budget: float | None = None,
    peak_cap: float | None = None,
    min_re_utilization: float | None = None,
    time_limit: float = 20.0,
    enforce_carbon: bool = False,
) -> dict:
    """
    Single-region LP (HiGHS) for storage + grid + renewable allocation.

    Official energy balance (attachment):
      GridPurchase + AvailableRE + Discharge
        = TotalLoad + Charge + GridSell + Curtailment

    objective:
      - "cost": minimize purchase cost - sell revenue
      - "carbon": minimize grid-purchase carbon
    """
    import highspy

    T = POWER_HOURS
    price = data.price[region_idx]
    sell_p = data.sell_price[region_idx]
    ci = data.carbon[region_idx]
    avail = np.asarray(available_re, dtype=float)
    load = np.asarray(total_load, dtype=float)

    max_imp = float(data.max_import[region_idx])
    max_exp = float(min(data.max_export[region_idx], data.sell_limit[region_idx]))
    max_ch = float(data.max_charge[region_idx])
    max_dch = float(data.max_discharge[region_idx])
    eta_c = float(data.eta_c[region_idx])
    eta_d = float(data.eta_d[region_idx])
    soc_min = float(data.min_soc[region_idx])
    soc_max = float(data.storage_cap[region_idx])
    soc0 = float(data.init_soc[region_idx])

    # 0 gp, 1 sell, 2 charge, 3 dch, 4 curt, 5 soc
    NV = 6
    n = T * NV

    def vid(t: int, k: int) -> int:
        return t * NV + k

    def _solve(use_carbon: bool, use_peak: bool, use_re: bool):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("presolve", "on")

        cost = np.zeros(n)
        if objective == "carbon":
            for t in range(T):
                cost[vid(t, 0)] = float(ci[t])
        else:
            for t in range(T):
                cost[vid(t, 0)] = float(price[t])
                cost[vid(t, 1)] = -float(sell_p[t])

        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.col_cost_ = cost.tolist()
        col_lower = np.zeros(n)
        col_upper = np.full(n, highspy.kHighsInf)
        for t in range(T):
            col_upper[vid(t, 0)] = max_imp
            col_upper[vid(t, 1)] = max_exp
            col_upper[vid(t, 2)] = max_ch
            col_upper[vid(t, 3)] = max_dch
            col_upper[vid(t, 4)] = float(max(avail[t], 0.0))
            col_lower[vid(t, 5)] = soc_min
            col_upper[vid(t, 5)] = soc_max
        lp.col_lower_ = col_lower.tolist()
        lp.col_upper_ = col_upper.tolist()

        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        row_lower: list[float] = []
        row_upper: list[float] = []

        def add_row(inds: list[int], vals: list[float], lo: float, up: float) -> None:
            indices.extend(inds)
            values.extend(vals)
            starts.append(len(indices))
            row_lower.append(lo)
            row_upper.append(up)

        for t in range(T):
            # gp - sell - charge + dch - curt = load - avail
            rhs = float(load[t] - avail[t])
            add_row(
                [vid(t, 0), vid(t, 1), vid(t, 2), vid(t, 3), vid(t, 4)],
                [1.0, -1.0, -1.0, 1.0, -1.0],
                rhs,
                rhs,
            )
            if t == 0:
                add_row(
                    [vid(t, 5), vid(t, 2), vid(t, 3)],
                    [1.0, -eta_c, 1.0 / eta_d],
                    soc0,
                    soc0,
                )
            else:
                add_row(
                    [vid(t, 5), vid(t - 1, 5), vid(t, 2), vid(t, 3)],
                    [1.0, -1.0, -eta_c, 1.0 / eta_d],
                    0.0,
                    0.0,
                )
            if use_peak and peak_cap is not None:
                add_row([vid(t, 0), vid(t, 1)], [1.0, -1.0], -highspy.kHighsInf, float(peak_cap))

        add_row([vid(T - 1, 5)], [1.0], soc0, highspy.kHighsInf)

        if use_carbon and carbon_budget is not None:
            inds = [vid(t, 0) for t in range(T)]
            vals = [float(ci[t]) for t in range(T)]
            add_row(inds, vals, -highspy.kHighsInf, float(carbon_budget))

        if use_re and min_re_utilization is not None:
            total_avail = float(avail.sum())
            if total_avail > 1e-9:
                inds = [vid(t, 4) for t in range(T)]
                vals = [1.0] * T
                add_row(
                    inds,
                    vals,
                    -highspy.kHighsInf,
                    float((1.0 - min_re_utilization) * total_avail),
                )

        lp.num_row_ = len(row_lower)
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.num_col_ = n
        lp.a_matrix_.num_row_ = len(row_lower)
        lp.a_matrix_.start_ = starts
        lp.a_matrix_.index_ = indices
        lp.a_matrix_.value_ = values

        h.passModel(lp)
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        sol = np.asarray(h.getSolution().col_value, dtype=float)
        return h, status, sol

    def _ok(status: str, sol: np.ndarray) -> bool:
        return sol.size == n and np.isfinite(sol).all() and status in ("Optimal", "Time limit reached")

    # Constraint ladder. Carbon is never dropped when enforce_carbon=True.
    attempts: list[tuple[bool, bool, bool]] = [
        (carbon_budget is not None, peak_cap is not None, min_re_utilization is not None),
        (carbon_budget is not None, peak_cap is not None, False),
        (carbon_budget is not None, False, False),
    ]
    if not enforce_carbon:
        attempts.append((False, False, False))

    h = status = sol = None
    used_carbon = False
    for use_c, use_p, use_r in attempts:
        if enforce_carbon and carbon_budget is not None and not use_c:
            continue
        h, status, sol = _solve(use_c, use_p, use_r)
        used_carbon = bool(use_c and carbon_budget is not None)
        if _ok(status, sol):
            break
    else:
        # Last attempt failed
        if enforce_carbon and carbon_budget is not None:
            return {
                "region": REGIONS[region_idx],
                "status": status or "Infeasible",
                "feasible": False,
                "carbon_enforced": True,
                "carbon_budget": float(carbon_budget),
                "objective": objective,
                "cost": float("nan"),
                "carbon": float("nan"),
                "re_utilization_of_deliverable": float("nan"),
                "peak_net_import": float("nan"),
                "mean_net_import": float("nan"),
                "grid_purchase": np.full(T, np.nan),
                "grid_sell": np.full(T, np.nan),
                "used_re": np.full(T, np.nan),
                "re_charge": np.full(T, np.nan),
                "grid_charge": np.full(T, np.nan),
                "discharge": np.full(T, np.nan),
                "curtailment": np.full(T, np.nan),
                "soc": np.full(T, np.nan),
                "net_import": np.full(T, np.nan),
                "total_load": load.copy(),
                "available_re": avail.copy(),
            }
        raise RuntimeError(f"Power LP failed for {REGIONS[region_idx]}: status={status}")

    mat = sol.reshape(T, NV)
    gp, gs, charge, dch, curt, soc = [mat[:, i] for i in range(NV)]
    net = gp - gs

    utilized = np.maximum(avail - curt, 0.0)
    re_ch = np.minimum(charge, utilized)
    sell_from_re = np.minimum(gs, np.maximum(utilized - re_ch, 0.0))
    used_re = np.maximum(utilized - re_ch - sell_from_re, 0.0)
    grid_ch = np.maximum(charge - re_ch, 0.0)

    re_den = float(avail.sum())
    util_deliv = float(utilized.sum() / re_den) if re_den > 1e-9 else 0.0
    carbon_val = float(np.dot(gp, ci))
    feasible = True
    if used_carbon and carbon_budget is not None and carbon_val > float(carbon_budget) + 1e-4:
        feasible = False

    return {
        "region": REGIONS[region_idx],
        "status": status,
        "feasible": feasible,
        "carbon_enforced": bool(enforce_carbon and carbon_budget is not None),
        "carbon_budget": None if carbon_budget is None else float(carbon_budget),
        "objective": objective,
        "objective_value": float(h.getObjectiveValue()),
        "cost": float(np.dot(gp, price) - np.dot(gs, sell_p)),
        "carbon": carbon_val,
        "re_utilization_of_deliverable": util_deliv,
        "peak_net_import": float(net.max()) if len(net) else 0.0,
        "mean_net_import": float(net.mean()) if len(net) else 0.0,
        "grid_purchase": gp,
        "grid_sell": gs,
        "used_re": used_re,
        "re_charge": re_ch,
        "grid_charge": grid_ch,
        "discharge": dch,
        "curtailment": curt,
        "soc": soc,
        "net_import": net,
        "total_load": load.copy(),
        "available_re": avail.copy(),
    }


def optimize_region_power(
    data: Q4Data,
    region_idx: int,
    total_load: np.ndarray,
    available_re: np.ndarray,
    *,
    carbon_budget: float | None = None,
    peak_cap: float | None = None,
    min_re_utilization: float | None = None,
    time_limit: float = 20.0,
    enforce_carbon: bool = False,
) -> dict:
    return _region_lp(
        data,
        region_idx,
        total_load,
        available_re,
        objective="cost",
        carbon_budget=carbon_budget,
        peak_cap=peak_cap,
        min_re_utilization=min_re_utilization,
        time_limit=time_limit,
        enforce_carbon=enforce_carbon,
    )


def minimize_region_carbon(
    data: Q4Data,
    region_idx: int,
    total_load: np.ndarray,
    available_re: np.ndarray,
    *,
    peak_cap: float | None = None,
    time_limit: float = 20.0,
) -> dict:
    return _region_lp(
        data,
        region_idx,
        total_load,
        available_re,
        objective="carbon",
        carbon_budget=None,
        peak_cap=peak_cap,
        min_re_utilization=None,
        time_limit=time_limit,
        enforce_carbon=False,
    )


def facility_loads(data: Q4Data, ai_power: np.ndarray) -> list[np.ndarray]:
    loads = []
    for r in range(len(REGIONS)):
        it = data.non_ai[r] + ai_power[r, :POWER_HOURS]
        loads.append(it * data.pue[r])
    return loads


def non_ai_carbon_floor(data: Q4Data, time_limit: float = 15.0) -> float:
    """Minimum system carbon with AI load = 0 (NonAI + storage only)."""
    ai = np.zeros((len(REGIONS), POWER_HOURS))
    loads = facility_loads(data, ai)
    total = 0.0
    for r in range(len(REGIONS)):
        res = minimize_region_carbon(data, r, loads[r], data.available_re[r], time_limit=time_limit)
        total += float(res["carbon"])
    return total


def optimize_all_regions(
    data: Q4Data,
    ai_power: np.ndarray,
    available_re: np.ndarray | None = None,
    *,
    carbon_budget_total: float | None = None,
    peak_caps: np.ndarray | None = None,
    min_re_utilization: float | None = None,
    time_limit: float = 20.0,
    enforce_carbon: bool = False,
) -> list[dict]:
    re = data.available_re if available_re is None else available_re
    re_raw = data.available_re_raw
    loads = facility_loads(data, ai_power)

    meta = {
        "carbon_budget_total": None if carbon_budget_total is None else float(carbon_budget_total),
        "carbon_min_given_schedule": None,
        "carbon_feasible": True,
        "carbon_infeasible_reason": None,
    }

    regional_budgets = [None] * len(REGIONS)
    emin = np.zeros(len(REGIONS))
    emin_results: list[dict] | None = None

    if carbon_budget_total is not None:
        emin_results = []
        for r in range(len(REGIONS)):
            p_cap = None if peak_caps is None else float(peak_caps[r])
            res = minimize_region_carbon(
                data, r, loads[r], re[r], peak_cap=p_cap, time_limit=time_limit
            )
            emin[r] = float(res["carbon"])
            emin_results.append(res)
        emin_total = float(emin.sum())
        meta["carbon_min_given_schedule"] = emin_total

        if emin_total > float(carbon_budget_total) + 1e-3:
            meta["carbon_feasible"] = False
            meta["carbon_infeasible_reason"] = (
                f"schedule_carbon_floor {emin_total:.3f} > budget {float(carbon_budget_total):.3f}"
            )
            load_sum = max(sum(float(x.sum()) for x in loads), 1e-9)
            for r, res in enumerate(emin_results):
                res["available_re_raw"] = re_raw[r].copy()
                res["carbon_budget"] = float(carbon_budget_total) * float(loads[r].sum() / load_sum)
                res["feasible"] = False
                res["carbon_enforced"] = True
                res["carbon_min_region"] = float(emin[r])
            emin_results[0]["_meta"] = meta
            return emin_results

        slack = float(carbon_budget_total) - emin_total
        # Residual slack proportional to (cost-min carbon − emin) headroom.
        headroom = np.zeros(len(REGIONS))
        for r in range(len(REGIONS)):
            p_cap = None if peak_caps is None else float(peak_caps[r])
            cost_res = optimize_region_power(
                data,
                r,
                loads[r],
                re[r],
                peak_cap=p_cap,
                min_re_utilization=None,
                time_limit=time_limit,
                enforce_carbon=False,
            )
            headroom[r] = max(float(cost_res["carbon"]) - emin[r], 0.0)
        if headroom.sum() > 1e-6:
            weights = headroom / headroom.sum()
        else:
            totals = np.array([float(x.sum()) for x in loads])
            weights = totals / max(totals.sum(), 1e-9)
        for r in range(len(REGIONS)):
            regional_budgets[r] = emin[r] + slack * float(weights[r])

    results = []
    for r in range(len(REGIONS)):
        p_cap = None if peak_caps is None else float(peak_caps[r])
        res = optimize_region_power(
            data,
            r,
            loads[r],
            re[r],
            carbon_budget=regional_budgets[r],
            peak_cap=p_cap,
            min_re_utilization=min_re_utilization,
            time_limit=time_limit,
            enforce_carbon=enforce_carbon and carbon_budget_total is not None,
        )
        res["available_re_raw"] = re_raw[r].copy()
        res["carbon_min_region"] = float(emin[r]) if carbon_budget_total is not None else None
        results.append(res)

    if carbon_budget_total is not None:
        finite = [float(r["carbon"]) for r in results if np.isfinite(r["carbon"])]
        actual = float(sum(finite)) if finite else float("inf")
        ok = (
            len(finite) == len(results)
            and all(bool(r.get("feasible", True)) for r in results)
            and actual <= float(carbon_budget_total) + 1e-2
        )
        meta["carbon_feasible"] = bool(ok)
        if not ok and meta["carbon_infeasible_reason"] is None:
            meta["carbon_infeasible_reason"] = (
                f"actual_carbon {actual:.3f} exceeds budget {float(carbon_budget_total):.3f}"
            )
        results[0]["_meta"] = meta
    return results
