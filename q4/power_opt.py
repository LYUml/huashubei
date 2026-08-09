from __future__ import annotations

import numpy as np

from config import POWER_HOURS, REGIONS
from data_loader import Q4Data


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
) -> dict:
    """
    Single-region LP (HiGHS) for storage + grid + renewable allocation.

    Official energy balance (attachment):
      GridPurchase + AvailableRE + Discharge
        = TotalLoad + Charge + GridSell + Curtailment
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

    def _solve(use_carbon, use_peak, use_re):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("presolve", "on")

        cost = np.zeros(n)
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

    h, status, sol = _solve(True, True, True)
    if sol.size != n or not np.isfinite(sol).all() or status not in ("Optimal", "Time limit reached"):
        h, status, sol = _solve(False, False, False)
        if sol.size != n or not np.isfinite(sol).all() or status not in ("Optimal", "Time limit reached"):
            raise RuntimeError(f"Power LP failed for {REGIONS[region_idx]}: status={status}")

    mat = sol.reshape(T, NV)
    gp, gs, charge, dch, curt, soc = [mat[:, i] for i in range(NV)]
    net = gp - gs

    # Reporting split of renewable utilization (attachment definition):
    # utilized = AvailableRE - Curtailment = direct use + RE charge + export.
    utilized = np.maximum(avail - curt, 0.0)
    # Heuristic split for time-series CSV only.
    re_ch = np.minimum(charge, utilized)
    sell_from_re = np.minimum(gs, np.maximum(utilized - re_ch, 0.0))
    used_re = np.maximum(utilized - re_ch - sell_from_re, 0.0)
    grid_ch = np.maximum(charge - re_ch, 0.0)

    # Operational fill-rate only; headline utilization vs raw AvailableRE is
    # computed in metrics.aggregate_power_metrics.
    re_den = float(avail.sum())
    util_deliv = float(utilized.sum() / re_den) if re_den > 1e-9 else 0.0

    return {
        "region": REGIONS[region_idx],
        "status": status,
        "objective": float(h.getObjectiveValue()),
        "cost": float(np.dot(gp, price) - np.dot(gs, sell_p)),
        "carbon": float(np.dot(gp, ci)),
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


def optimize_all_regions(
    data: Q4Data,
    ai_power: np.ndarray,
    available_re: np.ndarray | None = None,
    *,
    carbon_budget_total: float | None = None,
    peak_caps: np.ndarray | None = None,
    min_re_utilization: float | None = None,
    time_limit: float = 20.0,
) -> list[dict]:
    re = data.available_re if available_re is None else available_re
    re_raw = data.available_re_raw
    totals = []
    loads = []
    for r in range(len(REGIONS)):
        it = data.non_ai[r] + ai_power[r]
        total = it * data.pue[r]
        loads.append(total)
        totals.append(float(total.sum()))
    totals_arr = np.asarray(totals)
    share = totals_arr / max(totals_arr.sum(), 1e-9)

    # For tight carbon budgets, first estimate an unavoidable floor by
    # minimizing carbon alone (same LP with ci as purchase cost, no sell reward).
    min_carbon = np.zeros(len(REGIONS))
    if carbon_budget_total is not None:
        for r in range(len(REGIONS)):
            # Proxy floor: purchase at least unmet load after RE, times min carbon intensity.
            unmet = np.maximum(loads[r] - re[r], 0.0)
            min_carbon[r] = float(np.dot(unmet, data.carbon[r]))

    results = []
    for r in range(len(REGIONS)):
        c_budget = None
        if carbon_budget_total is not None:
            # Soften infeasibility: never ask below 1% above regional unavoidable floor.
            c_budget = max(float(carbon_budget_total * share[r]), 1.01 * min_carbon[r])
        p_cap = None if peak_caps is None else float(peak_caps[r])
        res = optimize_region_power(
            data,
            r,
            loads[r],
            re[r],
            carbon_budget=c_budget,
            peak_cap=p_cap,
            min_re_utilization=min_re_utilization,
            time_limit=time_limit,
        )
        res["available_re_raw"] = re_raw[r].copy()
        results.append(res)
    return results
