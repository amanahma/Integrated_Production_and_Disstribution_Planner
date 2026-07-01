"""
milp.py — Model 3: multi-SKU, multi-plant capacitated lot-sizing MILP (PuLP/CBC).

Standalone and independently tested. NOT wired into a cascade (Phase 4 does that).

Modeling choices (documented):
  - Inventory I[sku,t] is aggregated across plants (a single network-wide stock
    per SKU), not tracked per plant. Production X is per (plant, sku, t).
  - Demand demand_total[sku,t] is the SKU's demand summed across ALL regions in
    period t (region allocation is the transportation model's job, not this one).
  - Initial inventory I[sku,-1] = 0; no ending-inventory requirement (the model
    will not over-produce because production cost + holding are penalized).
  - Capacity: 1 unit of any product consumes UNIT_CAPACITY_CONSUMPTION units of
    plant capacity. Setup-time capacity consumption is omitted for now.
"""

from __future__ import annotations

import math
import os
import sys
import time

# Make this module importable standalone: ensure src/ (which holds config.py) is
# on the path regardless of how the module is loaded.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pulp

import config


def _demand_total(params) -> dict:
    """{(sku, t_index): demand summed across regions} for t in 0..T-1."""
    inst = params.instance
    periods = inst.periods
    pivot = (inst.demand.groupby(["sku", "period"], observed=True)["demand"].sum()
             .unstack("period").reindex(columns=periods, fill_value=0.0))
    dt = {}
    for sku in inst.skus:
        for ti, p in enumerate(periods):
            dt[(sku, ti)] = float(pivot.at[sku, p]) if sku in pivot.index else 0.0
    return dt


def solve_milp(params,
               restrict_setups_to: dict | None = None,
               solver_msg: bool = False,
               time_limit_sec: float | None = None,
               gap_rel: float | None = None) -> dict:
    """Capacitated lot-sizing MILP. See module + Phase-3 spec for details."""
    inst = params.instance
    plants = list(inst.plants)
    skus = list(inst.skus)
    T = len(inst.periods)
    a_cap = config.UNIT_CAPACITY_CONSUMPTION

    st = params.sku_table
    setup_c = {s: float(st.at[s, "setup_cost"]) for s in skus}
    hold_c = {s: float(st.at[s, "holding_cost"]) for s in skus}
    prod_c = {s: float(st.at[s, "production_cost"]) for s in skus}
    cap = {p: float(params.plant_capacity[p]) for p in plants}
    dt = _demand_total(params)

    # Big-M, tightened (and documented): a single (plant,sku,period) run never
    # needs to exceed the demand still outstanding from t to the end of the
    # horizon (no backorders, no ending inventory => producing more is never
    # useful), and can never exceed that plant's per-period capacity. So
    #   M[plant,sku,t] = min(plant_capacity[plant], remaining_demand[sku,t]).
    # This is valid and far tighter than "M = SKU total horizon demand", which
    # made the LP relaxation very weak and CBC extremely slow.
    rem = {}  # rem[(s,t)] = sum_{u>=t} demand_total[s,u]
    for s in skus:
        acc = 0.0
        for t in range(T - 1, -1, -1):
            acc += dt[(s, t)]
            rem[(s, t)] = acc
    bigM = {(p, s, t): min(cap[p], rem[(s, t)])
            for p in plants for s in skus for t in range(T)}

    prob = pulp.LpProblem("capacitated_lot_sizing", pulp.LpMinimize)

    X = {(p, s, t): pulp.LpVariable(f"X_{p}_{s}_{t}", lowBound=0)
         for p in plants for s in skus for t in range(T)}
    Y = {(p, s, t): pulp.LpVariable(f"Y_{p}_{s}_{t}", cat="Binary")
         for p in plants for s in skus for t in range(T)}
    I = {(s, t): pulp.LpVariable(f"I_{s}_{t}", lowBound=0)
         for s in skus for t in range(T)}
    S = {(s, t): pulp.LpVariable(f"S_{s}_{t}", lowBound=0)
         for s in skus for t in range(T)}

    # --- objective ----------------------------------------------------------
    prob += (
        pulp.lpSum(setup_c[s] * Y[(p, s, t)] for p in plants for s in skus for t in range(T))
        + pulp.lpSum(hold_c[s] * I[(s, t)] for s in skus for t in range(T))
        + pulp.lpSum(prod_c[s] * X[(p, s, t)] for p in plants for s in skus for t in range(T))
        + pulp.lpSum(config.SHORTAGE_PENALTY * S[(s, t)] for s in skus for t in range(T))
    )

    # --- (a) inventory balance ---------------------------------------------
    for s in skus:
        for t in range(T):
            prev = I[(s, t - 1)] if t > 0 else 0.0
            prob += (
                I[(s, t)] == prev
                + pulp.lpSum(X[(p, s, t)] for p in plants)
                + S[(s, t)] - dt[(s, t)]
            ), f"bal_{s}_{t}"

    # --- (b) setup linking (big-M) -----------------------------------------
    for p in plants:
        for s in skus:
            for t in range(T):
                prob += (X[(p, s, t)] <= bigM[(p, s, t)] * Y[(p, s, t)],
                         f"link_{p}_{s}_{t}")

    # --- (c) shared plant capacity -----------------------------------------
    for p in plants:
        for t in range(T):
            prob += (pulp.lpSum(a_cap * X[(p, s, t)] for s in skus) <= cap[p],
                     f"cap_{p}_{t}")

    # --- DP seeding: fix Y=0 for disallowed periods ------------------------
    if restrict_setups_to is not None:
        for s in skus:
            allowed = set(restrict_setups_to.get(s, set()))
            for t in range(T):
                if t not in allowed:
                    for p in plants:
                        prob += Y[(p, s, t)] == 0, f"seed0_{p}_{s}_{t}"

    # --- solve (HiGHS via PuLP / highspy) ----------------------------------
    # HiGHS replaces CBC: CBC's feasibility pump spun without honoring the time
    # limit on this big-M model. HiGHS returns a near-optimal incumbent in
    # seconds and honors time_limit/gap. Single-threaded => HiGHS is
    # deterministic by default (no explicit seed option needed; this PuLP build's
    # solverParams pass-through is broken, so we rely on default determinism).
    solver_kwargs = dict(
        msg=bool(solver_msg),
        threads=1,
    )
    if time_limit_sec is not None:
        solver_kwargs["timeLimit"] = float(time_limit_sec)
    if gap_rel is not None:
        solver_kwargs["gapRel"] = gap_rel
    solver = pulp.HiGHS(**solver_kwargs)
    t0 = time.time()
    prob.solve(solver)
    solve_time = time.time() - t0
    status = pulp.LpStatus[prob.status]

    # achieved MIP gap + best (dual) bound from the underlying HiGHS model
    best_bound, mip_gap = None, None
    try:
        info = prob.solverModel.getInfo()
        bb = getattr(info, "mip_dual_bound", None)
        gp = getattr(info, "mip_gap", None)
        best_bound = float(bb) if bb is not None else None
        mip_gap = float(gp) if gp is not None else None
    except Exception:
        pass

    objective_val = pulp.value(prob.objective)
    has_incumbent = objective_val is not None and math.isfinite(objective_val)

    # --- extract ------------------------------------------------------------
    def val(v):
        x = v.value()
        return 0.0 if x is None else float(x)

    cost_setup = sum(setup_c[s] * val(Y[(p, s, t)])
                     for p in plants for s in skus for t in range(T))
    cost_holding = sum(hold_c[s] * val(I[(s, t)]) for s in skus for t in range(T))
    cost_production = sum(prod_c[s] * val(X[(p, s, t)])
                          for p in plants for s in skus for t in range(T))
    total_shortage = sum(val(S[(s, t)]) for s in skus for t in range(T))
    cost_shortage = config.SHORTAGE_PENALTY * total_shortage
    n_setups = sum(1 for p in plants for s in skus for t in range(T)
                   if val(Y[(p, s, t)]) > 0.5)

    production_plan = {(p, s, t): val(X[(p, s, t)])
                       for p in plants for s in skus for t in range(T)
                       if val(X[(p, s, t)]) > 1e-6}
    inventory = {(s, t): val(I[(s, t)])
                 for s in skus for t in range(T) if val(I[(s, t)]) > 1e-6}
    utilization = {}
    for p in plants:
        for t in range(T):
            used = sum(a_cap * val(X[(p, s, t)]) for s in skus)
            utilization[(p, t)] = used / cap[p] if cap[p] else 0.0

    # achieved relative gap: prefer HiGHS's own mip_gap, else compute from bound
    if mip_gap is not None and math.isfinite(mip_gap):
        achieved_gap = mip_gap
    elif best_bound is not None and has_incumbent and abs(objective_val) > 0:
        achieved_gap = abs(objective_val - best_bound) / abs(objective_val)
    else:
        achieved_gap = float("nan")

    # --- print status + shortage FIRST (accept time-limited incumbents) -----
    print(f"[MILP] status = {status} | incumbent = "
          f"{objective_val if not has_incumbent else round(objective_val, 2)} | "
          f"gap = {achieved_gap:.4%} | total_shortage_units = {total_shortage:,.4f}"
          f" | n_setups = {n_setups} | solve = {solve_time:.2f}s")
    if not has_incumbent:
        print("  *** NO FEASIBLE INCUMBENT returned — inspect the model. ***")
    if total_shortage > 1e-3:
        carriers = [(s, t, val(S[(s, t)])) for s in skus for t in range(T)
                    if val(S[(s, t)]) > 1e-6]
        print(f"  *** MATERIAL SHORTAGE on {len(carriers)} (sku,period) cells: "
              f"{carriers[:10]}{' ...' if len(carriers) > 10 else ''} ***")

    return {
        "status": status,
        "has_incumbent": bool(has_incumbent),
        "objective": float(objective_val) if has_incumbent else float("nan"),
        "best_bound": best_bound,
        "mip_gap": achieved_gap,
        "cost_setup": float(cost_setup),
        "cost_holding": float(cost_holding),
        "cost_production": float(cost_production),
        "cost_shortage": float(cost_shortage),
        "total_shortage_units": float(total_shortage),
        "n_setups": int(n_setups),
        "production_plan": production_plan,
        "inventory": inventory,
        "utilization": utilization,
        "solve_time_sec": float(solve_time),
    }
