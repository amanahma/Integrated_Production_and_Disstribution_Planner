"""
dp.py — Model 1: single-item Wagner-Whitin lot-sizing via dynamic programming.

Pure Python, no solver. O(n^2) DP using cumulative arrays.

HOLDING COST CONVENTION (fixed for the whole project):
  Demand for period t, if produced in an earlier period s <= t, incurs holding
  cost = holding_cost * demand[t] * (t - s). Inventory is charged on end-of-period
  stock; demand produced in its own period (s == t) incurs zero holding. This
  exact convention is mirrored by the brute-force checker in validate.py.
"""

from __future__ import annotations

EPS = 1e-9  # float zero-tolerance for demand comparisons


def wagner_whitin(demand: list[float],
                  setup_cost: float,
                  holding_cost: float) -> dict:
    """Minimum (setup + holding) cost production plan over the horizon.

    No backorders, zero starting/ending inventory. See module docstring for the
    holding convention. Returns the dict described in the Phase-3 spec.
    """
    n = len(demand)
    if n == 0:
        return {"total_cost": 0.0, "production_periods": [], "lot_sizes": {},
                "setup_cost_total": 0.0, "holding_cost_total": 0.0}

    # Cumulative demand:        D[k] = sum_{t<k} demand[t]
    # Cumulative weighted dem.: W[k] = sum_{t<k} t * demand[t]
    # Then sum_{t=j..k} demand[t]*(t-j) = (W[k+1]-W[j]) - j*(D[k+1]-D[j]).
    D = [0.0] * (n + 1)
    W = [0.0] * (n + 1)
    for t in range(n):
        D[t + 1] = D[t] + demand[t]
        W[t + 1] = W[t] + t * demand[t]

    def holding(j: int, k: int) -> float:
        """Holding cost of producing in period j all demand of periods j..k."""
        units_weighted = (W[k + 1] - W[j]) - j * (D[k + 1] - D[j])
        return holding_cost * units_weighted

    # F[k] = min cost to satisfy periods 0..k-1 ending at a regeneration point.
    # F[0] = 0. Answer is F[n]. arg[k] = chosen last production period (0-indexed)
    # for the block ending at period k-1.
    F = [0.0] + [float("inf")] * n
    arg = [-1] * (n + 1)
    for k in range(1, n + 1):
        # If the block (the periods from some j..k-1) has zero demand, producing
        # in it would charge a needless setup. The recurrence naturally avoids
        # that because a zero-demand tail adds 0 holding and an unavoidable setup;
        # but to honor "no setup on a zero-demand block", we only *open* a block
        # at j if demand[j..k-1] > 0. A pure zero block is folded into F[k]=F[k-1].
        if demand[k - 1] <= EPS and F[k - 1] < F[k]:
            # period k-1 has no demand: it can extend the previous solution at no
            # cost (carry the regeneration point forward).
            F[k] = F[k - 1]
            arg[k] = arg[k - 1]
        for j in range(1, k + 1):
            # block covers periods j-1 .. k-1 (0-indexed), produced in period j-1
            if D[k] - D[j - 1] <= EPS:
                continue  # zero-demand block: never open a setup for it
            cand = F[j - 1] + setup_cost + holding(j - 1, k - 1)
            if cand < F[k] - 1e-12:
                F[k] = cand
                arg[k] = j - 1  # 0-indexed production period

    # --- recover plan via backpointers --------------------------------------
    production_periods: list[int] = []
    k = n
    while k > 0:
        j = arg[k]
        if j == -1:
            # no production needed for the remaining prefix (all zero demand)
            break
        production_periods.append(j)
        k = j  # previous regeneration point ends at period j-1 -> F[j]
    production_periods.sort()

    # --- lot sizes: each production period covers demand up to the next one ---
    lot_sizes: dict[int, float] = {}
    for idx, p in enumerate(production_periods):
        nxt = production_periods[idx + 1] if idx + 1 < len(production_periods) else n
        qty = sum(demand[p:nxt])
        lot_sizes[p] = qty

    setup_total = setup_cost * len(production_periods)
    holding_total = 0.0
    for idx, p in enumerate(production_periods):
        nxt = production_periods[idx + 1] if idx + 1 < len(production_periods) else n
        for t in range(p, nxt):
            holding_total += holding_cost * demand[t] * (t - p)

    total_cost = setup_total + holding_total
    return {
        "total_cost": total_cost,
        "production_periods": production_periods,
        "lot_sizes": lot_sizes,
        "setup_cost_total": setup_total,
        "holding_cost_total": holding_total,
    }
    # NOTE: an O(n log n) Wagner-Whitin variant exists (geometric/convex-hull
    # technique); not implemented here — O(n^2) is ample for a 12-period horizon.


def windowed_restrict(dp_seed: dict, T: int, W: int,
                      symmetric: bool = True) -> dict:
    """Expand each SKU's WW production periods by a window, clipped to [0, T-1].

    symmetric=True (default): expand by [-W, +W] around each WW period.
    symmetric=False: forward-only, [0, +W] — allows the MILP to defer
    production by up to W periods without opening any earlier periods.

    Returns {sku: sorted list of allowed periods}.
    """
    result = {}
    for sku, info in dp_seed.items():
        base = info["dp_production_periods"]
        allowed = set()
        for p in base:
            lo = -W if symmetric else 0
            for delta in range(lo, W + 1):
                t = p + delta
                if 0 <= t < T:
                    allowed.add(t)
        result[sku] = sorted(allowed)
    return result


def dp_all_skus(params, capacity_aware: bool = True) -> dict:
    """Run Wagner-Whitin for every SKU on its all-regions-aggregated demand.

    Returns {sku: {"production_periods", "dp_production_periods", "total_cost",
    "total_demand", "n_runs"}} and prints a one-line batching summary.

    "production_periods" == "dp_production_periods": both are the pure WW result.
    Windowed expansion and feasibility escalation live in validate.suite_milp
    (or any MILP caller) via windowed_restrict(); they are not applied here.

    If capacity_aware=True (default), prints a preview of the W=SEED_WINDOW
    window expansion and its approximate per-period load (informational only —
    it does NOT modify "production_periods").
    """
    inst = params.instance
    periods = inst.periods
    T = len(periods)
    dem = inst.demand
    pivot = (dem.groupby(["sku", "period"], observed=True)["demand"].sum()
             .unstack("period").reindex(columns=periods, fill_value=0.0))

    out: dict = {}
    runs = []
    for sku in inst.skus:
        series = [float(pivot.at[sku, p]) if sku in pivot.index else 0.0
                  for p in periods]
        h = float(params.sku_table.at[sku, "holding_cost"])
        a = float(params.sku_table.at[sku, "setup_cost"])
        res = wagner_whitin(series, setup_cost=a, holding_cost=h)
        n_runs = len(res["production_periods"])
        out[sku] = {
            "production_periods": list(res["production_periods"]),
            "dp_production_periods": list(res["production_periods"]),
            "total_cost": res["total_cost"],
            "total_demand": sum(series),
            "n_runs": n_runs,
        }
        runs.append(n_runs)

    avg = sum(runs) / len(runs)
    print(f"[DP] dp_all_skus: avg runs/SKU = {avg:.2f}  "
          f"(min {min(runs)}, max {max(runs)}) over {len(runs)} SKUs; "
          f"horizon = {T} periods")
    if avg >= T:
        print("  *** LOUD WARNING: avg runs/SKU >= horizon -> NO batching is "
              "happening. Setup costs are too low / capacity too loose. Phase 2 "
              "calibration needs revisiting. ***")
    else:
        print(f"  Batching confirmed: avg runs {avg:.2f} < {T} "
              f"(target interval ~{params.meta.get('target_reorder_interval')}).")
    assert avg < T, "dp_all_skus: average runs/SKU must be < horizon"

    if capacity_aware:
        import config as _cfg
        W0 = _cfg.SEED_WINDOW
        cap_total = sum(float(v) for v in params.plant_capacity.values())
        preview = windowed_restrict(out, T, W0)
        exp_runs = [len(preview[s]) for s in inst.skus]
        avg_exp = sum(exp_runs) / len(exp_runs)
        load = [0.0] * T
        for s in inst.skus:
            avg_lot = out[s]["total_demand"] / max(1, out[s]["n_runs"])
            for t in preview[s]:
                load[t] += avg_lot
        peak_ratio = max(load) / cap_total if cap_total else float("inf")
        print(f"  windowed-seed preview (W={W0}): avg allowed/SKU = {avg_exp:.2f} "
              f"(min {min(exp_runs)}, max {max(exp_runs)}); "
              f"approx peak load = {peak_ratio*100:.1f}% of pooled capacity "
              f"[pre-check heuristic; escalation runs in validate suite_milp]")

    return out
