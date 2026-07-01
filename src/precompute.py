"""
precompute.py — Phase 6: pre-compute baseline + optimized pipeline results.

Run ONCE before launching the dashboard:
    python src/precompute.py

This is the ONLY place any OR solver is called (~120s for the MILP solve).
Writes data/processed/dashboard_cache.pkl which app.py loads read-only.
Re-run after any config or parameter change to refresh the cache.
"""

from __future__ import annotations

import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_MODELS = os.path.join(_HERE, "models")
for _p in (_HERE, _MODELS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from parameters import Parameters
from baseline import run_baseline
from cascade import run_optimized_pipeline


def main() -> None:
    print("=" * 70)
    print("PlanFlow precompute — building dashboard cache")
    print("=" * 70)

    if not os.path.isfile(config.PARAMETERS_PATH):
        print(f"ERROR: no parameters at {config.PARAMETERS_PATH}.")
        print("Run `python src/build_parameters.py` first.")
        sys.exit(1)

    params = Parameters.load(config.PARAMETERS_PATH)
    inst = params.instance
    periods = list(inst.periods)
    plants = list(inst.plants)
    regions = list(inst.regions)
    dcs = list(inst.dcs)
    T = len(periods)

    # ---- Baseline -----------------------------------------------------------
    print("\n[1/2] Running baseline...")
    baseline_result = run_baseline(params)

    # ---- Optimized pipeline -------------------------------------------------
    print("\n[2/2] Running optimized pipeline (MILP ~120s)...")
    optimized_result = run_optimized_pipeline(params)

    # ---- Pre-process for dashboard ------------------------------------------
    print("\nPre-processing for dashboard...")

    b_kpi = baseline_result["kpi"]
    o_kpi = optimized_result["kpi"]

    # Per-period total demand (ordered by period label)
    period_demand_series = inst.demand.groupby("period")["demand"].sum().reindex(periods)
    period_demand = {p: float(period_demand_series[p]) for p in periods}

    # Per-region demand
    region_demand_series = inst.demand.groupby("region")["demand"].sum()
    region_total_demand = {r: float(region_demand_series.get(r, 0.0)) for r in regions}

    # MILP: per-period total production + per-plant-per-period breakdown
    milp_pp = optimized_result["milp"]["production_plan"]  # {(plant, sku, ti): qty}
    milp_prod_period = {p: 0.0 for p in periods}
    milp_prod_by_plant = {pl: {p: 0.0 for p in periods} for pl in plants}
    for (pl, sku, ti), qty in milp_pp.items():
        plabel = periods[ti]
        milp_prod_period[plabel] += qty
        milp_prod_by_plant[pl][plabel] += qty

    # Plant total production (summed across all periods)
    plant_total_production = {pl: sum(milp_prod_by_plant[pl].values()) for pl in plants}

    # Per-plant average utilization from MILP
    milp_util = optimized_result["milp"]["utilization"]  # {(plant, t): fraction}
    plant_avg_util = {
        pl: sum(milp_util.get((pl, t), 0.0) for t in range(T)) / T
        for pl in plants
    }

    # DP diagnostic: per-SKU n_runs
    dp_diag = optimized_result["dp_diagnostic"]  # {sku: {n_runs, ...}}
    dp_n_runs = {int(sku): int(info["n_runs"]) for sku, info in dp_diag.items()}
    runs_list = list(dp_n_runs.values())
    dp_summary = {
        "avg": round(sum(runs_list) / len(runs_list), 2),
        "min": min(runs_list),
        "max": max(runs_list),
        "n_skus": len(runs_list),
        "target_interval": params.meta.get("target_reorder_interval", 3),
    }

    # Per-period total inventory held (sum across all SKUs at end of each period)
    milp_inv = optimized_result["milp"]["inventory"]  # {(sku, ti): qty}, zeros omitted
    milp_inventory_period = {p: 0.0 for p in periods}
    for (sku, ti), qty in milp_inv.items():
        milp_inventory_period[periods[ti]] += qty

    # Transport flows from optimized pipeline: {(plant, region): qty}
    opt_flows = optimized_result["transportation"]["flows"]

    # Baseline shipping: {region: source_id}
    baseline_source = baseline_result["shipping"]["region_to_source"]

    # Plant-city and DC-city mappings (for map labels)
    plant_city = dict(config.PLANTS)
    dc_city = dict(config.DCS)

    # Processing cost DataFrame + assignment
    processing_cost_df = params.processing_cost
    lines = list(params.lines)
    families = list(params.families)
    optimal_assignment = optimized_result["assignment"]["assignment"]
    baseline_assignment = baseline_result["line_assignment"]["line_to_family"]

    # ---- Assemble cache -----------------------------------------------------
    cache = {
        # KPIs
        "baseline_kpi": dict(b_kpi),
        "optimized_kpi": dict(o_kpi),
        "baseline_n_setups": int(baseline_result["production"]["n_setups"]),

        # Production plan
        "periods": periods,
        "period_demand": period_demand,
        "milp_prod_period": milp_prod_period,
        "milp_prod_by_plant": milp_prod_by_plant,
        "milp_inventory_period": milp_inventory_period,  # {period_label: total_inv}

        # DP diagnostic
        "dp_n_runs": dp_n_runs,
        "dp_summary": dp_summary,

        # Transport
        "opt_flows": opt_flows,             # {(plant, region): qty}
        "plant_total_production": plant_total_production,
        "region_total_demand": region_total_demand,
        "baseline_source": baseline_source, # {region: source_id}

        # Geographic
        "coords": dict(params.coords),      # all offset facility + region coords
        "plants": plants,
        "dcs": dcs,
        "regions": regions,
        "plant_city": plant_city,           # {plant_id: city_name}
        "dc_city": dc_city,                 # {dc_id: city_name}

        # Assignment
        "processing_cost_df": processing_cost_df,
        "lines": lines,
        "families": families,
        "optimal_assignment": dict(optimal_assignment),
        "baseline_assignment": dict(baseline_assignment),

        # Operational
        "bottleneck_plant": optimized_result["bottleneck_plant"],
        "plant_avg_utilization": plant_avg_util,
    }

    # ---- Save ---------------------------------------------------------------
    cache_path = os.path.join(_ROOT, "data", "processed", "dashboard_cache.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Verification -------------------------------------------------------
    print("\n" + "=" * 70)
    print("CACHE VERIFICATION  (must match Phase 4 report within 0.1%)")
    print("=" * 70)

    b_ctrl = b_kpi["controllable_cost"]
    o_ctrl = o_kpi["controllable_cost"]
    b_tr   = b_kpi["transport_cost"]
    o_tr   = o_kpi["transport_cost"]
    b_ns   = cache["baseline_n_setups"]
    o_ns   = o_kpi["n_setups"]
    b_pt   = b_kpi["processing_time"]
    o_pt   = o_kpi["processing_time"]
    gap    = o_kpi["mip_gap"]

    # MIP gap is a time-limited heuristic result; allow 5% relative tolerance.
    # All other KPIs are deterministic; fail hard if they drift > 0.1%.
    rows = [
        ("Controllable cost baseline",  b_ctrl, 4_411_461.06,  "14,.2f",  0.001),
        ("Controllable cost optimized", o_ctrl, 4_248_169.06,  "14,.2f",  0.001),
        ("Transport cost baseline",     b_tr,   396_960.20,    "14,.2f",  0.001),
        ("Transport cost optimized",    o_tr,   821_163.34,    "14,.2f",  0.001),
        ("Setup count baseline",        b_ns,   360,           "14d",     0.001),
        ("Setup count optimized",       o_ns,   279,           "14d",     0.001),
        ("Processing time baseline",    b_pt,   23.9084,       "14.4f",   0.001),
        ("Processing time optimized",   o_pt,   22.4568,       "14.4f",   0.001),
        ("MIP gap (time-limited)",      gap,    0.0278,        "14.4%",   0.05),
    ]

    hard_fail = False
    for label, actual, expected, fmt, tol in rows:
        diff = abs(actual - expected) / abs(expected) if expected != 0 else 0
        status = "OK " if diff <= tol else "WARN"
        if diff > tol:
            hard_fail = True
        suffix = " time units" if "Processing" in label else ""
        print(f"  {status}  {label:<38} {format(actual, fmt)}{suffix}")

    if not hard_fail:
        print("\n  All numbers match Phase 4 (within tolerance). Cache is valid.")
    else:
        print("\n  HARD MISMATCH on deterministic KPI — review before proceeding.")
        sys.exit(1)

    print(f"\n  Cache written -> {cache_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
