"""
baseline.py — the locked "before" landed cost (Phase 2).

Computes the REAL baseline using the synthesized Parameters (Phase 2). Three
deliberately dumb policies, emitting the same KPI structure later phases produce:

  1. Production = lot-for-lot     (produce each SKU every period it has demand;
                                   a setup is charged each such period; zero
                                   inventory held)
  2. Shipping   = greedy nearest-source (each region served entirely from the
                                   single cheapest source by transport cost;
                                   capacity ignored)
  3. Line assign = arbitrary identity order (lines -> families in default order)

Baseline total landed cost = production + setup + holding + transport.
Processing cost (assignment) is reported separately in the breakdown.

Run:  python src/baseline.py
"""

from __future__ import annotations

import sys

import pandas as pd

import config
from parameters import Parameters


# --------------------------------------------------------------------------- #
# 1. Production: lot-for-lot
# --------------------------------------------------------------------------- #
def baseline_production(params: Parameters) -> dict:
    inst = params.instance
    setup = params.setup_cost
    prod_cost = params.production_cost

    # demand per (sku, period), summed across regions
    sp = inst.demand.groupby(["sku", "period"], observed=True)["demand"].sum()

    # a setup is charged for every (sku, period) with positive demand
    positive = sp[sp > 0]
    n_setups = int(len(positive))
    setup_cost = float(sum(setup[s] for s, _ in positive.index))

    total_by_sku = inst.demand.groupby("sku")["demand"].sum()
    units_produced = float(total_by_sku.sum())
    production_cost = float(sum(prod_cost[s] * q for s, q in total_by_sku.items()))
    holding_cost = 0.0  # lot-for-lot carries nothing

    return {
        "policy": "lot-for-lot",
        "units_produced": units_produced,
        "n_setups": n_setups,
        "setup_cost": setup_cost,
        "production_cost": production_cost,
        "holding_cost": holding_cost,
    }


# --------------------------------------------------------------------------- #
# 2. Shipping: greedy nearest-source (by transport cost)
# --------------------------------------------------------------------------- #
def baseline_shipping(params: Parameters) -> dict:
    inst = params.instance
    # unit transport cost from every source (plants + DCs) to every region
    unit_cost = pd.concat([params.cost_plant_region, params.cost_dc_region])  # rows=sources
    region_demand = inst.demand.groupby("region")["demand"].sum()

    assignment, rows, transport_cost = {}, [], 0.0
    for r in inst.regions:
        col = unit_cost[r]
        best = col.idxmin()
        ucost = float(col.min())
        units = float(region_demand.get(r, 0.0))
        cost = ucost * units
        assignment[r] = best
        transport_cost += cost
        rows.append((r, best, round(ucost, 5), units, cost))

    detail = pd.DataFrame(rows, columns=["region", "source", "unit_cost", "units", "cost"])
    return {
        "policy": "greedy-nearest-source",
        "region_to_source": assignment,
        "transport_cost": transport_cost,
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# 3. Line assignment: arbitrary identity order
# --------------------------------------------------------------------------- #
def baseline_line_assignment(params: Parameters) -> dict:
    lines, families = params.lines, params.families
    pc = params.processing_cost
    assignment = {l: f for l, f in zip(lines, families)}      # identity order
    # NOTE: the matrix holds processing TIME (base_time * line_efficiency * noise),
    # not a currency cost. This is a separate time-based KPI; it is NOT part of
    # controllable or landed cost.
    processing_time = float(sum(pc.at[l, f] for l, f in assignment.items()))

    return {
        "policy": "arbitrary-identity",
        "n_families": len(families),
        "n_lines": len(lines),
        "line_to_family": assignment,
        "processing_time": processing_time,  # time units, separate dimension
    }


# --------------------------------------------------------------------------- #
# Combined KPI report
# --------------------------------------------------------------------------- #
def run_baseline(params: Parameters) -> dict:
    prod = baseline_production(params)
    ship = baseline_shipping(params)
    line = baseline_line_assignment(params)

    setup_cost = prod["setup_cost"]
    holding_cost = prod["holding_cost"]
    transport_cost = ship["transport_cost"]
    production_passthrough = prod["production_cost"]  # fixed: total demand produced

    # CONTROLLABLE COST = the headline KPI the optimization can actually change.
    controllable_cost = setup_cost + holding_cost + transport_cost
    total_landed_cost = controllable_cost + production_passthrough

    # This is the KPI dict later phases reuse — comparison is apples-to-apples
    # on `controllable_cost`. Processing time is a separate dimension (time units).
    return {
        "production": prod,
        "shipping": ship,
        "line_assignment": line,
        "kpi": {
            "controllable_cost": controllable_cost,   # <-- headline
            "setup_cost": setup_cost,
            "holding_cost": holding_cost,
            "transport_cost": transport_cost,
            "production_cost_passthrough": production_passthrough,
            "total_landed_cost": total_landed_cost,
            "processing_time": line["processing_time"],  # separate, time units
        },
    }


def print_report(res: dict) -> None:
    prod, ship, line, kpi = (res["production"], res["shipping"],
                             res["line_assignment"], res["kpi"])

    print("=" * 70)
    print("BASELINE (locked 'before' — naive policies, real Phase-2 parameters)")
    print("=" * 70)
    print("\n[1] Production - lot-for-lot (setup charged every period a SKU sells)")
    print(f"    units produced : {prod['units_produced']:,.1f}")
    print(f"    # setups       : {prod['n_setups']}")
    print(f"    setup cost     : {prod['setup_cost']:,.2f}")
    print(f"    production cost: {prod['production_cost']:,.2f}")
    print(f"    holding cost   : {prod['holding_cost']:,.2f}")

    print("\n[2] Shipping - greedy nearest-source by transport cost (capacity ignored)")
    print(ship["detail"].to_string(index=False))
    print(f"    transport cost : {ship['transport_cost']:,.2f}")

    print("\n[3] Line assignment - arbitrary identity order")
    for l, f in line["line_to_family"].items():
        print(f"      {l:<8} -> {f}")
    print(f"    processing time: {line['processing_time']:,.2f}  (time units)")

    ctrl = kpi["controllable_cost"]
    print("\n" + "=" * 70)
    print("HEADLINE KPI — CONTROLLABLE COST  (= setup + holding + transport)")
    print("  This is what the optimization can change; later phases compare here.")
    print("-" * 70)
    print(f"    setup_cost       : {kpi['setup_cost']:>16,.2f}  "
          f"({kpi['setup_cost']/ctrl*100:5.1f}%)")
    print(f"    holding_cost     : {kpi['holding_cost']:>16,.2f}  "
          f"({kpi['holding_cost']/ctrl*100:5.1f}%)")
    print(f"    transport_cost   : {kpi['transport_cost']:>16,.2f}  "
          f"({kpi['transport_cost']/ctrl*100:5.1f}%)")
    print(f"    {'CONTROLLABLE COST':<16} : {ctrl:>16,.2f}  (100.0%)")
    print("-" * 70)
    print(f"    production_cost  : {kpi['production_cost_passthrough']:>16,.2f}  "
          f"<- PASS-THROUGH (not optimized; total demand is produced regardless)")
    print("-" * 70)
    print(f"    {'TOTAL LANDED':<16} : {kpi['total_landed_cost']:>16,.2f}  "
          f"(= controllable + pass-through; for completeness only)")
    print("-" * 70)
    print(f"    processing_time  : {kpi['processing_time']:>16,.2f}  "
          f"<- SEPARATE time-based KPI (time units, NOT currency, NOT in landed)")
    print("=" * 70)


def main() -> None:
    try:
        params = Parameters.load(config.PARAMETERS_PATH)
    except FileNotFoundError:
        print(f"ERROR: no parameters at {config.PARAMETERS_PATH}. "
              "Run `python src/build_parameters.py` first.")
        sys.exit(1)
    res = run_baseline(params)
    print_report(res)


if __name__ == "__main__":
    main()
