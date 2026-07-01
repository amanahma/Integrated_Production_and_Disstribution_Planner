"""
build_parameters.py — orchestrates Phase 2.

Loads the saved ProblemInstance (does NOT rebuild it), synthesizes the full
parameter layer reproducibly (numpy seeded with config.RANDOM_SEED), saves a
Parameters object to data/processed/ as the new single source of truth, and
prints the sanity report.

Run:  python src/build_parameters.py

No optimization is performed — parameter generation only.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

import config
from instance import ProblemInstance
from parameters import Parameters
from utils import haversine


# --------------------------------------------------------------------------- #
# Perishable / family lookup from items.csv (family also lives on the instance)
# --------------------------------------------------------------------------- #
def load_item_attrs(skus: list) -> pd.DataFrame:
    """Return DataFrame indexed by sku with columns [family, perishable]."""
    items = pd.read_csv(config.ITEMS_CSV).set_index("item_nbr")
    rows = []
    for s in skus:
        if s in items.index:
            fam = str(items.at[s, "family"])
            per = int(items.at[s, "perishable"])
        else:
            fam, per = "UNKNOWN", 0
        rows.append((s, fam, per))
    return pd.DataFrame(rows, columns=["sku", "family", "perishable"]).set_index("sku")


# --------------------------------------------------------------------------- #
# Distance / transport-cost matrices
# --------------------------------------------------------------------------- #
def _dist_matrix(coords, sources, dests) -> pd.DataFrame:
    data = [[haversine(coords[s], coords[d]) for d in dests] for s in sources]
    return pd.DataFrame(data, index=sources, columns=dests)


def reorder_periods_to_midhorizon(inst: ProblemInstance) -> dict:
    """Relabel the periods into a unimodal slack->peak->slack planning sequence.

    Deterministic rule:
      1. Rank the existing periods by total demand (summed over SKU & region),
         largest first.
      2. Place the largest at PEAK_TARGET_INDEX. Then walk outward, alternating
         left/right of the peak, placing the next-largest periods. This yields a
         "tent": demand rises toward the interior peak and falls after it, with
         the smallest (slack) periods at the ends — so capacity slack accumulates
         BEFORE the peak and the overflow can be pre-built.
      3. Relabel the periods at their new positions P01..P12 (left-to-right =
         planning sequence) and remap the demand DataFrame's period column.

    Mutates inst.demand (period labels) and inst.periods in place. Returns a dict
    {new_label: old_label} for provenance. Chronology carries no modeling meaning
    (the window is an arbitrary 12-week sample), so this is a valid relabeling.
    """
    old_periods = list(inst.periods)
    n = len(old_periods)
    tot = inst.demand.groupby("period")["demand"].sum().reindex(old_periods)
    desc = sorted(old_periods, key=lambda l: float(tot[l]), reverse=True)

    peak_idx = config.PEAK_TARGET_INDEX
    pos = [None] * n
    pos[peak_idx] = desc[0]
    left, right = peak_idx - 1, peak_idx + 1
    go_left = True
    for r in desc[1:]:
        if (go_left and left >= 0) or right >= n:
            pos[left] = r
            left -= 1
        else:
            pos[right] = r
            right += 1
        go_left = not go_left

    new_labels = [f"P{i+1:02d}" for i in range(n)]
    old_to_new = {pos[i]: new_labels[i] for i in range(n)}      # old label -> new
    new_to_old = {new_labels[i]: pos[i] for i in range(n)}      # provenance

    inst.demand = inst.demand.assign(period=inst.demand["period"].map(old_to_new))
    inst.periods = new_labels
    return new_to_old


def check_prebuild_feasibility(demand_per_period, C_total, peak_label):
    """Verify the reordered instance admits a feasible pre-build of the peak.

    Returns (ok, info). Checks: peak is interior; cumulative capacity through the
    peak exceeds cumulative demand through the peak by >= the peak overflow.
    """
    periods = list(demand_per_period.index)
    n = len(periods)
    peak_pos = periods.index(peak_label)
    interior = (peak_pos != 0) and (peak_pos != n - 1)
    cum_cap = (peak_pos + 1) * C_total
    cum_dem = float(demand_per_period.iloc[:peak_pos + 1].sum())
    overflow = float(demand_per_period.loc[peak_label]) - C_total
    margin = cum_cap - cum_dem
    ok = interior and (margin >= overflow) and (overflow > 0)
    return ok, {
        "peak_pos": peak_pos, "interior": interior,
        "cum_cap": cum_cap, "cum_dem": cum_dem,
        "overflow": overflow, "margin": margin,
    }


def offset_facility_coords(inst: ProblemInstance) -> dict:
    """Return a coords dict where every plant/DC is shifted FACILITY_OFFSET_KM
    from its city centroid in a seeded random bearing, so no region has a
    zero-distance source. Region centroids are kept as-is.

    Uses an independent RNG (FACILITY_OFFSET_SEED) so the cost/processing draws
    in build() remain byte-identical to before this fix.
    """
    rng = np.random.default_rng(config.FACILITY_OFFSET_SEED)
    coords = dict(inst.coords)  # region centroids stay; facilities overwritten
    d = config.FACILITY_OFFSET_KM
    for fac in list(inst.plants) + list(inst.dcs):
        lat, lon = inst.coords[fac]
        bearing = rng.uniform(0.0, 2.0 * math.pi)
        dlat = (d * math.cos(bearing)) / 111.32
        dlon = (d * math.sin(bearing)) / (111.32 * math.cos(math.radians(lat)))
        coords[fac] = (lat + dlat, lon + dlon)
    return coords


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> Parameters:
    if not os.path.isfile(config.INSTANCE_PATH):
        print(f"ERROR: no instance at {config.INSTANCE_PATH}. "
              "Run `python src/build_instance.py` first.")
        sys.exit(1)

    inst = ProblemInstance.load(config.INSTANCE_PATH)
    skus = list(inst.skus)
    rng = np.random.default_rng(config.RANDOM_SEED)  # reproducible draws

    # --- period reordering (structural fix: peak -> mid-horizon) ------------
    reorder_provenance = None
    if config.REORDER_PEAK_TO_MIDHORIZON:
        reorder_provenance = reorder_periods_to_midhorizon(inst)
        seq = inst.demand.groupby("period")["demand"].sum().reindex(inst.periods)
        peak_label = seq.idxmax()
        print("[reorder] periods relabeled to slack->peak->slack planning order; "
              f"peak {peak_label} at index {list(inst.periods).index(peak_label)} "
              f"(was old {reorder_provenance[peak_label]})")

    # --- per-SKU attributes (family / perishable) ---------------------------
    attrs = load_item_attrs(skus)

    # --- demand aggregates --------------------------------------------------
    dem = inst.demand
    total_by_sku = dem.groupby("sku")["demand"].sum().reindex(skus).fillna(0.0)
    avg_demand = total_by_sku / len(inst.periods)            # d̄_k

    # === reproducible random draws (fixed order) ============================
    # 1) unit value per SKU
    unit_value = rng.uniform(config.UNIT_VALUE_MIN, config.UNIT_VALUE_MAX, size=len(skus))
    unit_value = pd.Series(unit_value, index=skus)
    perish = attrs["perishable"].reindex(skus).fillna(0).astype(int)
    unit_value = unit_value * np.where(perish.values == 1,
                                       config.PERISHABLE_VALUE_MULTIPLIER, 1.0)

    # --- holding, setup, production -----------------------------------------
    hold_mult = np.where(perish.values == 1, config.PERISHABLE_HOLDING_MULTIPLIER, 1.0)
    holding = config.HOLDING_RATE_WEEKLY * unit_value * hold_mult
    T = config.TARGET_REORDER_INTERVAL
    setup = (T ** 2 / 2.0) * holding * avg_demand            # part-period calibration
    production = config.PRODUCTION_COST_FRACTION * unit_value

    sku_table = pd.DataFrame({
        "family": attrs["family"].reindex(skus),
        "perishable": perish,
        "unit_value": unit_value,
        "total_demand": total_by_sku,
        "avg_demand": avg_demand,
        "holding_cost": holding,
        "setup_cost": setup,
        "production_cost": production,
    })
    sku_table.index.name = "sku"

    # --- capacity -----------------------------------------------------------
    demand_per_period = dem.groupby("period")["demand"].sum().reindex(inst.periods)
    D_mean = float(demand_per_period.mean())
    D_max = float(demand_per_period.max())
    peak_label = demand_per_period.idxmax()

    # Choose capacity tightness: start at config value, escalate ONLY as far as
    # needed to make the mid-horizon pre-build feasible, never so far that the
    # peak stops binding (D_max <= C_total would kill the pre-build behavior).
    tightness_candidates = [config.CAPACITY_TIGHTNESS] + list(
        config.CAPACITY_TIGHTNESS_ESCALATION)
    tightness = tightness_candidates[0]
    prebuild_ok, fb = False, {}
    for cand in tightness_candidates:
        C_try = cand * D_mean
        if D_max <= C_try:
            break  # would unbind the peak; do not use this or anything larger
        ok, info = check_prebuild_feasibility(demand_per_period, C_try, peak_label)
        tightness, prebuild_ok, fb = cand, ok, info
        if ok:
            break

    C_total = tightness * D_mean
    shares = config.PLANT_CAPACITY_SHARES
    if len(shares) != len(inst.plants):
        print(f"ERROR: PLANT_CAPACITY_SHARES has {len(shares)} entries but there "
              f"are {len(inst.plants)} plants.")
        sys.exit(1)
    plant_capacity = {p: C_total * sh for p, sh in zip(inst.plants, shares)}
    total_demand = float(demand_per_period.sum())
    horizon_cap = len(inst.periods) * C_total
    feasible = horizon_cap >= total_demand
    peak_gap = D_max - C_total

    # --- print the new sequence + pre-build feasibility check ----------------
    print("-" * 70)
    print(f"[capacity] CAPACITY_TIGHTNESS used = {tightness:.3f}  "
          f"C_total/period = {C_total:,.1f}  D_max = {D_max:,.1f}  "
          f"(peak binds: {D_max > C_total})")
    print("[demand] new per-period sequence (planning order):")
    for p, v in demand_per_period.items():
        mark = "  <-- PEAK" if p == peak_label else ""
        print(f"    {p}: {v:>12,.1f}{mark}")
    print(f"[pre-build feasibility] peak {peak_label} at index {fb['peak_pos']} "
          f"(interior={fb['interior']})")
    print(f"    cumulative capacity thru peak  = {fb['cum_cap']:>14,.1f}")
    print(f"    cumulative demand   thru peak  = {fb['cum_dem']:>14,.1f}")
    print(f"    peak overflow (D_max-C_total)  = {fb['overflow']:>14,.1f}")
    print(f"    early-slack margin             = {fb['margin']:>14,.1f}  "
          f"(must be >= overflow)")
    print(f"    PRE-BUILD FEASIBLE: {'PASS' if prebuild_ok else 'FAIL'}")
    if not prebuild_ok:
        print("    *** FAIL: not enough pre-peak slack even at max escalation. "
              "Revisit CAPACITY_TIGHTNESS_ESCALATION. ***")
        sys.exit(1)
    print("-" * 70)

    # --- transport matrices (OFFSET facility coords => all distances > 0) ----
    co = offset_facility_coords(inst)
    dist_plant_region = _dist_matrix(co, inst.plants, inst.regions)
    dist_dc_region = _dist_matrix(co, inst.dcs, inst.regions)
    dist_plant_dc = _dist_matrix(co, inst.plants, inst.dcs)
    k = config.COST_PER_UNIT_KM
    cost_plant_region = dist_plant_region * k
    cost_dc_region = dist_dc_region * k
    cost_plant_dc = dist_plant_dc * k

    # --- baseline aggregates for controllable-cost calibration --------------
    # setup_total: a setup per (sku, period) with positive demand (lot-for-lot)
    sp = dem.groupby(["sku", "period"], observed=True)["demand"].sum()
    positive = sp[sp > 0]
    setup_total = float(sum(sku_table.at[s, "setup_cost"] for s, _ in positive.index))
    # transport_base: rate-independent volume*distance using nearest source
    region_demand = dem.groupby("region")["demand"].sum()
    nearest_dist = pd.concat([dist_plant_region, dist_dc_region]).min(axis=0)  # per region
    transport_base = float((nearest_dist.reindex(inst.regions)
                            * region_demand.reindex(inst.regions)).sum())
    transport_cost_baseline = k * transport_base
    holding_baseline = 0.0  # lot-for-lot carries nothing
    controllable_baseline = setup_total + holding_baseline + transport_cost_baseline
    transport_share = (transport_cost_baseline / controllable_baseline
                       if controllable_baseline else 0.0)
    # recommended rate to hit the exact target share (holding=0):
    tgt = config.TARGET_TRANSPORT_SHARE
    recommended_k = (tgt * setup_total / (transport_base * (1.0 - tgt))
                     if transport_base else 0.0)

    # --- line x family processing-cost matrix -------------------------------
    families = sorted(sku_table["family"].unique().tolist())
    F = len(families)
    lines = [f"LINE_{i+1}" for i in range(F)]               # square assignment
    base_time = rng.uniform(config.BASE_TIME_MIN, config.BASE_TIME_MAX, size=F)
    line_eff = rng.uniform(config.LINE_EFFICIENCY_MIN, config.LINE_EFFICIENCY_MAX, size=F)
    noise = rng.uniform(-config.PROCESSING_NOISE, config.PROCESSING_NOISE, size=(F, F))
    proc = (base_time[None, :] * line_eff[:, None]) * (1.0 + noise)
    processing_cost = pd.DataFrame(proc, index=lines, columns=families)

    params = Parameters(
        instance=inst,
        coords=co,
        sku_table=sku_table,
        demand_per_period=demand_per_period,
        D_mean=D_mean, D_max=D_max, C_total=C_total,
        plant_capacity=plant_capacity,
        feasible=bool(feasible), peak_gap=peak_gap,
        dist_plant_region=dist_plant_region,
        dist_dc_region=dist_dc_region,
        dist_plant_dc=dist_plant_dc,
        cost_plant_region=cost_plant_region,
        cost_dc_region=cost_dc_region,
        cost_plant_dc=cost_plant_dc,
        families=families, lines=lines,
        base_time=dict(zip(families, base_time)),
        line_efficiency=dict(zip(lines, line_eff)),
        processing_cost=processing_cost,
        meta={
            "random_seed": config.RANDOM_SEED,
            "capacity_tightness": tightness,
            "capacity_tightness_config": config.CAPACITY_TIGHTNESS,
            "reorder_peak_to_midhorizon": config.REORDER_PEAK_TO_MIDHORIZON,
            "peak_target_index": config.PEAK_TARGET_INDEX,
            "period_reorder_new_to_old": reorder_provenance,
            "peak_label": peak_label,
            "prebuild_feasibility": fb,
            "plant_capacity_shares": list(shares),
            "target_reorder_interval": T,
            "annual_holding_rate": config.ANNUAL_HOLDING_RATE,
            "unit_capacity_consumption": config.UNIT_CAPACITY_CONSUMPTION,
            "cost_per_unit_km": k,
            "facility_offset_km": config.FACILITY_OFFSET_KM,
            "facility_offset_seed": config.FACILITY_OFFSET_SEED,
            "target_transport_share": config.TARGET_TRANSPORT_SHARE,
            "recommended_cost_per_unit_km": recommended_k,
            "baseline": {
                "setup_cost": setup_total,
                "holding_cost": holding_baseline,
                "transport_cost": transport_cost_baseline,
                "transport_base_rate1": transport_base,
                "transport_share": transport_share,
            },
            "draw_order": ["unit_value[N]", "base_time[F]", "line_efficiency[F]",
                           "processing_noise[FxF]",
                           "(facility bearings: separate RNG seed)"],
        },
    )
    return params


def main() -> None:
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    params = build()
    params.save(config.PARAMETERS_PATH)
    print(f"Saved parameters -> {config.PARAMETERS_PATH}\n")
    params.sanity_report()


if __name__ == "__main__":
    main()
