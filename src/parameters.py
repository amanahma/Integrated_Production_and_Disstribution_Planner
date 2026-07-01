"""
parameters.py — the Parameters container (Phase 2).

Holds the Phase-0 ProblemInstance plus every SYNTHESIZED parameter the four OR
models will need: per-SKU unit value / holding / setup / production cost, plant
capacities, distance & transport-cost matrices, and the line x family
processing-cost matrix.

All synthesized values are reproducible (numpy seeded with config.RANDOM_SEED in
build_parameters.py). No optimization is performed here.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import pandas as pd

from instance import ProblemInstance


@dataclass
class Parameters:
    """Derived parameter layer; the new single source of truth for later phases."""

    instance: ProblemInstance

    # Full coordinate map used for all distances: region centroids (from the
    # instance) + OFFSET facility coordinates (plants/DCs shifted off centroids).
    coords: dict

    # Per-SKU table: index=sku, cols = family, perishable, unit_value,
    # avg_demand, total_demand, holding_cost, setup_cost, production_cost
    sku_table: pd.DataFrame

    # Capacity
    demand_per_period: pd.Series           # D_t over the 12 periods
    D_mean: float                          # D̄
    D_max: float                           # D_max
    C_total: float                         # system capacity per period
    plant_capacity: dict                   # plant_id -> capacity per period
    feasible: bool
    peak_gap: float                        # D_max - C_total (positive => binding)

    # Transport (labeled DataFrames; rows=sources, cols=destinations)
    dist_plant_region: pd.DataFrame
    dist_dc_region: pd.DataFrame
    dist_plant_dc: pd.DataFrame
    cost_plant_region: pd.DataFrame
    cost_dc_region: pd.DataFrame
    cost_plant_dc: pd.DataFrame

    # Assignment
    families: list
    lines: list
    base_time: dict                        # family -> base processing time
    line_efficiency: dict                  # line -> efficiency multiplier
    processing_cost: pd.DataFrame          # rows=lines, cols=families

    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Convenience accessors (dicts keyed by sku)
    # ------------------------------------------------------------------ #
    @property
    def unit_value(self) -> dict:
        return self.sku_table["unit_value"].to_dict()

    @property
    def holding_cost(self) -> dict:
        return self.sku_table["holding_cost"].to_dict()

    @property
    def setup_cost(self) -> dict:
        return self.sku_table["setup_cost"].to_dict()

    @property
    def production_cost(self) -> dict:
        return self.sku_table["production_cost"].to_dict()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Parameters":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} did not contain a Parameters object")
        return obj

    # ------------------------------------------------------------------ #
    # Sanity report
    # ------------------------------------------------------------------ #
    def sanity_report(self) -> None:
        inst = self.instance
        total_demand = float(self.demand_per_period.sum())

        print("=" * 70)
        print("Parameters sanity report")
        print("=" * 70)
        print(f"SKUs={len(inst.skus)}  regions={len(inst.regions)}  "
              f"periods={len(inst.periods)}  plants={len(inst.plants)}  "
              f"DCs={len(inst.dcs)}")
        print(f"Families={len(self.families)}  Lines={len(self.lines)}  "
              f"(square assignment: {len(self.lines)}x{len(self.families)})")
        print("-" * 70)

        # --- capacity / tightness diagnostics -------------------------------
        print("CAPACITY / TIGHTNESS")
        print(f"  Total demand (horizon)      : {total_demand:>16,.1f} units")
        print(f"  Mean demand / period  (D̄)   : {self.D_mean:>16,.1f}")
        print(f"  Peak demand / period (D_max) : {self.D_max:>16,.1f}")
        print(f"  System capacity / period (C_total) : {self.C_total:>11,.1f}")
        print(f"  Capacity tightness          : {self.meta.get('capacity_tightness'):.3f} "
              f"(C_total / D̄)")
        print("  Per-plant capacity / period :")
        for p, cap in self.plant_capacity.items():
            share = cap / self.C_total if self.C_total else 0.0
            print(f"      {p:<18}: {cap:>14,.1f}  ({share*100:4.1f}%)")
        horizon_cap = len(inst.periods) * self.C_total
        print(f"  Horizon capacity (12*C_total): {horizon_cap:>14,.1f}")
        print(f"  Peak-period gap (D_max - C_total): {self.peak_gap:>11,.1f} "
              f"({'peak BINDS -> must pre-build' if self.peak_gap > 0 else 'peak fits'})")
        feas = "PASS" if self.feasible else "FAIL"
        print(f"  Feasibility (horizon_cap >= total_demand): {feas}")
        if not self.feasible:
            print("  >>> INFEASIBLE: raise CAPACITY_TIGHTNESS in config.py and rebuild.")
        print("-" * 70)

        # --- distance diagnostics + strictly-positive assertion -------------
        print("DISTANCE MATRICES (km) — offset facilities, must be strictly > 0")
        all_min = float("inf")
        for name, mat in [("plant->region", self.dist_plant_region),
                          ("DC->region", self.dist_dc_region),
                          ("plant->DC", self.dist_plant_dc)]:
            mn, mean, mx = float(mat.values.min()), float(mat.values.mean()), float(mat.values.max())
            all_min = min(all_min, mn)
            assert mn > 0.0, f"{name}: min distance {mn} is not > 0 (facility on a centroid?)"
            print(f"  {name:<14}: min={mn:8.1f}  mean={mean:8.1f}  max={mx:8.1f}  "
                  f"[min>0 OK]")
        print(f"  Facility offset magnitude    : {self.meta.get('facility_offset_km')} km "
              f"(seeded bearings)")
        print(f"  Global min distance (all)    : {all_min:.1f} km  "
              f"-> ALL DISTANCES STRICTLY POSITIVE")
        print("-" * 70)

        # --- baseline controllable-cost calibration -------------------------
        bl = self.meta.get("baseline", {})
        setup_b = bl.get("setup_cost", 0.0)
        hold_b = bl.get("holding_cost", 0.0)
        trans_b = bl.get("transport_cost", 0.0)
        controllable = setup_b + hold_b + trans_b
        print("BASELINE CONTROLLABLE-COST CALIBRATION")
        print(f"  COST_PER_UNIT_KM (calibrated): {self.meta.get('cost_per_unit_km')}")
        print(f"  Controllable = setup + holding + transport")
        if controllable > 0:
            print(f"    setup     : {setup_b:>16,.2f}  ({setup_b/controllable*100:5.1f}%)")
            print(f"    holding   : {hold_b:>16,.2f}  ({hold_b/controllable*100:5.1f}%)")
            print(f"    transport : {trans_b:>16,.2f}  ({trans_b/controllable*100:5.1f}%)")
            print(f"    CONTROLLABLE TOTAL : {controllable:>11,.2f}")
            share = trans_b / controllable * 100
            tgt = self.meta.get("target_transport_share", 0.09) * 100
            ok = "OK" if 5.0 <= share <= 15.0 else "OUT OF 5-15% BAND"
            print(f"  >> Transport share of controllable: {share:.2f}%  "
                  f"(target ~{tgt:.0f}%)  [{ok}]")
        rec = self.meta.get("recommended_cost_per_unit_km")
        if rec is not None:
            print(f"  (recommended COST_PER_UNIT_KM for exact target: {rec:.4f})")
        print("-" * 70)

        # --- per-SKU calibration sample -------------------------------------
        print("PER-SKU CALIBRATION (sample of 8 SKUs)")
        cols = ["family", "perishable", "unit_value", "avg_demand",
                "holding_cost", "setup_cost", "production_cost"]
        sample = self.sku_table[cols].head(8)
        with pd.option_context("display.float_format", lambda v: f"{v:,.3f}",
                               "display.max_columns", None, "display.width", 120):
            print(sample.to_string())
        print("  ...")
        print("  Aggregate ranges:")
        for c in ["unit_value", "holding_cost", "setup_cost", "production_cost"]:
            s = self.sku_table[c]
            print(f"    {c:<16}: min={s.min():,.3f}  mean={s.mean():,.3f}  "
                  f"max={s.max():,.3f}")
        n_perish = int(self.sku_table["perishable"].sum())
        print(f"  Perishable SKUs: {n_perish} / {len(self.sku_table)}")
        print("-" * 70)

        # --- processing matrix peek -----------------------------------------
        print("PROCESSING COST MATRIX (lines x families)")
        with pd.option_context("display.float_format", lambda v: f"{v:,.2f}",
                               "display.max_columns", None, "display.width", 160):
            print(self.processing_cost.to_string())
        print("=" * 70)
