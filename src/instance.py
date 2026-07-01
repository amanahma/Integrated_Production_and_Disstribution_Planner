"""
instance.py — the ProblemInstance, the single source of truth all later phases load.

Holds the REAL demand instance (SKUs, regions, periods) plus the SYNTHESIZED
facilities (plants, DCs) and their coordinates. Serializable and self-describing.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ProblemInstance:
    """Serialized problem instance for PlanFlow.

    Attributes
    ----------
    skus : list[int]
        Selected SKU ids (Favorita item_nbr).
    regions : list[str]
        Demand region (city) names.
    periods : list[str]
        12 weekly period labels (P01..P12).
    plants : list[str]
        Synthesized plant ids.
    dcs : list[str]
        Synthesized DC ids.
    demand : pd.DataFrame
        Tidy demand table with columns [sku, region, period, demand].
        One row per (SKU, region, period) cell.
    coords : dict[str, tuple[float, float]]
        Maps every region / plant / DC id to (lat, lon).
    sku_family : dict[int, str]
        Maps each SKU to its item family (kept for Phase 2 line assignment).
    meta : dict
        Free-form provenance / build metadata.
    """

    skus: list
    regions: list
    periods: list
    plants: list
    dcs: list
    demand: pd.DataFrame
    coords: dict
    sku_family: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Pickle the whole instance to `path` (round-trips cleanly)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "ProblemInstance":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} did not contain a ProblemInstance")
        return obj

    # ------------------------------------------------------------------ #
    # Convenience views
    # ------------------------------------------------------------------ #
    def demand_pivot(self) -> pd.DataFrame:
        """(SKU, region) rows x period columns matrix of demand."""
        return (
            self.demand.pivot_table(
                index=["sku", "region"], columns="period", values="demand",
                aggfunc="sum", fill_value=0.0,
            )
            .reindex(columns=self.periods, fill_value=0.0)
        )

    def demand_by_period(self) -> pd.Series:
        return self.demand.groupby("period")["demand"].sum().reindex(self.periods)

    def demand_by_region(self) -> pd.Series:
        return self.demand.groupby("region")["demand"].sum().reindex(self.regions)

    # ------------------------------------------------------------------ #
    # Sanity summary
    # ------------------------------------------------------------------ #
    def summary(self) -> None:
        total = float(self.demand["demand"].sum())
        n_cells = len(self.demand)
        expected_cells = len(self.skus) * len(self.regions) * len(self.periods)
        n_zero = int((self.demand["demand"] == 0).sum())

        print("=" * 64)
        print("ProblemInstance summary")
        print("=" * 64)
        print(f"SKUs            : {len(self.skus)}")
        print(f"Regions         : {len(self.regions)}  -> {self.regions}")
        print(f"Periods         : {len(self.periods)}  -> {self.periods}")
        print(f"Plants          : {len(self.plants)}  -> {self.plants}")
        print(f"DCs             : {len(self.dcs)}  -> {self.dcs}")
        print(f"Families         : {len(set(self.sku_family.values()))}")
        print("-" * 64)
        print(f"(SKU,region,period) cells : {n_cells} (expected {expected_cells})")
        print(f"Zero-demand cells          : {n_zero} "
              f"({100.0 * n_zero / max(n_cells, 1):.1f}%)")
        print(f"Total demand               : {total:,.1f} units")
        print(f"Mean demand per cell       : {total / max(n_cells, 1):,.2f}")
        print("-" * 64)
        print("Demand per period:")
        for p, v in self.demand_by_period().items():
            print(f"  {p}: {v:>14,.1f}")
        print("-" * 64)
        print("Demand per region:")
        for r, v in self.demand_by_region().sort_values(ascending=False).items():
            print(f"  {r:<16}: {v:>14,.1f}")
        print("=" * 64)
