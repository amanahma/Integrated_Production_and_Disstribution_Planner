"""
build_instance.py — orchestrates Phase 0.

Reads the raw Favorita CSVs, samples/aggregates to weekly regional demand,
attaches synthesized facilities + coordinates, constructs a ProblemInstance,
saves it to data/processed/, and prints a summary.

Run:  python src/build_instance.py

train.csv (~5GB / ~125M rows) is NEVER loaded fully — it is read in chunks,
filtered to the 12-week window on the fly (string date compare), negatives
clipped, and aggregated immediately. Reading stops early once the file's dates
pass the window (train.csv is date-sorted ascending).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

import config
from instance import ProblemInstance


# --------------------------------------------------------------------------- #
# Raw-file presence check (graceful exit with instructions)
# --------------------------------------------------------------------------- #
def check_raw_files() -> None:
    missing = []
    for label, path in [("train.csv", config.TRAIN_CSV), ("items.csv", config.ITEMS_CSV)]:
        if not os.path.isfile(path):
            missing.append((label, path))
    if missing:
        print("ERROR: required raw data file(s) are missing.\n")
        for label, path in missing:
            print(f"  - place '{label}' at: {path}")
        print("\nDownload the Corporación Favorita dataset and copy the CSVs into "
              f"{config.RAW_DIR}, then re-run.")
        sys.exit(1)

    if not os.path.isfile(config.STORES_CSV):
        print(f"NOTE: '{config.STORES_CSV}' not found — using the canonical embedded "
              "Favorita stores table (config.STORES_FALLBACK).")
        print("      Drop a real stores.csv into data/raw/ to override it.\n")


# --------------------------------------------------------------------------- #
# Store -> (city, state) mapping
# --------------------------------------------------------------------------- #
def load_store_city() -> pd.DataFrame:
    """Return DataFrame indexed by store_nbr with a 'city' column."""
    if os.path.isfile(config.STORES_CSV):
        stores = pd.read_csv(config.STORES_CSV)
        stores = stores[["store_nbr", "city"]].copy()
    else:
        stores = pd.DataFrame(
            [(s, c) for s, (c, _state) in config.STORES_FALLBACK.items()],
            columns=["store_nbr", "city"],
        )
    stores["store_nbr"] = stores["store_nbr"].astype("int32")
    return stores.set_index("store_nbr")


# --------------------------------------------------------------------------- #
# Chunked scan of train.csv -> weekly (item, store) demand in the window
# --------------------------------------------------------------------------- #
def scan_window_demand() -> pd.DataFrame:
    """Aggregate clipped weekly demand per (item_nbr, store_nbr, period)."""
    start = config.WINDOW_START
    end_excl = config.WINDOW_END_EXCL
    start_ts = pd.Timestamp(start)

    print(f"Scanning {os.path.basename(config.TRAIN_CSV)} for window "
          f"[{start}, {end_excl}) in chunks of {config.CHUNK_SIZE:,} rows ...")

    parts = []
    rows_seen = 0
    reader = pd.read_csv(
        config.TRAIN_CSV,
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        dtype={"store_nbr": "int32", "item_nbr": "int32", "unit_sales": "float32"},
        chunksize=config.CHUNK_SIZE,
    )
    for i, chunk in enumerate(reader):
        rows_seen += len(chunk)
        # 'date' is a 'YYYY-MM-DD' string -> lexicographic compare is valid.
        cmin = chunk["date"].iloc[0]
        cmax = chunk["date"].iloc[-1]

        # Early stop: file is date-sorted ascending; once min date >= end, done.
        if cmin >= end_excl:
            print(f"  chunk {i}: dates from {cmin} >= window end -> stopping early.")
            break

        sel = chunk[(chunk["date"] >= start) & (chunk["date"] < end_excl)]
        if not sel.empty:
            sel = sel.copy()
            sel["unit_sales"] = sel["unit_sales"].clip(lower=0.0)  # returns -> 0
            dts = pd.to_datetime(sel["date"])
            sel["period_idx"] = ((dts - start_ts).dt.days // 7).astype("int16")
            agg = (
                sel.groupby(["item_nbr", "store_nbr", "period_idx"], observed=True)
                ["unit_sales"].sum().reset_index()
            )
            parts.append(agg)

        if i % 5 == 0:
            print(f"  chunk {i}: rows_seen={rows_seen:,}  last_date={cmax}  "
                  f"kept_groups={sum(len(p) for p in parts):,}")

    if not parts:
        print("ERROR: no rows found in the chosen window. Adjust WINDOW_START in "
              "config.py to a date range present in train.csv.")
        sys.exit(1)

    out = (
        pd.concat(parts, ignore_index=True)
        .groupby(["item_nbr", "store_nbr", "period_idx"], observed=True)
        ["unit_sales"].sum().reset_index()
    )
    print(f"Scan complete. {rows_seen:,} rows read; "
          f"{len(out):,} (item,store,week) groups in window.")
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> ProblemInstance:
    check_raw_files()

    raw = scan_window_demand()                      # item_nbr, store_nbr, period_idx, unit_sales
    store_city = load_store_city()                  # store_nbr -> city

    # --- select top-N SKUs by total unit sales in window --------------------
    item_totals = raw.groupby("item_nbr")["unit_sales"].sum().sort_values(ascending=False)
    skus = item_totals.head(config.N_SKUS).index.astype(int).tolist()
    print(f"\nSelected {len(skus)} SKUs (top by window unit sales).")

    raw = raw[raw["item_nbr"].isin(skus)].copy()

    # --- attach city, drop stores with no known city -----------------------
    raw["city"] = raw["store_nbr"].map(store_city["city"])
    unknown = raw["city"].isna().sum()
    if unknown:
        print(f"  warning: {unknown} rows had unknown store->city; dropped.")
        raw = raw.dropna(subset=["city"])

    # --- select top-N regions (cities) by total demand of selected SKUs -----
    city_totals = raw.groupby("city")["unit_sales"].sum().sort_values(ascending=False)
    regions = city_totals.head(config.N_REGIONS).index.tolist()
    print(f"Selected {len(regions)} regions (top cities by demand): {regions}")

    raw = raw[raw["city"].isin(regions)].copy()

    # --- aggregate store-level -> region (city) level demand ----------------
    demand = (
        raw.groupby(["item_nbr", "city", "period_idx"], observed=True)["unit_sales"]
        .sum().reset_index()
    )

    # --- densify to a full (SKU x region x period) grid (zeros filled) -------
    idx = pd.MultiIndex.from_product(
        [skus, regions, range(config.N_PERIODS)],
        names=["item_nbr", "city", "period_idx"],
    )
    demand = (
        demand.set_index(["item_nbr", "city", "period_idx"])
        .reindex(idx, fill_value=0.0)
        .reset_index()
    )
    demand["period"] = demand["period_idx"].map(
        {i: lbl for i, lbl in enumerate(config.PERIOD_LABELS)}
    )
    demand = demand.rename(columns={"item_nbr": "sku", "city": "region",
                                    "unit_sales": "demand"})
    demand = demand[["sku", "region", "period", "demand"]].sort_values(
        ["sku", "region", "period"]).reset_index(drop=True)

    # --- SKU -> family (from items.csv; kept for Phase 2) -------------------
    items = pd.read_csv(config.ITEMS_CSV)
    fam_map = items.set_index("item_nbr")["family"].to_dict()
    sku_family = {int(s): str(fam_map.get(s, "UNKNOWN")) for s in skus}

    # --- coordinates for regions + synthesized plants/DCs -------------------
    coords: dict[str, tuple[float, float]] = {}
    for region in regions:
        if region not in config.CITY_COORDS:
            print(f"  warning: no coordinates for region city '{region}'.")
        coords[region] = config.CITY_COORDS.get(region, (0.0, 0.0))
    for pid, city in config.PLANTS.items():
        coords[pid] = config.CITY_COORDS[city]
    for did, city in config.DCS.items():
        coords[did] = config.CITY_COORDS[city]

    inst = ProblemInstance(
        skus=skus,
        regions=regions,
        periods=list(config.PERIOD_LABELS),
        plants=list(config.PLANTS.keys()),
        dcs=list(config.DCS.keys()),
        demand=demand,
        coords=coords,
        sku_family=sku_family,
        meta={
            "window_start": config.WINDOW_START,
            "window_end_excl": config.WINDOW_END_EXCL,
            "n_skus": config.N_SKUS,
            "n_regions": config.N_REGIONS,
            "n_periods": config.N_PERIODS,
            "plant_cities": dict(config.PLANTS),
            "dc_cities": dict(config.DCS),
            "stores_source": ("stores.csv" if os.path.isfile(config.STORES_CSV)
                              else "config.STORES_FALLBACK"),
        },
    )
    return inst


def main() -> None:
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    inst = build()
    inst.save(config.INSTANCE_PATH)
    # also drop a tidy parquet of demand for quick inspection / notebooks
    inst.demand.to_parquet(config.DEMAND_PARQUET, index=False)
    print(f"\nSaved instance -> {config.INSTANCE_PATH}")
    print(f"Saved demand    -> {config.DEMAND_PARQUET}\n")
    inst.summary()


if __name__ == "__main__":
    main()
