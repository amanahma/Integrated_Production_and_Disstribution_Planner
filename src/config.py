"""
config.py — Single source of truth for all PlanFlow knobs (Phase 0).

Every tunable number for the data instance and the naive baseline lives here so
later phases can change the problem size / geography without touching logic.

Phase 0 scope: data instance + documented baseline only. No optimization models.
Cost knobs that belong to Phase 2 (setup, holding) are present as clearly marked
PLACEHOLDER constants with TODOs; they get calibrated for real in Phase 2.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = parent of this file's directory (src/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Required raw source files (user places these manually in data/raw/).
TRAIN_CSV = os.path.join(RAW_DIR, "train.csv")
ITEMS_CSV = os.path.join(RAW_DIR, "items.csv")
STORES_CSV = os.path.join(RAW_DIR, "stores.csv")

# Where the serialized ProblemInstance is written / read.
INSTANCE_PATH = os.path.join(PROCESSED_DIR, "instance.pkl")
DEMAND_PARQUET = os.path.join(PROCESSED_DIR, "demand.parquet")

# --------------------------------------------------------------------------- #
# Instance size (lock here, easy to change)
# --------------------------------------------------------------------------- #
N_SKUS = 30          # top-N highest-selling items become SKUs
N_PERIODS = 12       # weekly buckets
N_REGIONS = 10       # top-N cities (by demand) become demand regions

# 12 consecutive weekly periods. WINDOW_START must be a Monday for clean weeks.
# 2016 window with good coverage; lexicographic (YYYY-MM-DD) compare is valid.
WINDOW_START = "2016-01-04"   # Monday
WINDOW_DAYS = N_PERIODS * 7   # 84 days -> 12 weeks
# WINDOW_END is exclusive (the day after the last day of week 12).
import datetime as _dt
_start = _dt.date.fromisoformat(WINDOW_START)
WINDOW_END_EXCL = (_start + _dt.timedelta(days=WINDOW_DAYS)).isoformat()  # "2016-03-28"

# Chunked reading of the ~5GB train.csv.
CHUNK_SIZE = 3_000_000

# Period labels P01..P12
PERIOD_LABELS = [f"P{i:02d}" for i in range(1, N_PERIODS + 1)]

# --------------------------------------------------------------------------- #
# Synthesized facilities (NOT in the dataset). Placed at real city coordinates.
# Chosen to be geographically spread across Ecuador.
# --------------------------------------------------------------------------- #
# plant_id -> city
PLANTS = {
    "PLANT_QUITO": "Quito",        # northern Andes
    "PLANT_GUAYAQUIL": "Guayaquil",  # coastal south-west
    "PLANT_CUENCA": "Cuenca",      # southern Andes
}
# dc_id -> city
DCS = {
    "DC_SANTO_DOMINGO": "Santo Domingo",  # central, near coast
    "DC_AMBATO": "Ambato",                # central Andes
    "DC_MACHALA": "Machala",              # far south coast
}

# --------------------------------------------------------------------------- #
# Geocoding — hardcoded approximate (lat, lon) for every city we might use.
# Approximate real coordinates for Ecuadorian cities (no live API).
# Documented as APPROXIMATE in ASSUMPTIONS.md.
# --------------------------------------------------------------------------- #
CITY_COORDS = {
    "Quito": (-0.1807, -78.4678),
    "Guayaquil": (-2.1709, -79.9224),
    "Cuenca": (-2.9006, -79.0045),
    "Santo Domingo": (-0.2522, -79.1719),
    "Ambato": (-1.2491, -78.6168),
    "Machala": (-3.2581, -79.9554),
    "Latacunga": (-0.9333, -78.6167),
    "Riobamba": (-1.6635, -78.6546),
    "Ibarra": (0.3517, -78.1223),
    "Manta": (-0.9676, -80.7089),
    "Loja": (-3.9931, -79.2042),
    "Esmeraldas": (0.9682, -79.6517),
    "Salinas": (-2.2139, -80.9583),
    "Babahoyo": (-1.8019, -79.5346),
    "Quevedo": (-1.0286, -79.4636),
    "Cayambe": (0.0403, -78.1456),
    "Daule": (-1.8667, -79.9833),
    "Playas": (-2.6333, -80.3833),
    "Libertad": (-2.2333, -80.9000),
    "Guaranda": (-1.5928, -79.0000),
    "Puyo": (-1.4924, -77.9961),
    "El Carmen": (-0.2667, -79.4333),
}

# --------------------------------------------------------------------------- #
# Cost knobs
# --------------------------------------------------------------------------- #
EARTH_RADIUS_KM = 6371.0     # haversine

# Transport: placeholder cost per unit per km (currency / unit / km).
# Used by the naive shipping baseline. Refined / kept in Phase 2.
COST_PER_KM = 0.01

# --- Phase 2 placeholders (SUPERSEDED) ---------------------------------------
# Kept only so old Phase-0 baseline code paths still import cleanly. Phase 2
# replaces these with calibrated, per-SKU values (see Parameters below). Not used
# by the Phase-2 baseline.
SETUP_COST_PLACEHOLDER = 100.0
HOLDING_COST_PLACEHOLDER = 0.10
PRODUCTION_COST_PLACEHOLDER = 1.0

# =========================================================================== #
# PHASE 2 — Parameter Engineering knobs
# All synthesized cost / capacity / processing-time parameters are derived from
# these constants. Every random draw uses RANDOM_SEED for reproducibility.
# =========================================================================== #
RANDOM_SEED = 42

# Where the Parameters object (new single source of truth) is written / read.
PARAMETERS_PATH = os.path.join(PROCESSED_DIR, "parameters.pkl")

# --- 1. Unit value per SKU (synthetic price; Favorita has none) --------------
UNIT_VALUE_MIN = 1.0     # currency units
UNIT_VALUE_MAX = 50.0
# Perishables tend to be lower-ticket fast-movers; bias their value range down.
# Set to 1.0 to disable the bias. Applied as a multiplier on the drawn value.
PERISHABLE_VALUE_MULTIPLIER = 0.8

# --- 2. Holding cost ---------------------------------------------------------
ANNUAL_HOLDING_RATE = 0.22                       # 22% / year of unit value
HOLDING_RATE_WEEKLY = ANNUAL_HOLDING_RATE / 52.0  # per weekly period
# Perishables incur extra carrying cost (spoilage). 1.0 disables.
PERISHABLE_HOLDING_MULTIPLIER = 2.0

# --- 3. Setup cost calibration (part-period / EOQ logic) ---------------------
# A_k = (T^2 / 2) * h_k * dbar_k  ->  Wagner-Whitin optimal interval ~ T periods.
TARGET_REORDER_INTERVAL = 3   # periods (T)

# --- 4. Production cost ------------------------------------------------------
PRODUCTION_COST_FRACTION = 0.5  # p_k = fraction * unit_value_k

# --- 5. Plant capacity ------------------------------------------------------
# a_k = 1: one unit of any product consumes one unit of plant capacity.
UNIT_CAPACITY_CONSUMPTION = 1.0
CAPACITY_TIGHTNESS = 1.1        # C_total per period = tightness * mean demand/period
# Unequal split across the 3 plants -> a clear bottleneck plant. Must sum to 1.0
# and align with the plant ordering in the instance (PLANTS dict order).
# Calibrated in Phase 4 follow-up to track demand-gravity (Phase 4 diagnostic:
# gravity 75.7 / 16.2 / 8.1 by nearest plant for Quito/Guayaquil/Cuenca).
# Capacity kept intentionally tighter than gravity at Quito (55% cap vs 75.7%
# gravity) so a real bottleneck persists for the assignment model, but the
# original 40/35/25 split was checked to cause a 5.3% controllable-cost
# regression by forcing expensive cross-plant shipping to Quito.
PLANT_CAPACITY_SHARES = [0.55, 0.28, 0.17]

# Period reordering (Phase 3 structural fix). The 12 periods are an arbitrary
# sampling window; chronological order carries NO modeling meaning. The raw data
# happens to put the peak-demand period at index 0, where it cannot be pre-built
# (zero opening inventory, no earlier periods) -> forced shortage. Reorder the
# periods into a unimodal "slack-rises-to-peak-then-falls" sequence with the peak
# at PEAK_TARGET_INDEX, converting that impossible overflow into a genuine,
# feasible mid-horizon pre-build problem. Periods are relabeled P01..P12 in the
# new planning order. If pre-build is infeasible at CAPACITY_TIGHTNESS, the build
# escalates tightness through CAPACITY_TIGHTNESS_ESCALATION (only as far as
# needed, never so far that D_max <= C_total).
REORDER_PEAK_TO_MIDHORIZON = True
PEAK_TARGET_INDEX = 6                       # 0-indexed mid-horizon slot for the peak
CAPACITY_TIGHTNESS_ESCALATION = [1.15, 1.20]  # tried in order only if pre-build fails

# --- 6. Transport cost ------------------------------------------------------
# Facilities must NOT sit exactly on a region centroid (that made some region
# distances zero). Each facility is shifted a fixed magnitude in a seeded random
# bearing from its city centroid, so every plant->region / DC->region /
# plant->DC distance is strictly positive.
FACILITY_OFFSET_KM = 35.0          # fixed offset magnitude (in the 20-50 km band)
FACILITY_OFFSET_SEED = RANDOM_SEED + 1  # independent of the cost/processing draws

# COST_PER_UNIT_KM is CALIBRATED, not guessed. Target: baseline transport cost
# should be ~8-10% of controllable cost (setup + holding + transport). With the
# offset geometry the calibrated value below lands transport at ~9% of
# controllable (verified in the sanity report's transport-share line).
# Calibration formula (holding=0 in the lot-for-lot baseline):
#   rate = TARGET_TRANSPORT_SHARE * setup_total
#          / (transport_base * (1 - TARGET_TRANSPORT_SHARE))
# where transport_base = sum_region (min source distance) * region_demand.
TARGET_TRANSPORT_SHARE = 0.09      # midpoint of the 8-10% target band
# CALIBRATED so baseline transport ~= 9% of controllable cost. Derived from the
# formula above (setup_total=4,014,500.86, transport_base=215,739,240) -> 0.00184.
# The sanity report's transport-share line confirms ~9.0% (inside the 5-15% band).
COST_PER_UNIT_KM = 0.00184         # CALIBRATED (target ~9% transport share)

# --- 7. Line x family processing-cost matrix --------------------------------
BASE_TIME_MIN = 1.0            # base processing time per family (seeded)
BASE_TIME_MAX = 5.0
LINE_EFFICIENCY_MIN = 0.8      # per-line multiplier (seeded)
LINE_EFFICIENCY_MAX = 1.2
PROCESSING_NOISE = 0.05        # +/- 5% noise so optimal assignment != identity

# =========================================================================== #
# PHASE 3 — OR model knobs (Wagner-Whitin DP, transportation LP, CLSP MILP)
# =========================================================================== #
# Penalty per unit of unmet demand (shortage/slack) in the MILP. Must be >> any
# real unit cost so shortages are avoided whenever feasible, while never letting
# infeasibility hard-fail the solve. ~100x the max possible unit value
# (UNIT_VALUE_MAX = 50) -> 5000.
SHORTAGE_PENALTY = 5000.0

# Validation tolerances.
COST_MATCH_TOL = 1e-4       # abs/rel tolerance for DP-vs-bruteforce, PuLP-vs-networkx
OPT_GAP_TOL = 1e-3          # relative gap allowed: DP-seeded MILP vs unrestricted
ASSIGNMENT_VALIDATE_TOL = 1e-6  # Hungarian vs PuLP MILP cross-check on processing time

# DP seeding window (Phase 3 windowed-seeding fix).
# Pure WW restriction piles all SKUs' lots into the same few periods after
# the global period-reorder synchronizes every SKU's seasonal peak. Windowing
# expands each SKU's allowed setup periods to +/-W of each WW period, giving
# the MILP slack to spread load. Feasibility pre-check escalates W until the
# approximate per-period load fits within pooled capacity, capped at SEED_WINDOW_MAX.
SEED_WINDOW = 1          # starting expansion (+/- W periods around each WW period)
SEED_WINDOW_MAX = 4      # maximum W before stopping escalation

# Solver determinism: CBC run single-threaded with a fixed random seed so
# repeated runs give identical objectives (see models/milp.py).
CBC_RANDOM_SEED = RANDOM_SEED

# --------------------------------------------------------------------------- #
# Canonical Favorita stores table (fallback used ONLY if data/raw/stores.csv is
# absent). This is the public 54-store Corporación Favorita stores table. If
# data/raw/stores.csv exists it takes precedence. Documented in ASSUMPTIONS.md.
# Format: store_nbr -> (city, state)
# --------------------------------------------------------------------------- #
STORES_FALLBACK = {
    1:  ("Quito", "Pichincha"),
    2:  ("Quito", "Pichincha"),
    3:  ("Quito", "Pichincha"),
    4:  ("Quito", "Pichincha"),
    5:  ("Santo Domingo", "Santo Domingo de los Tsachilas"),
    6:  ("Quito", "Pichincha"),
    7:  ("Quito", "Pichincha"),
    8:  ("Quito", "Pichincha"),
    9:  ("Quito", "Pichincha"),
    10: ("Quito", "Pichincha"),
    11: ("Cayambe", "Pichincha"),
    12: ("Latacunga", "Cotopaxi"),
    13: ("Latacunga", "Cotopaxi"),
    14: ("Riobamba", "Chimborazo"),
    15: ("Ibarra", "Imbabura"),
    16: ("Santo Domingo", "Santo Domingo de los Tsachilas"),
    17: ("Quito", "Pichincha"),
    18: ("Quito", "Pichincha"),
    19: ("Guaranda", "Bolivar"),
    20: ("Quito", "Pichincha"),
    21: ("Santo Domingo", "Santo Domingo de los Tsachilas"),
    22: ("Puyo", "Pastaza"),
    23: ("Ambato", "Tungurahua"),
    24: ("Guayaquil", "Guayas"),
    25: ("Salinas", "Santa Elena"),
    26: ("Guayaquil", "Guayas"),
    27: ("Daule", "Guayas"),
    28: ("Guayaquil", "Guayas"),
    29: ("Guayaquil", "Guayas"),
    30: ("Guayaquil", "Guayas"),
    31: ("Babahoyo", "Los Rios"),
    32: ("Guayaquil", "Guayas"),
    33: ("Quevedo", "Los Rios"),
    34: ("Guayaquil", "Guayas"),
    35: ("Playas", "Guayas"),
    36: ("Libertad", "Guayas"),
    37: ("Cuenca", "Azuay"),
    38: ("Loja", "Loja"),
    39: ("Cuenca", "Azuay"),
    40: ("Machala", "El Oro"),
    41: ("Machala", "El Oro"),
    42: ("Cuenca", "Azuay"),
    43: ("Esmeraldas", "Esmeraldas"),
    44: ("Quito", "Pichincha"),
    45: ("Quito", "Pichincha"),
    46: ("Quito", "Pichincha"),
    47: ("Quito", "Pichincha"),
    48: ("Quito", "Pichincha"),
    49: ("Quito", "Pichincha"),
    50: ("Ambato", "Tungurahua"),
    51: ("Guayaquil", "Guayas"),
    52: ("Manta", "Manabi"),
    53: ("Manta", "Manabi"),
    54: ("El Carmen", "Manabi"),
}
