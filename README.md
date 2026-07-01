# PlanFlow — Integrated Production & Distribution Planner

PlanFlow is a hierarchical planning cascade for a multi-plant FMCG manufacturer,
combining four Operations Research techniques into a single end-to-end system:
**Dynamic Programming** (lot-sizing diagnostic), **Mixed-Integer Linear Programming**
(capacitated production plan), the **Transportation Problem** (minimum-cost aggregate
shipping), and the **Assignment Problem** (production-line-to-family mapping). Each
technique owns a distinct, real sub-decision — not four disconnected solver calls —
and the output of each feeds the next. The demand backbone is real: 30 top-selling
SKUs, 10 Ecuadorian demand regions, and 12 weekly periods from the public Corporación
Favorita grocery dataset. Cost parameters, plant capacities, and processing times are
synthesized with calibrated, documented logic (no prices exist in the raw data); the
honest breakdown is in [ASSUMPTIONS.md](ASSUMPTIONS.md).

---

## Decision Cascade

```
Real demand (Favorita, 30 SKUs × 10 regions × 12 weeks)
        │
        ├─► [DP] Wagner-Whitin per SKU ─────────────────────────────────────►  diagnostic
        │    Uncapacitated single-item lot-sizing optimum (17/17 vs brute force).        │
        │    Tested as MILP warm-start — does not decompose this instance (see below).   │
        │                                                                                 │
        └─► [MILP] Capacitated lot-sizing (HiGHS, 120s, 2.78% gap) ──────────►  production plan
                  Multi-SKU, multi-plant, shared capacity, pre-build inventory.          │
                  restrict_setups_to=None (full model; DP restriction not viable here).  │
                        │                                                                 │
                        ├─► [Transport LP] Aggregate-horizon min-cost shipping ──────►  shipment plan
                        │    supply[plant] = total MILP production over 12 periods.              │
                        │    demand[region] = total real demand over 12 periods.                 │
                        │    Validated: PuLP vs NetworkX on 3+ instances (incl. unbalanced).     │
                        │                                                                         │
                        └─► [Assignment] Hungarian algorithm at bottleneck plant ────►  line mapping
                             8 lines × 8 SKU families, O(n³) optimal.                           │
                             Validated: Hungarian = independent PuLP MILP to 3.55×10⁻¹⁵.        │
                                                                                                  │
                                                                              KPI comparison ◄────┘
                                              controllable cost (setup + holding + transport)
                                              processing time (time units — separate KPI dimension)
```

### Why DP does not seed the MILP here (and what that tells us)

Wagner-Whitin solves each SKU's uncapacitated lot-sizing problem in isolation, batching
production into its EOQ-optimal periods (~4 runs per 12-period horizon). The MILP then
shares plant capacity across all 30 SKUs simultaneously. For DP seeding to help, each
SKU's DP-optimal periods should be spread across the horizon so the MILP has slack to
accept them. Here they are not: the period-reorder fix that converts the raw data's
impossible period-0 peak into a solvable mid-horizon pre-build (see Phase 3 in
ASSUMPTIONS.md) applies the **same permutation to every SKU**. Because the top-volume
SKUs dominate the aggregate signal, all 30 SKUs' WW lots pile into the same 3–5
mid-horizon periods — a pooled load of ~261% of total plant capacity. Any restriction
tight enough to be binding inherits that capacity infeasibility; the MILP correctly
ignores it and solves the full problem.

DP is still doing real, validated work: it independently confirms the uncapacitated
lot-sizing optimum (17/17 vs brute force), and its batching signal (avg 4 runs per SKU
at `TARGET_REORDER_INTERVAL = 3`) validates that setup costs are calibrated correctly.
The finding that DP seeding does not decompose this instance is a real, characterized
empirical result about when single-item decomposition helps (answer: not when a
common demand-driven permutation synchronizes all SKU peaks).

---

## Results

### 1. Production: setup cost and setup count

| KPI | Baseline (lot-for-lot) | Optimized (MILP) | Change |
|---|---|---|---|
| Setup cost | 4,014,500.86 | 3,222,002.10 | **−19.7%** |
| Setup count | 360 | 279 | **−22.5%** |

The lot-for-lot baseline charges a setup every period any SKU has demand — 360 setups
across 30 SKUs × 12 periods. The MILP batches production into fewer, larger runs
timed to serve multiple periods' demand from inventory, reducing setups by 81 runs and
setup cost by ~$792K. This is the core production-side win that DP+MILP delivers.

### 2. Line assignment: provably optimal processing time

The Hungarian algorithm finds the minimum-weight assignment of 8 production lines to
8 SKU families (8×8 cost matrix). Processing time: **23.9084 → 22.4568 time units (−6.1%)**.

Processing time is a **time-unit KPI, not a currency figure**, and is never added to
setup/holding/transport costs. It represents bottleneck scheduling quality.

This result is cross-validated against an independently formulated PuLP binary MILP
(minimize ∑ cost[l,f] · x[l,f], x ∈ {0,1}, each line and each family used exactly
once). Agreement: |Hungarian − MILP| = 3.55×10⁻¹⁵ — machine epsilon, not a rounding
coincidence. The result is provably optimal, not just "a solver converged."

Bottleneck plant: **PLANT_GUAYAQUIL** (92.0% avg utilization over 12 periods;
PLANT_QUITO 91.5%, PLANT_CUENCA 87.0%).

### 3. Net controllable cost

**Controllable cost = setup + holding + transport.** This is the only cost the
optimization can change; production is a pass-through (total demand is produced
regardless of policy). It is the headline metric.

| Component | Baseline | Optimized | Change |
|---|---|---|---|
| Setup cost | 4,014,500.86 | 3,222,002.10 | −19.7% |
| Holding cost | 0.00 | 205,003.63 | *(pre-build)* |
| Transport cost | 396,960.20 | 821,163.34 | +106.9% |
| **Controllable cost** | **4,411,461.06** | **4,248,169.06** | **−3.7%** |

The holding cost increase is a feature: the MILP deliberately pre-builds inventory in
the six periods before the mid-horizon demand peak, paying ~$205K in holding cost to
avoid a forced shortage penalty of ~$101M that would occur if the peak were served
period-by-period. The transport increase is a structural limitation — explained in the
next section.

### 4. Production pass-through and total landed cost (completeness only)

| KPI | Baseline | Optimized | Change |
|---|---|---|---|
| Production cost (pass-through) | 77,035,465.86 | 77,035,465.81 | ~0 |
| **Total landed cost** | **81,446,926.93** | **81,283,634.87** | **−0.2%** |

Production cost is `unit_cost × total_demand`, fixed by demand regardless of schedule.
It is reported separately and labeled pass-through throughout; it is **not** part of
controllable cost and **not** the optimization target. Total landed cost is the sum of
controllable + pass-through, shown for completeness only — the ~$77M pass-through term
dominates it and masks the planning improvement.

---

## The Transport Trade-off

Transport cost rose from 396,960 to 821,163 (+106.9%). This is not a solver error;
it is a structural consequence of two interacting facts:

**Geography:** Quito region carries **65.7% of total horizon demand**. Its nearest
plant, PLANT_QUITO, holds **55% of total capacity** — still below gravity by 20.7
percentage points. At 91.5% average utilization, PLANT_QUITO simply cannot produce
enough to serve all of Quito's demand locally. The ~300K overflow units must ship from
PLANT_GUAYAQUIL (0.445/unit) or PLANT_CUENCA (0.503/unit) — 7–8× PLANT_QUITO's
local rate of 0.064/unit.

**Decoupling:** The MILP optimizes setup + holding costs without any transport term in
its objective. It schedules each plant's production to minimize lot-sizing costs,
geography-blind, then the transport LP minimizes cost given the fixed production
geography. A fully sequential (decompose-then-optimize) pipeline cannot recover the
gap between "production-optimal geography" and "transport-optimal geography" — this
is a well-documented property of hierarchical planning systems in the OR literature.

The net: setup savings of −$792K absorb holding pre-build of +$205K and transport
premium of +$424K, yielding a −$163K (−3.7%) reduction in controllable cost despite
the transport increase. The natural extension to close the remaining gap is a
**transport-aware MILP** — adding an approximate per-plant transport-cost term to the
production objective so the lot-sizing and shipping decisions co-optimize. This is not
built here to keep the MILP tractable within the project's scope and to keep the
technique boundaries clean.

> **An aside on instance design rigor:** the original capacity shares were 40/35/25
> (Quito/Guayaquil/Cuenca), chosen only to create a MILP bottleneck. They were never
> checked against demand geography. The first Phase 4 run produced a +5.3% cost
> *regression* (transport +197%), which triggered a demand-gravity diagnostic. Shares
> were recalibrated to 55/28/17 — tracking the 75.7/16.2/8.1 gravity split while
> intentionally keeping Quito's capacity below its raw gravity share to preserve a real
> bottleneck. Finding and fixing your own instance design before reporting results is
> part of doing OR correctly.

---

## Architecture — What Each Technique Does

### Wagner-Whitin Dynamic Programming (`src/models/dp.py`)

**Sub-decision:** When should each SKU produce, and how much, ignoring shared capacity?

**Formulation:** Classic Wagner-Whitin recurrence over the lot-sizing horizon.
Forward DP with O(n²) cumulative-demand arrays; horizon n=12, 30 SKUs solved
independently. Objective: minimize setup + holding cost per SKU.

**Validation:** Exhaustive brute-force comparison over all production-period subsets
for each of 17 test cases. 17/17 match exactly. This is a deliberate implementation
exercise — no external lot-sizing library was used.

**Role in cascade:** Standalone diagnostic. Reports average lot sizes and batching
signals. Tested empirically as a MILP warm-start restriction; found not to decompose
this instance (see above). Not consumed downstream.

---

### Capacitated Lot-Sizing MILP (`src/models/milp.py`)

**Sub-decision:** How much should each plant produce of each SKU in each period,
respecting shared plant capacity and minimizing setup + holding + production + shortage
costs?

**Formulation (compact):**
```
min  Σ A_k·y[p,k,t] + Σ h_k·I[k,t] + Σ c_k·X[p,k,t] + Σ M·S[k,t]
s.t. I[k,t] = I[k,t-1] + Σ_p X[p,k,t] + S[k,t] - d[k,t]    (inventory balance)
     Σ_k X[p,k,t] ≤ C[p]                                      (plant capacity)
     X[p,k,t] ≤ U·y[p,k,t]                                    (big-M setup forcing)
     y[p,k,t] ∈ {0,1},  X[p,k,t] ≥ 0,  I[k,t] ≥ 0,  S[k,t] ≥ 0
```

Inventory is pooled across plants (a documented modeling choice; see Limitations).
Solver: HiGHS via PuLP. Time limit: 120s. Reported gap: **2.78%** — time-limited,
not proven global optimum. Zero shortage at optimality.

**Validation:** Shortage verified = 0 at every run. Objective compared before/after
period-reorder fix (101M forced-shortage → 80.5M zero-shortage). MIP gap reported
honestly — not hidden.

---

### Transportation LP (`src/models/transportation.py`)

**Sub-decision:** How should total production from each plant be allocated across
demand regions to minimize aggregate shipping cost?

**Formulation:**
```
min  Σ_{p,r} cost[p,r] · f[p,r]
s.t. Σ_r f[p,r] ≤ supply[p]   for each plant p   (can't ship more than produced)
     Σ_p f[p,r] ≥ demand[r]   for each region r  (must cover all demand)
     f[p,r] ≥ 0
```

Solved once over the aggregate horizon (see Limitations for why not per-period).

**Validation:** PuLP LP solution cross-checked against NetworkX minimum-cost flow
on 3+ randomly generated instances including unbalanced supply-side cases (supply >
demand). Exact agreement in all cases.

---

### Assignment Model (`src/models/assignment.py`)

**Sub-decision:** Which production line should run which SKU family to minimize total
processing time at the bottleneck plant?

**Formulation:** Linear sum assignment (min-weight perfect matching on a bipartite
graph). 8 lines × 8 SKU families; cost matrix entries = `base_time[f] × efficiency[l]
× (1 + ε)` where ε introduces noise so the optimal assignment is not the identity.

**Primary solver:** `scipy.optimize.linear_sum_assignment` (Hungarian algorithm, O(n³)).

**Validation:** Independent PuLP binary MILP formulated from scratch
(x[l,f] ∈ {0,1}, each line and family used exactly once, minimize ∑ cost · x).
Result: |Hungarian − MILP| = 3.55×10⁻¹⁵. Provably optimal.

---

## Data and Parameters

**Real backbone:** SKU demand quantities, SKU-to-family mapping, and store-to-city
geography come directly from the Corporación Favorita dataset (top-30 SKUs by volume,
top-10 cities as demand regions, 12 weekly periods from 2016-01-04). City coordinates
are hardcoded approximate real locations (haversine distances, no API calls).

**Synthesized:** Unit values, holding costs, setup costs, production costs, plant
capacities, processing times, and transport rates are all synthesized. The dataset has
no prices. Every parameter is derived from documented calibration logic (see
[ASSUMPTIONS.md](ASSUMPTIONS.md) §Phase 2) with a single reproducible seed
(`RANDOM_SEED = 42`). Transport rate is calibrated so baseline transport equals
~9% of controllable cost — a specific, verifiable target, not a guess.

**Instance design lesson:** Capacity shares were originally set as 40/35/25
(Quito/Guayaquil/Cuenca) to create a planning bottleneck and were never checked against
demand geography (Quito carries 65.7% of demand but received only 40% of capacity —
a 35.7 pp structural mismatch). The initial Phase 4 run produced a +5.3% cost
regression, triggering a demand-gravity diagnostic and recalibration to 55/28/17.
This is documented in full in [ASSUMPTIONS.md](ASSUMPTIONS.md) §Phase 4 follow-up.

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Data wrangling | `pandas >= 2.0`, `numpy >= 1.24`, `pyarrow >= 12.0` |
| MILP + Transport LP | `pulp >= 2.7` with `highspy >= 1.5` (HiGHS solver) |
| Assignment cross-check | `pulp >= 2.7` (CBC solver, binary MILP) |
| Hungarian algorithm | `scipy >= 1.10` (`scipy.optimize.linear_sum_assignment`) |
| Transport validation | `networkx >= 3.0` (min-cost flow cross-check) |
| Wagner-Whitin DP | Custom implementation (`src/models/dp.py`) — no library |
| Geospatial | Haversine formula in `src/utils.py` — no mapping API |

---

## How to Run

**Prerequisites:** Python 3.10+, raw Favorita files in `data/raw/`
(`train.csv`, `items.csv`, `stores.csv`). Install dependencies:

```bash
pip install -r requirements.txt
```

**Pipeline (run in order from the project root):**

```bash
# 1. Build the problem instance from raw data (~3–5 min, chunks 5GB train.csv)
python src/build_instance.py
#    Writes: data/processed/instance.pkl, data/processed/demand.parquet

# 2. Synthesize parameters and run sanity report (~5 seconds)
python src/build_parameters.py
#    Writes: data/processed/parameters.pkl
#    Prints: capacity tightness, peak binding check, pre-build feasibility,
#            distance matrix stats, per-SKU cost calibration, baseline KPIs

# 3. Run the full validation suite (~3–5 min — includes 120s MILP solve)
python src/models/validate.py
#    Validates: DP (17/17 vs brute force), Transportation LP (PuLP vs NetworkX),
#               MILP (Step 1 unrestricted + Step 2 DP-seeding experiment)

# 4. Run the Phase 4 cascade and KPI comparison report (~3–5 min)
python src/run_phase4_report.py
#    Runs: Baseline → MILP → Transport LP → Hungarian assignment → KPI comparison
#    Prints: full baseline report, optimized pipeline output, KPI comparison table
```

Steps 3 and 4 each include a 120-second HiGHS MILP solve. Steps 1–2 only need to be
re-run if the raw data or config changes. After that, steps 3 and 4 are self-contained.

**Dashboard (Phase 6):**

```bash
# 5. Pre-compute dashboard cache (one 120s MILP solve; writes data/processed/dashboard_cache.pkl)
python src/precompute.py

# 6. Launch the Streamlit dashboard
streamlit run app.py
```

The dashboard reads the cache and never calls any OR solver. Re-run step 5 if
`config.py` or `parameters.pkl` change.

**Streamlit Cloud deployment:**
Push the repository to GitHub, connect it in [share.streamlit.io](https://share.streamlit.io),
set the main file to `app.py`, and add the packages in `requirements.txt`. The cache file
(`data/processed/dashboard_cache.pkl`) must be committed or pre-generated via a
start-up script — it is not produced automatically by the app itself.

---

## Project Structure

```
PlanFlow/
├── ASSUMPTIONS.md              # every number, every modeling decision, every finding
├── README.md                   # this file
├── requirements.txt
│
├── data/
│   ├── raw/                    # place train.csv, items.csv, stores.csv here
│   └── processed/
│       ├── instance.pkl        # ProblemInstance (demand + geography)
│       ├── demand.parquet      # aggregated weekly demand
│       └── parameters.pkl      # Parameters (instance + all costs/capacities)
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
└── src/
    ├── config.py               # single source of truth for all knobs
    ├── build_instance.py       # Phase 0: raw → instance.pkl
    ├── build_parameters.py     # Phase 2: instance → parameters.pkl + sanity report
    ├── instance.py             # ProblemInstance dataclass
    ├── parameters.py           # Parameters dataclass + sanity_report()
    ├── baseline.py             # lot-for-lot / greedy / identity baseline
    ├── cascade.py              # Phase 4: full optimized pipeline orchestrator
    ├── kpi_report.py           # Phase 4: KPI comparison formatter
    ├── run_phase4_report.py    # Phase 4: entry point
    ├── utils.py                # haversine, shared utilities
    └── models/
        ├── dp.py               # Wagner-Whitin DP (frozen)
        ├── milp.py             # capacitated lot-sizing MILP (frozen)
        ├── transportation.py   # min-cost transportation LP (frozen)
        ├── assignment.py       # Hungarian + PuLP binary MILP cross-check
        └── validate.py         # full validation suite (DP, transport, MILP)
```

---

## Limitations and Honest Scope

**Aggregate-horizon transport.** The MILP tracks inventory pooled across all plants
(`I[sku, t]` with no per-plant subscript). Per-period transport would require knowing,
for each period, how much each plant ships to each region — which is unknowable without
per-plant inventory tracking. The transport LP is therefore solved once over the full
horizon (aggregate supply vs. aggregate demand), identical in structure to the greedy
baseline. Extending the MILP to track per-plant inventory would enable a proper
per-period cascade and is the correct next modeling step.

**Decoupled production and transport objectives.** The MILP minimizes setup + holding
costs without transport cost in its objective. The transport LP then minimizes shipping
cost given the fixed production geography. This sequential structure cannot recover the
gap between production-optimal and transport-optimal plant loading — an inherent property
of hierarchical (decompose-then-optimize) planning. A transport-aware MILP objective
(approximate per-plant transport cost as a production penalty) would close this gap.

**Synthesized parameters.** Demand and geography are real; cost, capacity, and
processing-time parameters are synthesized. Results should be read as demonstrating
model behavior on a realistic instance, not as operational numbers for a real
manufacturer.

**Reported, not proven, optimality.** The MILP is solved to a reported **2.78% MIP
gap** at a 120-second time limit (HiGHS). This is standard practice for MILPs of this
scale in production OR settings — a 120-second time budget is a deliberate, explicit
scope boundary, not a limitation of the formulation. The gap is reported on every run
and carried through the KPI comparison.

---

## What I Would Build Next

1. **Transport-aware MILP objective** — add a term `Σ_{p,r} transport_cost[p,r] ×
   Σ_{k,t} X[p,k,t] × demand_share[p→r]` to the production objective, allowing
   lot-sizing decisions to internalize geography. This is the single highest-leverage
   extension.

2. **Per-plant inventory tracking** — replace the pooled inventory variable `I[k,t]`
   with `I[p,k,t]`, enabling per-period transport cascades and removing the aggregate-
   horizon approximation entirely.

3. **Scenario analysis** — demand spikes (e.g., +20% at Quito for two periods),
   capacity loss (one plant offline), and supplier disruptions; the modular cascade
   structure makes this straightforward to parameterize.

4. **Interactive planning dashboard** — the KPI comparison output is already structured
   for display; a Streamlit front-end would surface it for non-technical planners.
