# PlanFlow — ASSUMPTIONS

Single record of every assumption. Phase 0 produces the data instance + a naive,
documented baseline. **No optimization models are built in Phase 0.**

This file separates what is **REAL** (taken from the Corporación Favorita data)
from what is **SYNTHESIZED** (invented for the planning problem). Keep it updated.

---

## Data backbone

- **Dataset:** Corporación Favorita Grocery Sales (Kaggle). Files expected in
  `data/raw/`: `train.csv`, `items.csv`, `stores.csv`.
- `train.csv` (~5 GB / ~125M rows) is **never loaded fully** — it is read in
  chunks (`config.CHUNK_SIZE`), filtered to the 12-week window on the fly using
  lexicographic `YYYY-MM-DD` string comparison, negatives clipped, and
  aggregated immediately. Reading **stops early** once chunk dates pass the
  window end (the file is date-sorted ascending).

---

## REAL (derived from the data)

| Quantity | Source | Notes |
|---|---|---|
| Demand quantities | `train.csv` `unit_sales` | Aggregated to weekly buckets per (item, store), then summed to region (city) level. |
| SKU selection (30) | `train.csv` | Top-30 items by total `unit_sales` within the 12-week window. |
| Region structure (10) | `stores.csv` city + `train.csv` | Stores mapped to cities; top-10 cities by demand of the selected SKUs become demand regions. |
| Store → city mapping | `stores.csv` | If absent, the canonical public 54-store Favorita table embedded in `config.STORES_FALLBACK` is used (see below). |
| SKU → family | `items.csv` `family` | Kept on the instance for the Phase-2 line-assignment model. |

### Window

- **12 consecutive weekly periods**, `WINDOW_START = 2016-01-04` (Monday) through
  `2016-03-27` (84 days). Periods labelled `P01`..`P12`.
- **Returns / negatives:** `unit_sales < 0` are **clipped to 0** during
  aggregation (documented business choice: returns are not negative demand).

---

## SYNTHESIZED (invented — not in the dataset)

| Item | Value | Rationale |
|---|---|---|
| Plants (3) | `PLANT_QUITO`, `PLANT_GUAYAQUIL`, `PLANT_CUENCA` | Manufacturer's plants do not exist in the data. Placed at 3 geographically spread cities (north Andes / coast / south Andes). See `config.PLANTS`. |
| DCs (3) | `DC_SANTO_DOMINGO`, `DC_AMBATO`, `DC_MACHALA` | Distribution centres invented; placed at 3 spread cities. See `config.DCS`. |
| Coordinates | `config.CITY_COORDS` | **Approximate** real `(lat, lon)` for every Ecuadorian city used (regions + plant/DC cities). Hardcoded to avoid any live geocoding API. |
| `COST_PER_KM` | `0.01` | Placeholder transport cost per unit per km used by the naive shipping baseline. |
| `SETUP_COST_PLACEHOLDER` | `100.0` | **Phase-2 placeholder** (TODO). Calibrated for real in Phase 2 via part-period logic. |
| `HOLDING_COST_PLACEHOLDER` | `0.10` | **Phase-2 placeholder** (TODO). |
| `PRODUCTION_COST_PLACEHOLDER` | `1.0` | **Phase-2 placeholder** (TODO). |

### Embedded stores fallback

If `data/raw/stores.csv` is **present**, it is the source of truth for
store→city. If it is **absent**, the pipeline falls back to
`config.STORES_FALLBACK` — the canonical public 54-store Corporación Favorita
stores table (store_nbr → city, state). This keeps Phase 0 runnable without the
file; replace by dropping a real `stores.csv` into `data/raw/`.

---

## Baseline policies (naive "before" — Phase 0)

All three operate on the saved `ProblemInstance` and emit the same KPI structure
later phases will produce.

1. **Production = lot-for-lot.** Each region's demand is assigned to its nearest
   plant (haversine); that plant produces exactly the demand each period — no
   lot-sizing, no inventory. One setup charged per (SKU, plant, period) with
   positive production. Setup & production costs use Phase-2 placeholders.
2. **Shipping = greedy nearest-source.** Each region is served entirely from the
   single nearest facility (plants ∪ DCs) by haversine, ignoring capacity.
   Cost = `distance_km × COST_PER_KM × units`.
3. **Line assignment = arbitrary.** Lines assigned to SKU families in fixed
   identity order, no optimization. Processing cost is a Phase-2 placeholder
   (the processing-cost matrix is synthesized in Phase 2); the assignment is real.

The baseline's `total_landed_cost = production + setup + holding + transport +
processing`. Components marked PLACEHOLDER are not yet calibrated; they are set
properly in Phase 2.

---

## Geocoding note

All coordinates in `config.CITY_COORDS` are **approximate** real locations of
Ecuadorian cities, hardcoded to avoid any network dependency. Distances use the
haversine formula (`utils.haversine`, Earth radius 6371 km).

---

# Phase 2 — Parameter Engineering

Favorita has **no prices, costs, capacities, or processing times**. Phase 2
synthesizes all of them with calibrated, documented logic (not magic numbers) so
the downstream OR models behave non-trivially. Everything is reproducible:
`config.RANDOM_SEED = 42` drives a single `numpy.random.default_rng`, drawn in a
fixed order recorded in `Parameters.meta["draw_order"]`:
`unit_value[N]` → `base_time[F]` → `line_efficiency[F]` → `processing_noise[FxF]`.

The Parameters object (`data/processed/parameters.pkl`) wraps the instance and is
the **new single source of truth** for all later phases. No optimization is done
in this phase — parameter generation + baseline arithmetic only.

### 1. Unit value per SKU (synthetic price)
`unit_value_k ~ Uniform(UNIT_VALUE_MIN=1, UNIT_VALUE_MAX=50)`, seeded. Perishables
(from `items.csv` `perishable` flag) are biased **down** by
`PERISHABLE_VALUE_MULTIPLIER = 0.8` (low-ticket fast-movers). Family & perishable
flags are re-read from `items.csv` and stored on the SKU table (family is needed
for the assignment matrix).

### 2. Holding cost per unit per period
`h_k = HOLDING_RATE_WEEKLY × unit_value_k`, where
`HOLDING_RATE_WEEKLY = ANNUAL_HOLDING_RATE / 52` and `ANNUAL_HOLDING_RATE = 0.22`.
Perishable SKUs carry extra spoilage cost via
`PERISHABLE_HOLDING_MULTIPLIER = 2.0` (documented business choice).

### 3. Setup cost per SKU (calibrated, not arbitrary)
Part-period / EOQ relationship:
`A_k = (T² / 2) × h_k × d̄_k`, with `T = TARGET_REORDER_INTERVAL = 3` and
`d̄_k = total demand of SKU k / 12` (average per-period demand across all regions).
This makes the EOQ-optimal production interval `≈ sqrt(2A_k/(h_k d̄_k)) = T = 3`
periods, so Wagner-Whitin produces **real batching** (neither lot-for-lot nor one
giant batch) and auto-scales setup to each SKU's demand.

### 4. Production cost per unit
`p_k = PRODUCTION_COST_FRACTION × unit_value_k`, fraction `0.5`. This is **mostly a
constant** in the optimization (total demand is produced regardless of policy), so
it largely cancels between "before" and "after"; it is included only for a
complete landed-cost figure.

### 5. Plant capacity per period (the critical knob)
- `a_k = 1` (`UNIT_CAPACITY_CONSUMPTION`): one unit of any product consumes one
  unit of plant capacity.
- `D_t = Σ_{SKU,region} demand` per period; `D̄ = mean(D_t)`, `D_max = max(D_t)`.
- `C_total = CAPACITY_TIGHTNESS × D̄`, `CAPACITY_TIGHTNESS = 1.1`. This makes
  **peak periods binding** (`D_max > C_total` ⇒ the model must pre-build
  inventory) while keeping the full horizon feasible
  (`12 × C_total = 1.1 × total demand ≥ total demand`).
- `C_total` is split **unequally** across the 3 plants via
  `PLANT_CAPACITY_SHARES` (aligned to instance plant order),
  creating a clear bottleneck plant. Same capacity every period.
- **Feasibility check** (`12 × C_total ≥ total demand`) is printed PASS/FAIL with
  D̄, D_max, C_total, per-plant capacities and the peak gap. On FAIL the report
  tells you to raise `CAPACITY_TIGHTNESS`.
- All plants may produce all SKUs in the base model (no eligibility restriction);
  plant–product eligibility could be added later.

#### Phase 4 follow-up: demand-gravity recalibration of PLANT_CAPACITY_SHARES

The **original shares were `[0.40, 0.35, 0.25]`** (Quito/Guayaquil/Cuenca), chosen
in Phase 2 solely to create a non-trivial bottleneck for the assignment model.
They were never verified against demand geography.

A Phase 4 diagnostic (run after the initial Phase 4 KPI comparison) computed
each plant's **demand gravity** — the share of total horizon demand whose nearest
(cheapest) source is that plant:

| Plant | Capacity share (original) | Demand gravity | Excess demand (pp) |
|---|---|---|---|
| PLANT_QUITO | 40% | 75.7% | **+35.7 pp** ← structurally over-capacity |
| PLANT_GUAYAQUIL | 35% | 16.2% | −18.8 pp |
| PLANT_CUENCA | 25% | 8.1% | −16.9 pp |

Quito region alone accounts for **65.7%** of total demand but PLANT_QUITO had only
40% of capacity — 1.64× more demand gravity than capacity. With the MILP
saturating PLANT_QUITO and spilling ~1.25M units of Quito demand onto
PLANT_GUAYAQUIL (0.445/unit) and PLANT_CUENCA (0.503/unit) vs PLANT_QUITO's
local rate (0.064/unit), transport cost rose 197% and controllable cost
regressed **+5.3%** vs baseline.

**Fix:** `PLANT_CAPACITY_SHARES` recalibrated to **`[0.55, 0.28, 0.17]`** — tracking
the demand-gravity split while keeping Quito's capacity *intentionally* below its
raw 75.7% gravity share so a real bottleneck persists for the assignment model.

| Plant | Capacity share (recalibrated) | Demand gravity | Excess demand (pp) |
|---|---|---|---|
| PLANT_QUITO | **55%** | 75.7% | **+20.7 pp** (reduced from +35.7) |
| PLANT_GUAYAQUIL | **28%** | 16.2% | −11.8 pp |
| PLANT_CUENCA | **17%** | 8.1% | −8.9 pp |

Quito's cross-plant overflow dropped from **~1.25M units** to **~0.30M units**.
Transport cost in Phase 4 fell from +197% to +107% vs baseline, and controllable
cost swung from a **+5.3% regression** to a **−3.7% reduction**. All Phase 2
feasibility invariants (peak binding, pre-build feasibility, distances > 0) pass
unchanged at `CAPACITY_TIGHTNESS = 1.10` with no escalation needed.

### 6. Distance & transport-cost matrices
Haversine over coordinates, as labeled DataFrames (rows = sources, cols =
destinations): `plant→region`, `DC→region`, `plant→DC`. Transport cost =
`distance_km × COST_PER_UNIT_KM`.

**Facility offset (so transport is economically real).** Plants/DCs originally
sat exactly on city centroids, which made some region distances zero. Each
facility is now shifted a **fixed `FACILITY_OFFSET_KM = 35` km** in a *seeded
random bearing* from its city centroid (equirectangular Δlat/Δlon conversion;
1° lat ≈ 111.32 km). The bearings use an **independent RNG**
(`FACILITY_OFFSET_SEED = RANDOM_SEED + 1`) so the cost/processing draws stay
byte-identical to before this fix. Result: **every** plant→region, DC→region and
plant→DC distance is **strictly positive** (global min ≈ 16.3 km). The sanity
report prints min/mean/max per matrix and asserts `min > 0`.

**`COST_PER_UNIT_KM` is calibrated, not guessed.** Target: baseline transport
cost = **8–10 % of controllable cost** (setup + holding + transport). Since the
lot-for-lot baseline holds no inventory (holding = 0), with
`transport = rate × transport_base` and
`transport_base = Σ_region (min source distance) × region_demand`:

```
rate = TARGET_TRANSPORT_SHARE × setup_total / (transport_base × (1 − TARGET_TRANSPORT_SHARE))
```

With `setup_total = 4,014,500.86`, `transport_base ≈ 215,739,240`, and
`TARGET_TRANSPORT_SHARE = 0.09`, this gives **`COST_PER_UNIT_KM = 0.00184`**
(in `config.py`). The sanity report's transport-share line confirms **9.00 %**
(inside the 5–15 % band).

### 7. Line × SKU-family processing-cost matrix
The 30 SKUs group into their real `family` values → `F` families. `L = F`
production lines (square assignment). `base_time[f] ~ Uniform(1, 5)` per family;
`line_efficiency[l] ~ Uniform(0.8, 1.2)` per line;
`processing_cost[l,f] = base_time[f] × line_efficiency[l] × (1 + ε)` with
`ε ~ Uniform(−0.05, 0.05)`. The noise makes the **optimal assignment non-trivial**
(not the identity).

### Baseline (locked "before") — Phase 2
- **Production = lot-for-lot:** each SKU is produced every period it has demand →
  a setup is charged each such period (`Σ A_k`), zero holding.
- **Shipping = greedy nearest-source:** each region served entirely from the
  single cheapest source (by transport cost across plants ∪ DCs), capacity
  ignored → baseline transport cost.
- **Line assignment = arbitrary:** lines → families in fixed identity order →
  processing **time** summed from the matrix.

#### Headline KPI = controllable cost
- **Controllable cost = setup + holding + transport.** This is the cost the
  optimization can actually change; it is the **headline figure** and the basis
  for every optimized-vs-baseline comparison in later phases. The KPI dict
  returned by `run_baseline()` is the exact structure later phases reuse
  (`controllable_cost`, plus `setup_cost`/`holding_cost`/`transport_cost`
  components, `production_cost_passthrough`, `total_landed_cost`,
  `processing_time`).
- **Production cost is pass-through** (`p_k × total_demand_k`): incurred
  regardless of policy because total demand is fixed, so it is reported
  **separately and explicitly labelled "pass-through (not optimized)"** and is
  **not** part of controllable cost.
- **Total landed cost = controllable + pass-through**, shown **last**, for
  completeness only — it is not the optimization target.
- **Processing time is a separate, time-based KPI** (the matrix holds
  `base_time × line_efficiency × (1+ε)` — *time*, not currency). It is reported
  on its own line in its own units and is **never** added to controllable or
  landed cost.

Calibrated baseline numbers (seed 42): controllable **4,411,461.06**
(setup 91.0 % / holding 0.0 % / transport 9.0 %); production pass-through
77,035,465.86; total landed 81,446,926.93; processing time 23.91 units.

---

# Phase 3 — OR Models

## Structural fix: period reordering

### Why it was needed

The raw 12-week window (2016-01-04..2016-03-27) placed the highest-demand
week at **index 0** (the very first period). The MILP has zero opening
inventory, so it cannot pre-build ahead of period 0 — any demand that exceeds
`C_total` in period 0 is an unavoidable forced shortage. With
`CAPACITY_TIGHTNESS = 1.10` the aggregate peak demand was **547,692 units vs
C_total = 527,422**, yielding an unavoidable shortage of ≈20,270 units and a
forced shortage-penalty term of ~101M in the objective. CBC/HiGHS could not
close this gap because the LP relaxation itself was infeasible (the bound was
lower than any integer-feasible point).

### The permutation

`build_parameters.py::reorder_periods_to_midhorizon()` applies a single
**global permutation** of the 12 periods (the same permutation for every SKU
and every region), chosen so that:

1. The period of highest aggregate demand lands at `PEAK_TARGET_INDEX = 6`
   (0-indexed), placing it mid-horizon.
2. Remaining periods are arranged in a unimodal tent around the peak:
   alternating left/right outward by descending aggregate demand.
3. Periods are **relabelled P01..P12** in the new planning order; demand is
   remapped accordingly.

The permutation is valid because the 12 periods are a planning horizon, not a
fixed calendar — their relative order carries no modeling meaning. Pre-build
feasibility is verified post-permutation (`check_prebuild_feasibility`): the
cumulative pre-peak capacity must cover the cumulative pre-peak demand plus the
peak overflow. If not, `CAPACITY_TIGHTNESS` is escalated through
`CAPACITY_TIGHTNESS_ESCALATION = [1.15, 1.20]` until feasible.

Result with seed 42, `CAPACITY_TIGHTNESS = 1.10`: peak lands at **P07
(index 6)**, pre-build feasibility margin **≈ 287,267 units** >> overflow
**≈ 20,270 units**. MILP Step 1 objective dropped from ~101M (forced shortage)
to ~80.5M (zero shortage, 13.6% setup-cost reduction vs lot-for-lot).

### Caveat: synchronized SKU peaks

The permutation is based on **aggregate total demand** across all SKUs and
regions. Every SKU's demand series is reordered with the identical permutation.
Because the most-demanded SKUs dominate the aggregate signal, all 30 SKUs end
up with their individual seasonal peak concentrated near period index 6. As a
result, 30 independent per-SKU Wagner-Whitin solves all batch their big lots
into the same 3–5 periods near mid-horizon, producing a pooled load of
**~261% of total plant capacity** in those periods — far beyond what any
tight restriction of the MILP's setup variables can feasibly assign. This is
the root cause that required the windowed-seeding fix below.

---

## DP-seeded MILP: windowed seeding

### The problem with pure WW seeding

Restricting the MILP to exactly each SKU's Wagner-Whitin production periods
(`W = 0`) leaves the solver no room to spread production away from the
synchronized peak. With ~261% of pooled capacity demanded in period 6, the
MILP was forced into **479,473 units of shortage** (33 cells) even though the
full (unrestricted) MILP finds a zero-shortage solution.

### The fix: +/-W window expansion

`dp.windowed_restrict(dp_seed, T, W)` expands each SKU's allowed setup periods
to include all periods within `±W` of each WW-chosen period, clipped to
`[0, T-1]`. The MILP retains the WW signal (it can still produce in the
DP-optimal periods) but gains adjacent slots where it can legally pre-build or
defer to relieve the synchronized peak.

### Escalation logic (in `validate.suite_milp` Step 2)

Before calling the MILP, a fast **approximate feasibility pre-check** is run:
for each period `t`, sum `total_demand[sku] / n_runs[sku]` (the average WW lot
size) over all SKUs whose window includes `t`, then compare to `cap_total`.
This is an over-estimate (the MILP can spread across all allowed slots, not
just `t`), so it is conservative.

Starting at `W = SEED_WINDOW = 1`, if any period's approximate load exceeds
`cap_total`, increment `W` and re-check, up to `SEED_WINDOW_MAX = 4`. The
chosen `W` and per-period load margins are printed at each step. If all
windows remain overloaded at `W = SEED_WINDOW_MAX`, the MILP is still solved
and any residual shortage is reported as a finding (not a hard failure).

### Final experimental results (seed 42)

Three window variants were tested. All experiments used `time_limit_sec=120`,
`gap_rel=0.01`, HiGHS solver. Full (unrestricted) MILP benchmark: objective
**80,503,108**, shortage **0**, 287 setups, 120s solve, 2.74% MIP gap.

**1. Symmetric windows W=1 through W=4 — all degenerate to the full model.**

With ~4 WW production runs per SKU over a 12-period horizon, even a ±1 window
already covers effectively the whole horizon for every SKU (4 periods × 3-slot
window overlaps the entire 12-period range). For all W≥1, the approximate
pre-check load is a flat **1,438,423 units on every period** (2.73× cap_total
= 527,422), because every SKU is allowed in every period and the heuristic
sums total demand / n_runs for each. The MILP solution at W=4 is effectively
unrestricted: objective and n_setups are identical to Step 1 at every W tested,
and solve time is the same 120s time limit. The "restriction" restricts nothing.

**2. Forward-only window [0,+1], one-shot — genuinely restrictive but infeasible.**

With forward-only expansion, the average allowed periods/SKU drops to **8.0**
(vs 12 unrestricted). The MILP solved in **0.21s** at a tight **0.017% MIP
gap** — a genuinely restricted problem. However, it forced **1,534,317 units
of shortage across 97 cells** (objective 7,717,641,036 — 95× larger than the
full model). Forward-only deferral cannot help because the structural need
created by the peak-reorder fix is *backward* pre-building ahead of the
mid-horizon peak: SKUs whose WW period falls at or after the peak have no
earlier slots in a forward-only window, so their pre-peak demand is impossible
to cover.

**3. Conclusion — DP seeding does not decompose this instance.**

No window shape tested is simultaneously tight enough to meaningfully restrict
the search space AND loose enough to remain feasible. The reason is structural,
not a tuning failure:

- The global period-reorder (needed to convert the Phase-2 period-0 capacity
  infeasibility into a solvable mid-horizon pre-build) synchronizes every
  SKU's seasonal peak onto the same window of periods.
- Single-item Wagner-Whitin ignores shared plant capacity entirely, so its
  per-SKU optimal batching piles ~261% of pooled capacity into those same
  synchronized periods.
- Any restriction tight enough to be binding inherits that infeasibility.

**The full unrestricted MILP is the correct solver for this instance.**

---

# Phase 4 — Cascade and KPI Report

## Aggregate-horizon transportation (deliberate simplification)

The MILP tracks inventory `I[sku, t]` **pooled across all plants** (a documented
Phase 3 modeling choice). There is therefore no way, in any given period, to
determine which plant's output physically serves which region's demand — some
may come from inventory built at an unknown plant in an earlier period.

**Consequence for the transport model:** running the transportation LP
per-period would require knowing, per period, how much each plant ships to each
region. Without per-plant per-period inventory this is unknowable. Extending
the MILP to track per-plant inventory (which would enable a per-period
transport cascade) is a documented future extension.

**What Phase 4 does instead:** the transportation LP is solved **once over the
full horizon** (aggregate):

```
supply[plant]  = sum over (sku, t) of MILP production_plan at that plant
                 (total units produced there across all 12 periods)
demand[region] = sum over (sku, t, region) of real demand
                 (total horizon demand per region — same aggregation as
                  baseline_shipping() in baseline.py)
cost_matrix    = params.cost_plant_region (plant x region transport cost)
```

This is **apples-to-apples with the baseline**: `baseline_shipping()` also
aggregates demand over the full horizon and applies a per-unit transport cost.
Both use the same `cost_plant_region` matrix and the same total demand vector.
The only difference is that the baseline assigns each region to its single
cheapest source (greedy, capacity-ignored), while the optimized model solves
the LP to minimise total transport cost across all plants simultaneously.

**What this does NOT capture:** within-period capacity interaction between
production and shipping. A true integrated model would need per-plant per-period
inventory tracking. This is the correct next modeling step for Phase 5.

The Wagner-Whitin DP is independently valid and useful — it solves and
validates the uncapacitated single-item lot-sizing optimum for each SKU
(17/17 vs brute force), and its batching signal (avg 4 runs per 12-period
horizon, target interval ~3) confirms that setup-cost calibration is correct.
It was tested as a MILP warm-start/restriction and found NOT to decompose this
capacity-constrained, demand-correlated instance, which is a documented,
defensible OR finding about when single-item decomposition heuristics apply.
DP seeding remains a valid acceleration technique for instances where SKU
peaks are not synchronized by a common demand-driven permutation.

## Phase 4 final KPI results (after capacity recalibration)

All numbers below use `PLANT_CAPACITY_SHARES = [0.55, 0.28, 0.17]` and seed 42.
MILP: HiGHS, `time_limit=120s`, `gap_rel=0.01`, `restrict_setups_to=None`.
Transport: aggregate-horizon LP (one solve over full horizon). Assignment: Hungarian.

### Phase 3 Step 1 (unrestricted MILP, recalibrated capacity)

| Item | Value |
|---|---|
| Status | Optimal (time-limited) |
| Objective (incumbent) | 80,462,471.53 |
| MIP gap | 2.78% |
| Total shortage | 0.0000 units |
| n_setups | 279 |
| Solve time | 120.10 s |

### Phase 4 KPI comparison: baseline vs optimised

| KPI | Baseline | Optimised | Change |
|---|---|---|---|
| setup_cost | 4,014,500.86 | 3,222,002.10 | **−19.7%** |
| holding_cost | 0.00 | 205,003.63 | *(pre-build for peak)* |
| transport_cost | 396,960.20 | 821,163.34 | +106.9% |
| **CONTROLLABLE COST** | **4,411,461.06** | **4,248,169.06** | **−3.7%** |
| production pass-through | 77,035,465.86 | 77,035,465.81 | ~0 |
| total_landed_cost | 81,446,926.93 | 81,283,634.87 | −0.2% |
| processing_time (time units) | 23.9084 | 22.4568 | **−6.1%** |
| setup_count | 360 | 279 | **−22.5%** |
| peak plant utilisation | — | 100.0% | — |
| achieved MIP gap | — | 2.78% | 120s limit |

**Per-plant average utilisation:** PLANT_QUITO 91.5%, PLANT_GUAYAQUIL 92.0% (bottleneck), PLANT_CUENCA 87.0%.

**Assignment:** Hungarian = PuLP MILP = 22.4568 time units (diff 3.55×10⁻¹⁵); 6.07% reduction vs baseline identity mapping.

**Transport cost trade-off:** setup savings (−$792K) and holding pre-build cost (+$205K) net to −$587K, more than offsetting the transport premium (+$424K) under the recalibrated geography. The remaining +107% transport increase vs baseline is structural: PLANT_QUITO's 55% capacity share still falls below Quito's 75.7% demand gravity, so ~300K units of Quito demand must be served at cross-plant rates (0.44–0.50/unit vs 0.064/unit local). This residual mismatch is intentional — it preserves a non-trivial bottleneck at Quito for the assignment model.
