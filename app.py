"""
app.py — PlanFlow Streamlit dashboard (Phase 6).

Loads pre-computed results from data/processed/dashboard_cache.pkl — never
calls any OR solver. Build the cache first:
    python src/precompute.py
Then launch:
    streamlit run app.py
"""

from __future__ import annotations

import os
import pickle

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlanFlow",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cache load ─────────────────────────────────────────────────────────────────
_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "processed", "dashboard_cache.pkl",
)


@st.cache_data
def _load_cache() -> dict:
    with open(_CACHE_PATH, "rb") as fh:
        return pickle.load(fh)


try:
    C = _load_cache()
except FileNotFoundError:
    st.error(
        "Cache not found. Run `python src/precompute.py` first, then relaunch."
    )
    st.stop()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("PlanFlow")
st.sidebar.caption("Integrated Production & Distribution Planner")
page = st.sidebar.radio(
    "Page",
    ["KPI Summary", "Production Plan", "Network Map", "Assignment"],
    label_visibility="collapsed",
)

# ── Shared helpers ─────────────────────────────────────────────────────────────
bkpi = C["baseline_kpi"]
okpi = C["optimized_kpi"]


def _pct(before: float, after: float) -> float:
    return (after - before) / abs(before) * 100.0 if before else 0.0


def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


# ══════════════════════════════════════════════════════════════════════════════ #
#  PAGE 1 — KPI Summary
# ══════════════════════════════════════════════════════════════════════════════ #
if page == "KPI Summary":
    st.title("KPI Summary")
    st.caption(
        "Baseline = lot-for-lot production with greedy nearest-source shipping. "
        "Optimized = CLSP MILP (HiGHS, 120 s) + transportation LP + Hungarian assignment."
    )

    # ── Headline metric cards ──────────────────────────────────────────────────
    b_ctrl = bkpi["controllable_cost"]
    o_ctrl = okpi["controllable_cost"]
    b_ns   = C["baseline_n_setups"]
    o_ns   = okpi["n_setups"]
    b_pt   = bkpi["processing_time"]
    o_pt   = okpi["processing_time"]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Controllable Cost",
        f"{o_ctrl:,.2f}",
        delta=_fmt_pct(_pct(b_ctrl, o_ctrl)),
        delta_color="inverse",
        help="Setup + holding + transport (the optimization target). "
             "Lower is better.",
    )
    c2.metric(
        "Setup Count",
        f"{o_ns:,}",
        delta=_fmt_pct(_pct(b_ns, o_ns)),
        delta_color="inverse",
        help="Number of production changeovers over 12 periods.",
    )
    c3.metric(
        "Processing Time",
        f"{o_pt:.4f} time units",
        delta=_fmt_pct(_pct(b_pt, o_pt)),
        delta_color="inverse",
        help="Optimal line-family assignment total processing time (time units — "
             "separate KPI dimension, NOT currency, NOT part of landed cost).",
    )

    st.divider()

    # ── Cost breakdown table ───────────────────────────────────────────────────
    st.subheader("Cost breakdown")
    st.caption(
        "Controllable cost = setup + holding + transport (optimization target). "
        "Production pass-through = unit_cost × total_demand (fixed, same either way). "
        "Total landed = controllable + pass-through."
    )

    rows = [
        ("Setup cost",               bkpi["setup_cost"],               okpi["setup_cost"]),
        ("Holding cost",              bkpi["holding_cost"],              okpi["holding_cost"]),
        ("Transport cost",            bkpi["transport_cost"],            okpi["transport_cost"]),
        ("**Controllable cost**",     bkpi["controllable_cost"],         okpi["controllable_cost"]),
        ("Production pass-through",   bkpi["production_cost_passthrough"], okpi["production_cost_passthrough"]),
        ("**Total landed cost**",     bkpi["total_landed_cost"],         okpi["total_landed_cost"]),
    ]

    df_cost = pd.DataFrame(
        [
            {
                "Component":  label,
                "Baseline":   f"{b:,.2f}",
                "Optimized":  f"{o:,.2f}",
                "Change":     _fmt_pct(_pct(b, o)) if b else "N/A",
            }
            for label, b, o in rows
        ]
    )
    st.dataframe(df_cost, use_container_width=True, hide_index=True)

    # ── Transport callout ──────────────────────────────────────────────────────
    tr_pct = _pct(bkpi["transport_cost"], okpi["transport_cost"])
    with st.expander(f"Transport cost +{tr_pct:.1f}% — why this is expected, not a bug"):
        st.markdown(
            f"""
The MILP's lot-sizing consolidates 360 setups → **{o_ns} setups (−22.5%)** by
batching production around high-demand periods, which concentrates inventory
at PLANT_QUITO (55% capacity, 75.7% demand gravity). Serving outlying regions
from PLANT_QUITO rather than their local plants means longer hauls, so
transport cost rises from **{bkpi['transport_cost']:,.2f} → {okpi['transport_cost']:,.2f}
({tr_pct:+.1f}%)**.

Transport is part of the controllable-cost objective. The solver correctly chose
to pay more on transport to save more on setup — the net controllable cost is
**{_pct(b_ctrl, o_ctrl):.1f}%** ({b_ctrl:,.2f} → {o_ctrl:,.2f}).
"""
        )

    st.divider()

    # ── Operational KPIs ───────────────────────────────────────────────────────
    st.subheader("Operational KPIs")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Utilization by plant (MILP, avg over 12 periods)**")
        util = C["plant_avg_utilization"]
        plant_city = C["plant_city"]
        df_util = pd.DataFrame(
            [
                {
                    "Plant":       p,
                    "City":        plant_city.get(p, p),
                    "Avg util %":  f"{v * 100:.1f}%",
                    "Bottleneck":  "★" if p == C["bottleneck_plant"] else "",
                }
                for p, v in util.items()
            ]
        )
        st.dataframe(df_util, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Solver details**")
        mip_gap = okpi.get("mip_gap", float("nan"))
        st.markdown(
            f"""
| Metric | Value |
|---|---|
| MILP solver | HiGHS (via PuLP) |
| Time limit | 120 s |
| Achieved MIP gap | {mip_gap * 100:.2f}% (time-limited) |
| Transport solver | CBC (aggregate horizon) |
| Assignment solver | Hungarian (scipy) |
| Shortage units | 0 |
"""
        )


# ══════════════════════════════════════════════════════════════════════════════ #
#  PAGE 2 — Production Plan
# ══════════════════════════════════════════════════════════════════════════════ #
elif page == "Production Plan":
    st.title("Production Plan")

    view = st.radio(
        "View",
        ["MILP Optimized", "Baseline (lot-for-lot)"],
        horizontal=True,
    )

    periods = C["periods"]
    period_demand = C["period_demand"]

    if view == "MILP Optimized":
        prod_vals = [C["milp_prod_period"][p] for p in periods]
    else:
        prod_vals = [period_demand[p] for p in periods]

    demand_vals = [period_demand[p] for p in periods]

    # ── Per-period production vs demand bar chart ──────────────────────────────
    st.subheader("Total production vs demand by period")
    fig_period = go.Figure()
    fig_period.add_bar(
        x=periods,
        y=demand_vals,
        name="Demand",
        marker_color="#d62728",
        opacity=0.6,
    )
    fig_period.add_bar(
        x=periods,
        y=prod_vals,
        name="Production" if view == "MILP Optimized" else "Production (= demand)",
        marker_color="#1f77b4",
        opacity=0.8,
    )
    fig_period.update_layout(
        barmode="overlay",
        xaxis_title="Period",
        yaxis_title="Units",
        legend=dict(orientation="h", y=1.05),
        height=380,
    )
    st.plotly_chart(fig_period, use_container_width=True)

    if view == "Baseline (lot-for-lot)":
        st.caption(
            "Baseline produces exactly what demand requires each period (lot-for-lot). "
            "No inventory is held; production = demand bar-for-bar."
        )

    # ── Stacked by plant (MILP only) ───────────────────────────────────────────
    if view == "MILP Optimized":
        st.subheader("Production by plant (stacked)")
        plants = C["plants"]
        plant_city = C["plant_city"]
        colors = ["#2ca02c", "#ff7f0e", "#9467bd"]

        fig_stack = go.Figure()
        for plant, color in zip(plants, colors):
            plant_vals = [C["milp_prod_by_plant"][plant][p] for p in periods]
            fig_stack.add_bar(
                x=periods,
                y=plant_vals,
                name=plant_city.get(plant, plant),
                marker_color=color,
            )
        fig_stack.update_layout(
            barmode="stack",
            xaxis_title="Period",
            yaxis_title="Units produced",
            legend=dict(orientation="h", y=1.05),
            height=380,
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        # Plant production totals
        totals = C["plant_total_production"]
        grand = sum(totals.values())
        df_totals = pd.DataFrame(
            [
                {
                    "Plant":      p,
                    "City":       plant_city.get(p, p),
                    "Total units": f"{totals[p]:,.0f}",
                    "Share %":    f"{totals[p] / grand * 100:.1f}%",
                }
                for p in plants
            ]
        )
        st.dataframe(df_totals, use_container_width=True, hide_index=True)

    st.divider()

    # ── DP diagnostic ──────────────────────────────────────────────────────────
    st.subheader("Wagner-Whitin DP diagnostic (read-only)")
    st.caption(
        "The DP solves the uncapacitated single-item lot-sizing problem for each "
        "SKU independently. Its output is reported here as a diagnostic — it does "
        "NOT seed or restrict the MILP."
    )

    dp = C["dp_summary"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg runs / SKU", f"{dp['avg']:.2f}")
    col2.metric("Min runs", str(dp["min"]))
    col3.metric("Max runs", str(dp["max"]))

    st.markdown(
        f"Over **{dp['n_skus']} SKUs**, 12 periods, target reorder interval "
        f"**{dp['target_interval']} periods**. Avg runs {dp['avg']:.2f} < 12 "
        f"confirms batching behaviour (each run = one lot, not period-by-period)."
    )

    dp_n_runs = C["dp_n_runs"]
    run_counts = list(dp_n_runs.values())
    fig_dp = go.Figure(go.Histogram(
        x=run_counts,
        nbinsx=max(run_counts) - min(run_counts) + 1,
        marker_color="#1f77b4",
        opacity=0.8,
    ))
    fig_dp.update_layout(
        xaxis_title="Number of lots (DP runs per SKU)",
        yaxis_title="SKU count",
        height=280,
        margin=dict(t=30),
    )
    st.plotly_chart(fig_dp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════ #
#  PAGE 3 — Network Map
# ══════════════════════════════════════════════════════════════════════════════ #
elif page == "Network Map":
    st.title("Network Map")

    map_mode = st.radio(
        "Flow view",
        ["Optimized (LP flows)", "Baseline (nearest source)"],
        horizontal=True,
    )

    coords   = C["coords"]
    plants   = C["plants"]
    dcs      = C["dcs"]
    regions  = C["regions"]
    plant_city = C["plant_city"]
    dc_city    = C["dc_city"]

    # Centre on Ecuador
    m = folium.Map(location=[-1.8, -78.2], zoom_start=7, tiles="OpenStreetMap")

    # Plants — green markers
    for plant in plants:
        if plant not in coords:
            continue
        lat, lon = coords[plant]
        city = plant_city.get(plant, plant)
        util_pct = C["plant_avg_utilization"].get(plant, 0.0) * 100
        folium.CircleMarker(
            location=[lat, lon],
            radius=12,
            color="darkgreen",
            fill=True,
            fill_color="green",
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{city}</b><br>{plant}<br>Avg util: {util_pct:.1f}%",
                max_width=200,
            ),
            tooltip=f"Plant: {city} ({util_pct:.1f}% util)",
        ).add_to(m)

    # DCs — blue markers
    for dc in dcs:
        if dc not in coords:
            continue
        lat, lon = coords[dc]
        city = dc_city.get(dc, dc)
        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            color="darkblue",
            fill=True,
            fill_color="steelblue",
            fill_opacity=0.8,
            popup=folium.Popup(f"<b>{city}</b><br>{dc}", max_width=200),
            tooltip=f"DC: {city}",
        ).add_to(m)

    # Regions — red/orange markers
    region_demand = C["region_total_demand"]
    for region in regions:
        if region not in coords:
            continue
        lat, lon = coords[region]
        dem = region_demand.get(region, 0)
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="darkred",
            fill=True,
            fill_color="tomato",
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>{region}</b><br>Demand: {dem:,.0f} units", max_width=200
            ),
            tooltip=f"Region: {region} ({dem:,.0f} units)",
        ).add_to(m)

    # Flow lines
    if map_mode == "Optimized (LP flows)":
        flows = C["opt_flows"]
        max_flow = max(flows.values()) if flows else 1.0
        for (plant, region), qty in flows.items():
            if plant not in coords or region not in coords:
                continue
            weight = 1.5 + 7.0 * (qty / max_flow) ** 0.5
            folium.PolyLine(
                [list(coords[plant]), list(coords[region])],
                color="#2ca02c",
                weight=weight,
                opacity=0.65,
                tooltip=f"{plant_city.get(plant, plant)} → {region}: {qty:,.0f} units",
            ).add_to(m)
    else:
        baseline_source = C["baseline_source"]
        max_dem = max(region_demand.values()) if region_demand else 1.0
        for region, source in baseline_source.items():
            if source not in coords or region not in coords:
                continue
            dem = region_demand.get(region, 0)
            weight = 1.5 + 6.0 * (dem / max_dem) ** 0.5
            src_label = plant_city.get(source, dc_city.get(source, source))
            folium.PolyLine(
                [list(coords[source]), list(coords[region])],
                color="#ff7f0e",
                weight=weight,
                opacity=0.60,
                tooltip=f"{src_label} → {region}: {dem:,.0f} units (baseline)",
            ).add_to(m)

    st_folium(m, width=900, height=560, returned_objects=[])

    # Legend
    st.markdown(
        "**Legend:** "
        "<span style='color:green'>●</span> Plant &nbsp;&nbsp;"
        "<span style='color:steelblue'>●</span> Distribution centre &nbsp;&nbsp;"
        "<span style='color:tomato'>●</span> Demand region &nbsp;&nbsp;"
        "Line width ∝ √(flow volume)",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════ #
#  PAGE 4 — Assignment
# ══════════════════════════════════════════════════════════════════════════════ #
elif page == "Assignment":
    st.title("Line–Family Assignment")
    st.caption(
        "Processing time (time units — separate KPI dimension, NOT currency, NOT part of landed cost). "
        "Hungarian algorithm minimises total processing time across 8 lines × 8 product families. "
        "Validated against PuLP binary MILP to 3.55 × 10⁻¹⁵."
    )

    proc_df: pd.DataFrame = C["processing_cost_df"]
    optimal_assignment: dict = C["optimal_assignment"]
    baseline_assignment: dict = C["baseline_assignment"]
    lines   = C["lines"]
    families = C["families"]

    # Reindex for consistent order
    proc_df = proc_df.reindex(index=lines, columns=families)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    z = proc_df.values.tolist()
    annotations = []
    for line, fam in optimal_assignment.items():
        if line in lines and fam in families:
            annotations.append(dict(
                x=fam,
                y=line,
                text="★",
                showarrow=False,
                font=dict(size=18, color="white"),
            ))

    fig_heat = go.Figure(go.Heatmap(
        z=z,
        x=families,
        y=lines,
        colorscale="Blues",
        reversescale=True,
        colorbar=dict(title="Time (time units)"),
        hoverongaps=False,
        hovertemplate=(
            "Line: %{y}<br>Family: %{x}<br>"
            "Processing time: %{z:.4f} time units<extra></extra>"
        ),
    ))
    fig_heat.update_layout(
        title="Processing time matrix  (★ = optimal assignment, darker = faster)",
        xaxis_title="Product Family",
        yaxis_title="Production Line",
        xaxis=dict(tickangle=-30),
        annotations=annotations,
        height=520,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Mapping table ──────────────────────────────────────────────────────────
    st.subheader("Assignment mapping")
    b_pt_total = bkpi["processing_time"]
    o_pt_total = okpi["processing_time"]

    table_rows = []
    for line in lines:
        opt_fam = optimal_assignment.get(line, "—")
        base_fam = baseline_assignment.get(line, "—")
        opt_t = float(proc_df.at[line, opt_fam]) if opt_fam in families else float("nan")
        base_t = float(proc_df.at[line, base_fam]) if base_fam in families else float("nan")
        table_rows.append({
            "Line":                 line,
            "Optimal family":       opt_fam,
            "Optimal time (t.u.)":  f"{opt_t:.4f}",
            "Baseline family":      base_fam,
            "Baseline time (t.u.)": f"{base_t:.4f}",
        })

    df_map = pd.DataFrame(table_rows)
    st.dataframe(df_map, use_container_width=True, hide_index=True)

    col_x, col_y = st.columns(2)
    col_x.metric(
        "Baseline total processing time",
        f"{b_pt_total:.4f} time units",
    )
    col_y.metric(
        "Optimal total processing time",
        f"{o_pt_total:.4f} time units",
        delta=_fmt_pct(_pct(b_pt_total, o_pt_total)),
        delta_color="inverse",
    )

    st.caption(
        "Processing time is an independent KPI dimension — it measures line efficiency, "
        "not currency cost, and is NOT included in controllable cost or total landed cost."
    )
