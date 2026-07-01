"""
transportation.py — Model 2: transportation / min-cost-flow LP (PuLP) plus a
networkx cross-check helper.

Ship finished goods from sources (plants and/or DCs) to demand regions at minimum
transport cost. Unbalanced supply (total supply >= total demand) is supported by
the <= supply / >= demand inequalities.
"""

from __future__ import annotations

import pandas as pd
import pulp


def solve_transportation(supply: dict,
                         demand: dict,
                         cost_matrix: pd.DataFrame) -> dict:
    """Minimum-cost transportation LP. See Phase-3 spec for the formulation.

    Returns {status, total_cost, flows (nonzero only), feasible}.
    """
    sources = list(supply.keys())
    regions = list(demand.keys())

    total_supply = float(sum(supply.values()))
    total_demand = float(sum(demand.values()))

    # Up-front infeasibility: cannot meet demand if supply is short.
    if total_supply + 1e-9 < total_demand:
        return {
            "status": "Infeasible",
            "total_cost": float("nan"),
            "flows": {},
            "feasible": False,
            "message": (f"total supply {total_supply:,.2f} < total demand "
                        f"{total_demand:,.2f}: cannot satisfy demand."),
        }

    prob = pulp.LpProblem("transportation", pulp.LpMinimize)
    f = {(i, j): pulp.LpVariable(f"f_{i}_{j}", lowBound=0)
         for i in sources for j in regions}

    prob += pulp.lpSum(float(cost_matrix.at[i, j]) * f[(i, j)]
                       for i in sources for j in regions)
    for i in sources:
        prob += pulp.lpSum(f[(i, j)] for j in regions) <= supply[i], f"supply_{i}"
    for j in regions:
        prob += pulp.lpSum(f[(i, j)] for i in sources) >= demand[j], f"demand_{j}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0, threads=1))
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return {"status": status, "total_cost": float("nan"),
                "flows": {}, "feasible": False}

    flows = {}
    for (i, j), var in f.items():
        v = var.value()
        if v and abs(v) > 1e-9:
            flows[(i, j)] = float(v)
    return {
        "status": status,
        "total_cost": float(pulp.value(prob.objective)),
        "flows": flows,
        "feasible": True,
    }


# --------------------------------------------------------------------------- #
# networkx cross-check helper
# --------------------------------------------------------------------------- #
NX_SCALE = 10_000  # cost-scaling factor: networkx needs integer weights


def networkx_transport_cost(supply: dict,
                            demand: dict,
                            cost_matrix: pd.DataFrame,
                            scale: int = NX_SCALE) -> float:
    """Min-cost-flow cost via networkx, de-scaled back to real currency units.

    networkx.min_cost_flow needs integer capacities, weights, and node demands.
    We scale unit costs by `scale` and round to int; supplies/demands must be
    integer in the test instances (validation enforces this) so they map exactly.

    Graph: SUPER_SRC -(cap=supply, w=0)-> source -(cap=inf, w=cost*scale)->
           region -(cap=demand, w=0)-> SUPER_SINK. SUPER_SRC pushes exactly
           total_demand units (handles unbalanced supply > demand).
    Returns cost_of_flow / scale.
    """
    import networkx as nx

    sources = list(supply.keys())
    regions = list(demand.keys())
    total_demand = int(round(sum(demand.values())))

    G = nx.DiGraph()
    SRC, SINK = "__SUPER_SRC__", "__SUPER_SINK__"
    G.add_node(SRC, demand=-total_demand)
    G.add_node(SINK, demand=total_demand)
    big = total_demand + 1  # ample per-edge capacity for source->region arcs
    for i in sources:
        G.add_edge(SRC, i, capacity=int(round(supply[i])), weight=0)
    for j in regions:
        G.add_edge(j, SINK, capacity=int(round(demand[j])), weight=0)
    for i in sources:
        for j in regions:
            w = int(round(float(cost_matrix.at[i, j]) * scale))
            G.add_edge(i, j, capacity=big, weight=w)

    flow = nx.min_cost_flow(G)
    cost_scaled = nx.cost_of_flow(G, flow)
    return cost_scaled / scale
