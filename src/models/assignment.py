"""
assignment.py — Model 4: Hungarian line->family assignment.

Minimizes total processing time by optimally assigning production lines to
SKU families. scipy.optimize.linear_sum_assignment (Hungarian algorithm) is
the primary solver; a PuLP binary MILP is built as an independent cross-check
to prove the result is derivable from first principles.

Processing time is a SEPARATE KPI dimension (time units, not currency) and is
NEVER added to controllable cost or total landed cost — see ASSUMPTIONS.md.
"""

from __future__ import annotations

import os
import sys

# Make this module importable standalone: ensure src/ is on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy.optimize import linear_sum_assignment
import pulp

import config


def solve_assignment(processing_cost: pd.DataFrame) -> dict:
    """Hungarian (linear sum) assignment: lines -> families, min total processing time.

    Primary solver for Phase 4. O(n^3) exact optimal.

    Args:
        processing_cost: DataFrame, rows=lines, cols=families, values=processing time.

    Returns:
        {
          "assignment": dict[line, family],
          "total_processing_time": float,
          "n_lines": int,
          "n_families": int,
        }
    """
    lines = list(processing_cost.index)
    families = list(processing_cost.columns)
    cost_matrix = processing_cost.values

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = {lines[r]: families[c] for r, c in zip(row_ind, col_ind)}
    total_time = float(cost_matrix[row_ind, col_ind].sum())

    return {
        "assignment": assignment,
        "total_processing_time": total_time,
        "n_lines": len(lines),
        "n_families": len(families),
    }


def solve_assignment_milp(processing_cost: pd.DataFrame) -> dict:
    """Binary assignment MILP (PuLP) — cross-check for solve_assignment.

    Variables:  x[line, family] in {0, 1}
    Objective:  min  sum(cost[l,f] * x[l,f])
    Constraints:
        sum_f x[l,f] = 1   for each line    (each line assigned to one family)
        sum_l x[l,f] = 1   for each family  (each family covered by one line)

    Returns same shape dict as solve_assignment, plus "status".
    """
    lines = list(processing_cost.index)
    families = list(processing_cost.columns)

    prob = pulp.LpProblem("assignment_milp", pulp.LpMinimize)
    x = {(l, f): pulp.LpVariable(f"x_{l}_{f}", cat="Binary")
         for l in lines for f in families}

    prob += pulp.lpSum(float(processing_cost.at[l, f]) * x[(l, f)]
                       for l in lines for f in families)
    for l in lines:
        prob += pulp.lpSum(x[(l, f)] for f in families) == 1, f"line_{l}"
    for f in families:
        prob += pulp.lpSum(x[(l, f)] for l in lines) == 1, f"fam_{f}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0, threads=1))
    status = pulp.LpStatus[prob.status]

    assignment, total_time = {}, 0.0
    if status == "Optimal":
        for l in lines:
            for f in families:
                v = x[(l, f)].value()
                if v is not None and v > 0.5:
                    assignment[l] = f
                    total_time += float(processing_cost.at[l, f])
                    break

    return {
        "status": status,
        "assignment": assignment,
        "total_processing_time": total_time,
        "n_lines": len(lines),
        "n_families": len(families),
    }


def validate_assignment(params) -> None:
    """Cross-check Hungarian vs PuLP MILP; compare vs baseline identity assignment."""
    from baseline import baseline_line_assignment

    print("=" * 64)
    print("[ASSIGNMENT]  Hungarian vs PuLP MILP + baseline comparison")
    print("=" * 64)

    pc = params.processing_cost
    print(f"  Matrix: {len(params.lines)} lines x {len(params.families)} families")

    hungarian = solve_assignment(pc)
    milp_res = solve_assignment_milp(pc)

    tol = config.ASSIGNMENT_VALIDATE_TOL
    diff = abs(hungarian["total_processing_time"] - milp_res["total_processing_time"])
    ok = diff <= tol
    print(f"  Hungarian total_processing_time : {hungarian['total_processing_time']:,.6f}")
    print(f"  PuLP MILP total_processing_time : {milp_res['total_processing_time']:,.6f}")
    print(f"  Difference                      : {diff:.2e}  "
          f"(tol {tol:.0e})  [{'OK' if ok else 'MISMATCH'}]")
    assert ok, f"Hungarian vs MILP mismatch: {diff:.2e} > tol {tol}"

    baseline_assign = baseline_line_assignment(params)
    baseline_time = baseline_assign["processing_time"]
    reduction = (baseline_time - hungarian["total_processing_time"]) / baseline_time * 100
    print(f"\n  Baseline (arbitrary identity)  processing_time : {baseline_time:,.6f}")
    print(f"  Optimized (Hungarian)          processing_time : "
          f"{hungarian['total_processing_time']:,.6f}")
    print(f"  Reduction                                      : {reduction:.2f}%")
    assert hungarian["total_processing_time"] <= baseline_time + tol, (
        "Hungarian result is worse than identity assignment — check processing_cost matrix")

    print(f"\n  Optimal line -> family mapping:")
    for l, f in sorted(hungarian["assignment"].items()):
        print(f"    {str(l):<10} -> {f}")
    print()


if __name__ == "__main__":
    from parameters import Parameters
    params = Parameters.load(config.PARAMETERS_PATH)
    validate_assignment(params)
