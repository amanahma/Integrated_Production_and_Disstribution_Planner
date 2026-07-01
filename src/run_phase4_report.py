"""
run_phase4_report.py — Phase 4 entry point.

Loads the saved Parameters, runs the baseline (reusing baseline.run_baseline()),
runs the full optimized pipeline (cascade.run_optimized_pipeline()), and prints
the KPI comparison report.

Run:   python src/run_phase4_report.py
       python -m src.run_phase4_report       (from project root)
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))       # src/
_MODELS = os.path.join(_HERE, "models")                  # src/models/
for _p in (_HERE, _MODELS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from parameters import Parameters
from baseline import run_baseline, print_report as print_baseline_report
from cascade import run_optimized_pipeline
from kpi_report import print_kpi_comparison


def main() -> None:
    if not os.path.isfile(config.PARAMETERS_PATH):
        print(f"ERROR: no parameters at {config.PARAMETERS_PATH}.")
        print("Run `python src/build_parameters.py` first.")
        sys.exit(1)

    params = Parameters.load(config.PARAMETERS_PATH)

    # ------------------------------------------------------------------ #
    # Baseline
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("STEP 1 — BASELINE (naive lot-for-lot / greedy / identity)")
    print("=" * 70)
    baseline_result = run_baseline(params)
    print_baseline_report(baseline_result)

    # Augment baseline kpi with n_setups (from production sub-dict) so
    # the comparison table can show setup-count reduction.
    baseline_kpi = dict(baseline_result["kpi"])
    baseline_kpi["n_setups"] = baseline_result["production"]["n_setups"]

    # ------------------------------------------------------------------ #
    # Optimized pipeline
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("STEP 2 — OPTIMIZED PIPELINE")
    print("=" * 70)
    optimized_result = run_optimized_pipeline(params)
    optimized_kpi = optimized_result["kpi"]

    # ------------------------------------------------------------------ #
    # Assignment standalone validation (Hungarian vs PuLP cross-check)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("STEP 3 — ASSIGNMENT MODEL VALIDATION (Hungarian vs PuLP cross-check)")
    print("=" * 70)
    from assignment import validate_assignment
    validate_assignment(params)

    # ------------------------------------------------------------------ #
    # KPI comparison report
    # ------------------------------------------------------------------ #
    print_kpi_comparison(baseline_kpi, optimized_kpi)


if __name__ == "__main__":
    main()
