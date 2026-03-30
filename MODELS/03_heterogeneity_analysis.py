#!/usr/bin/env python3
from __future__ import annotations

# Post-process and visualize treatment-effect heterogeneity.

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
from importlib.util import module_from_spec, spec_from_file_location

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from heterogeneity_helpers import (
    format_repeated_clan_table,
    plot_permutation_importance,
    plot_sorted_gates,
    run_permutation_importance,
    run_repeated_sample_splitting,
)

DEFAULT_SPLITS = 100
DEFAULT_CF_TREES = 400
DEFAULT_PERMUTATIONS = 50


# Load the numbered main-model script by path.
def load_main_model_module():
    module_path = Path(__file__).resolve().parent / "02_main_model_building.py"
    spec = spec_from_file_location("main_model_building", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN_MODEL = load_main_model_module()
load_and_prepare = MAIN_MODEL.load_and_prepare


# Build report-ready repeated-split heterogeneity outputs.
def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    hetero_dir = outputs_dir / "heterogeneity_results"
    hetero_dir.mkdir(parents=True, exist_ok=True)

    # These defaults define the formal repeated-split heterogeneity results used in the paper.
    n_splits = int(os.environ.get("HET_N_SPLITS", str(DEFAULT_SPLITS)))
    cf_trees = int(os.environ.get("HET_CF_TREES", str(DEFAULT_CF_TREES)))
    n_perm = int(os.environ.get("HET_N_PERM", str(DEFAULT_PERMUTATIONS)))
    perm_cf_trees = int(os.environ.get("HET_PERM_CF_TREES", str(max(200, cf_trees // 2))))

    full_df, x_cols = load_and_prepare(project_root / "data" / "budgetfood.csv", town_threshold=4)
    # Repeated splitting separates score construction from evaluation to reduce overfitting noise.
    repeated_blp, repeated_sorted_gate, repeated_clan = run_repeated_sample_splitting(
        full_df,
        x_cols=x_cols,
        n_splits=n_splits,
        base_seed=2026,
        cf_trees=cf_trees,
    )
    # The permutation step asks whether variable importance exceeds what random ranking would generate.
    perm_importance_summary, perm_importance_draws = run_permutation_importance(
        full_df,
        x_cols=x_cols,
        n_perm=n_perm,
        base_seed=4040,
        cf_trees=perm_cf_trees,
    )
    repeated_clan_table = format_repeated_clan_table(repeated_clan)

    repeated_blp.to_csv(hetero_dir / "repeated_split_blp_summary.csv", index=False)
    repeated_sorted_gate.to_csv(hetero_dir / "repeated_split_sorted_gate_summary.csv", index=False)
    repeated_clan.to_csv(hetero_dir / "repeated_split_clan_summary.csv", index=False)
    repeated_clan_table.to_csv(hetero_dir / "repeated_split_clan_table.csv", index=False)
    perm_importance_summary.to_csv(hetero_dir / "permutation_importance_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_splits": n_splits,
                "cf_trees": cf_trees,
                "n_permutations": n_perm,
                "perm_cf_trees": perm_cf_trees,
            }
        ]
    ).to_csv(hetero_dir / "repeated_split_settings.csv", index=False)

    plot_sorted_gates(
        repeated_sorted_gate[repeated_sorted_gate["group"].isin(["Q1", "Q2", "Q3", "Q4"])].copy(),
        hetero_dir / "repeated_split_sorted_gate_plot.png",
    )
    plot_permutation_importance(perm_importance_summary, hetero_dir / "permutation_importance_plot.png")

    print("Saved heterogeneity outputs to:", hetero_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
