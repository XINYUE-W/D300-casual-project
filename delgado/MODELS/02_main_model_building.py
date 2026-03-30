#!/usr/bin/env python3
from __future__ import annotations

# Main estimation pipeline: RA benchmarks, DML, and causal forest.

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from main_model_helpers import (
    build_gate_table,
    build_gates_test_table,
    build_main_results_table,
    fit_blp_heterogeneity_test,
    fit_causal_forest_stage,
    fit_gwl_parametric_model,
    fit_linear_dml_stage,
    save_pipeline_outputs,
)

X_COLS = ["log_totexp", "size", "age", "sex_male"]
DEFAULT_DATA = "data/budgetfood.csv"
DEFAULT_OUTPUT = "outputs/main_model_results"
DEFAULT_SEED = 42
DEFAULT_TOWN_THRESHOLD = 4


# Map the raw sex field to a binary male indicator.
def _encode_sex(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        uniq = set(vals.dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            return vals.astype(float)
        return (vals > vals.median()).astype(float)

    s = series.astype(str).str.strip().str.lower()
    male_tokens = {"male", "man", "m", "1"}
    female_tokens = {"female", "woman", "f", "0"}
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s.isin(male_tokens)] = 1.0
    out[s.isin(female_tokens)] = 0.0
    return out


# Create the common Y/D/X dataset used by every estimator.
def load_and_prepare(csv_path: str | Path, town_threshold: int = 4) -> tuple[pd.DataFrame, list[str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run: python3 data_management.py")

    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"wfood", "totexp", "age", "size", "town", "sex"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[df["totexp"] > 0].copy()

    # Keep the covariate set aligned with the paper and the downstream DML stages.
    df["log_totexp"] = np.log(df["totexp"])
    df["sex_male"] = _encode_sex(df["sex"])
    df["D"] = (pd.to_numeric(df["town"], errors="coerce") >= town_threshold).astype(float)
    df["Y"] = pd.to_numeric(df["wfood"], errors="coerce")

    keep = ["Y", "D"] + X_COLS
    df = df[keep].dropna().reset_index(drop=True)

    if df["D"].nunique() < 2:
        raise ValueError("Treatment has no variation after preprocessing.")

    return df, X_COLS.copy()


# Fit a simple logit model for overlap diagnostics only.
def overlap_diagnostics(df: pd.DataFrame, x_cols: list[str], near_edge_threshold: float = 0.03) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[x_cols].to_numpy()
    d = df["D"].to_numpy(dtype=int)

    # This propensity model is diagnostic only; it is not the main causal estimator.
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(X, d)
    ehat = lr.predict_proba(X)[:, 1]

    ehat_t = ehat[d == 1]
    ehat_c = ehat[d == 0]
    overlap_lb = float(max(np.quantile(ehat_t, 0.05), np.quantile(ehat_c, 0.05)))
    overlap_ub = float(min(np.quantile(ehat_t, 0.95), np.quantile(ehat_c, 0.95)))

    near_zero = float((ehat <= near_edge_threshold).mean())
    near_one = float((ehat >= 1 - near_edge_threshold).mean())

    summary = pd.DataFrame(
        [
            {"metric": "n_obs", "value": float(len(df))},
            {"metric": "treated_n", "value": float(df["D"].sum())},
            {"metric": "control_n", "value": float(len(df) - df["D"].sum())},
            {"metric": "treatment_rate", "value": float(df["D"].mean())},
            {"metric": "propensity_min", "value": float(np.min(ehat))},
            {"metric": "propensity_p01", "value": float(np.quantile(ehat, 0.01))},
            {"metric": "propensity_p99", "value": float(np.quantile(ehat, 0.99))},
            {"metric": "propensity_max", "value": float(np.max(ehat))},
            {"metric": "propensity_overlap_lb_05", "value": overlap_lb},
            {"metric": "propensity_overlap_ub_95", "value": overlap_ub},
            {"metric": f"near_0_share_(<= {near_edge_threshold})", "value": near_zero},
            {"metric": f"near_1_share_(>= {1-near_edge_threshold})", "value": near_one},
        ]
    )
    return summary, ehat


# Estimate the main models and write report-ready outputs.
def run_pipeline(
    csv_path: str | Path = DEFAULT_DATA,
    output_dir: str | Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
    town_threshold: int = DEFAULT_TOWN_THRESHOLD,
) -> None:
    df, x_cols = load_and_prepare(csv_path=csv_path, town_threshold=town_threshold)

    # Store clipped propensity scores for BLP and overlap summaries.
    overlap_summary, ehat = overlap_diagnostics(df, x_cols)
    df["ehat"] = np.clip(ehat, 0.01, 0.99)

    # Use a held-out sample so that ATE and heterogeneity summaries are evaluated out of sample.
    train_df, test_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["D"])
    X_train = train_df[x_cols].to_numpy()
    Y_train = train_df["Y"].to_numpy(dtype=float)
    D_train = train_df["D"].to_numpy(dtype=int)
    X_test = test_df[x_cols].to_numpy()

    # The two GWL fits are the parametric benchmarks.
    gwl_base_model, ate_gwl_base_test, ate_gwl_base_ci_low, ate_gwl_base_ci_high = fit_gwl_parametric_model(
        train_df,
        test_df,
        interactions=False,
    )
    gwl_ext_model, ate_gwl_ext_test, ate_gwl_ext_ci_low, ate_gwl_ext_ci_high = fit_gwl_parametric_model(
        train_df,
        test_df,
        interactions=True,
    )

    # LinearDML keeps a linear final stage but learns nuisance functions flexibly.
    _, tau_dml_test, ate_dml_test, ate_dml_ci = fit_linear_dml_stage(
        X_train,
        Y_train,
        D_train,
        X_test,
        seed,
    )

    # The causal forest is the main flexible heterogeneity estimator.
    cf, tau_cf_test, ate_cf_test, ate_cf_ci = fit_causal_forest_stage(
        X_train,
        Y_train,
        D_train,
        X_test,
        seed,
    )

    # Keep the estimated individual effects so later helpers can build GATE and BLP summaries.
    test_pred = test_df.copy().reset_index(drop=True)
    test_pred["tau_dml"] = tau_dml_test
    test_pred["tau_hat"] = tau_cf_test

    # These objects feed the paper tables: ATE comparison, GATE sorting, and BLP heterogeneity.
    gate_table = build_gate_table(test_pred)
    blp_table = fit_blp_heterogeneity_test(test_pred)
    gates_test = build_gates_test_table(test_pred)
    importance = (
        pd.DataFrame({"feature": x_cols, "importance": cf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    out_dir = Path(output_dir)
    main_results = build_main_results_table(
        ate_gwl_base_test,
        ate_gwl_base_ci_low,
        ate_gwl_base_ci_high,
        ate_gwl_ext_test,
        ate_gwl_ext_ci_low,
        ate_gwl_ext_ci_high,
        ate_dml_test,
        ate_dml_ci,
        ate_cf_test,
        ate_cf_ci,
    )

    save_pipeline_outputs(
        out_dir=out_dir,
        overlap_summary=overlap_summary,
        gate_table=gate_table,
        blp_table=blp_table,
        gates_test=gates_test,
        importance=importance,
        test_pred=test_pred,
        full_df=df,
        main_results=main_results,
        gwl_base_model=gwl_base_model,
        gwl_ext_model=gwl_ext_model,
    )
    print("=== Delgado: DML + Causal Forest ===")
    print(f"N = {len(df)} | treated share = {df['D'].mean():.3f}")
    print(f"ATE (GWL Baseline HC3, test) = {ate_gwl_base_test:.4f} 95%CI[{ate_gwl_base_ci_low:.4f}, {ate_gwl_base_ci_high:.4f}]")
    print(f"ATE (GWL Extended HC3, test) = {ate_gwl_ext_test:.4f} 95%CI[{ate_gwl_ext_ci_low:.4f}, {ate_gwl_ext_ci_high:.4f}]")
    print(f"ATE (LinearDML, test) = {ate_dml_test:.4f} 95%CI[{float(ate_dml_ci[0]):.4f}, {float(ate_dml_ci[1]):.4f}]")
    print(f"ATE (CausalForestDML, test) = {ate_cf_test:.4f} 95%CI[{float(ate_cf_ci[0]):.4f}, {float(ate_cf_ci[1]):.4f}]")
    het_row = blp_table.loc[blp_table["term"] == "heterogeneity_signal"].iloc[0]
    print(f"BLP heterogeneity beta = {het_row['coef']:.4f}, p-value = {het_row['p_value']:.4f}")
    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    run_pipeline(
        csv_path=DEFAULT_DATA,
        output_dir=DEFAULT_OUTPUT,
        seed=DEFAULT_SEED,
        town_threshold=DEFAULT_TOWN_THRESHOLD,
    )
