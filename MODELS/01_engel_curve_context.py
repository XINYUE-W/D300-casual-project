#!/usr/bin/env python3
from __future__ import annotations

# Background plots linking the data to the Engel-curve literature.

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


# Load the context sample using the same variable definitions as the main pipeline.
def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"wfood", "totexp", "age", "size", "town", "sex"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[df["totexp"] > 0].copy()
    df["log_totexp"] = np.log(df["totexp"])
    df["sex_male"] = df["sex"].astype(str).str.strip().str.lower().isin(["man", "male", "m", "1"]).astype(float)
    df["urban"] = (pd.to_numeric(df["town"], errors="coerce") >= 4).astype(float)
    return df[["wfood", "totexp", "log_totexp", "age", "size", "town", "sex_male", "urban"]].dropna().reset_index(drop=True)


# Fit a GWL-style Engel curve for descriptive context only.
def fit_gwl_curve(df: pd.DataFrame) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    design = pd.DataFrame(
        {
            "log_totexp": df["log_totexp"],
            "log_totexp_sq": df["log_totexp"] ** 2,
            "age": df["age"],
            "size": df["size"],
            "town": df["town"],
            "sex_male": df["sex_male"],
        }
    )
    model = sm.OLS(df["wfood"], sm.add_constant(design, has_constant="add")).fit(cov_type="HC3")

    x_grid = np.linspace(df["log_totexp"].quantile(0.01), df["log_totexp"].quantile(0.99), 200)
    ref = {
        "age": float(df["age"].median()),
        "size": float(df["size"].median()),
        "town": float(df["town"].median()),
        "sex_male": float(df["sex_male"].median()),
    }
    pred_df = pd.DataFrame(
        {
            "log_totexp": x_grid,
            "log_totexp_sq": x_grid**2,
            "age": ref["age"],
            "size": ref["size"],
            "town": ref["town"],
            "sex_male": ref["sex_male"],
        }
    )
    pred = model.get_prediction(sm.add_constant(pred_df, has_constant="add")).summary_frame(alpha=0.05)
    curve = pred_df[["log_totexp"]].copy()
    curve["gwl_fit"] = pred["mean"].to_numpy()
    curve["gwl_ci_low"] = pred["mean_ci_lower"].to_numpy()
    curve["gwl_ci_high"] = pred["mean_ci_upper"].to_numpy()
    return curve, model


# Fit a LOWESS reference curve.
def fit_lowess_curve(df: pd.DataFrame) -> pd.DataFrame:
    smooth = lowess(
        endog=df["wfood"].to_numpy(),
        exog=df["log_totexp"].to_numpy(),
        frac=0.2,
        it=0,
        return_sorted=True,
    )
    return pd.DataFrame(smooth, columns=["log_totexp", "lowess_fit"])


# Plot the raw scatter together with the GWL and LOWESS fits.
def plot_engel_curve(df: pd.DataFrame, gwl_curve: pd.DataFrame, lowess_curve: pd.DataFrame, out_path: Path) -> None:
    sample = df.sample(min(len(df), 5000), random_state=42).copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sample["log_totexp"], sample["wfood"], s=8, alpha=0.18, color="#4C6A92", label="Sample observations")
    ax.plot(lowess_curve["log_totexp"], lowess_curve["lowess_fit"], color="#D1495B", lw=2.5, label="LOWESS smooth")
    ax.plot(gwl_curve["log_totexp"], gwl_curve["gwl_fit"], color="#2A9D8F", lw=2.5, label="GWL-style fit")
    ax.fill_between(
        gwl_curve["log_totexp"],
        gwl_curve["gwl_ci_low"],
        gwl_curve["gwl_ci_high"],
        color="#2A9D8F",
        alpha=0.15,
        label="GWL 95% CI",
    )
    ax.set_title("Food Share and Log Expenditure")
    ax.set_xlabel("log(total expenditure)")
    ax.set_ylabel("food share (wfood)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Run the context analysis and save the outputs.
def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "outputs" / "context_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(project_root / "data" / "budgetfood.csv")
    gwl_curve, model = fit_gwl_curve(df)
    lowess_curve = fit_lowess_curve(df)

    gwl_curve.to_csv(out_dir / "gwl_curve.csv", index=False)
    lowess_curve.to_csv(out_dir / "lowess_curve.csv", index=False)
    with (out_dir / "gwl_summary.txt").open("w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    plot_engel_curve(df, gwl_curve, lowess_curve, out_dir / "engel_curve_gwl_vs_lowess.png")

    print("Saved Engel-curve context outputs to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
