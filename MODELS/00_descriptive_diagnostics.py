#!/usr/bin/env python3
from __future__ import annotations

# EDA and basic diagnostics for the project.

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

# Keep notebook display interactive; use Agg only for script runs.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESEARCH_VARIABLES = ["wfood", "urban", "log_totexp", "age", "size", "sex_male"]
VARIABLE_LABELS = {
    "wfood": "Food Share (wfood)",
    "urban": "Urban Treatment",
    "log_totexp": "Log Total Expenditure",
    "age": "Age",
    "size": "Household Size",
    "sex_male": "Male Reference Person",
}


# Load the raw file and validate the required columns.
def load_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}.")

    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"wfood", "totexp", "age", "size", "town", "sex"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


# Create the study variables used in both EDA and estimation.
def prepare_features(df: pd.DataFrame, town_threshold: int = 4) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.loc[out["totexp"] > 0].copy()
    out["log_totexp"] = np.log(out["totexp"])
    sex = out["sex"].astype(str).str.strip().str.lower()
    out["sex_male"] = np.where(sex.isin(["man", "male", "m", "1"]), 1.0, np.nan)
    out.loc[sex.isin(["woman", "female", "f", "0"]), "sex_male"] = 0.0
    out["urban"] = (pd.to_numeric(out["town"], errors="coerce") >= town_threshold).astype(float)
    return out


# Return full-sample descriptive statistics for the study variables.
def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in RESEARCH_VARIABLES:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "variable": col,
                "label": VARIABLE_LABELS.get(col, col),
                "n_non_missing": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


# Return full-sample, treated, and control summaries.
def build_grouped_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "full_sample": df.index,
        "urban": df.index[df["urban"] == 1],
        "rural": df.index[df["urban"] == 0],
    }
    rows: list[dict[str, float | str]] = []
    for col in RESEARCH_VARIABLES:
        row: dict[str, float | str] = {
            "variable": col,
            "label": VARIABLE_LABELS.get(col, col),
            "n_non_missing": int(pd.to_numeric(df[col], errors="coerce").notna().sum()),
        }
        for group_name, idx in groups.items():
            s = pd.to_numeric(df.loc[idx, col], errors="coerce")
            row[f"{group_name}_mean"] = float(s.mean())
            row[f"{group_name}_std"] = float(s.std())
        rows.append(row)
    return pd.DataFrame(rows)


# Summarize missingness by variable.
def build_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "variable": df.columns,
            "missing_n": [int(df[c].isna().sum()) for c in df.columns],
            "missing_share": [float(df[c].isna().mean()) for c in df.columns],
        }
    )
    return out.sort_values(["missing_n", "variable"], ascending=[False, True]).reset_index(drop=True)


# Compute the standardized mean difference.
def _std_mean_diff(treated: pd.Series, control: pd.Series) -> float:
    vt = treated.var(ddof=1)
    vc = control.var(ddof=1)
    pooled = np.sqrt((vt + vc) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((treated.mean() - control.mean()) / pooled)


# Compute raw treated-control contrasts and standardized mean differences.
def build_balance_table(df: pd.DataFrame, covariates: list[str], treatment_col: str = "urban") -> pd.DataFrame:
    treated_mask = df[treatment_col] == 1
    control_mask = df[treatment_col] == 0
    rows: list[dict[str, float | str]] = []
    for col in covariates:
        s = pd.to_numeric(df[col], errors="coerce")
        treated = s[treated_mask & s.notna()]
        control = s[control_mask & s.notna()]
        rows.append(
            {
                "variable": col,
                "treated_mean": float(treated.mean()),
                "control_mean": float(control.mean()),
                "mean_diff": float(treated.mean() - control.mean()),
                "std_mean_diff": _std_mean_diff(treated, control),
                "treated_n": int(treated.shape[0]),
                "control_n": int(control.shape[0]),
            }
        )
    return pd.DataFrame(rows)


# Return compact sample facts for the notebook and appendix.
def build_sample_overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "n_total_raw", "value": float(len(df))},
            {"metric": "n_nonmissing_wfood", "value": float(df["wfood"].notna().sum())},
            {"metric": "n_nonmissing_totexp", "value": float(df["totexp"].notna().sum())},
            {"metric": "treatment_n_urban", "value": float((df["urban"] == 1).sum())},
            {"metric": "control_n_rural", "value": float((df["urban"] == 0).sum())},
            {"metric": "treatment_share_urban", "value": float((df["urban"] == 1).mean())},
        ]
    )


# Summarize the original town categories that define treatment.
def build_town_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df["town"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("town")
        .reset_index(name="count")
    )
    out["share"] = out["count"] / out["count"].sum()
    return out


# Save the descriptive and diagnostic plot panels.
def save_plots(df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("ggplot")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    axes[0].hist(df.loc[df["urban"] == 0, "log_totexp"].dropna(), bins=35, alpha=0.75, label="rural", color="#33658A")
    axes[0].hist(df.loc[df["urban"] == 1, "log_totexp"].dropna(), bins=35, alpha=0.75, label="urban", color="#F26419")
    axes[0].set_title("Expenditure by Treatment")
    axes[0].set_xlabel("log_totexp")
    axes[0].legend()

    rural_wfood = df.loc[df["urban"] == 0, "wfood"].dropna()
    urban_wfood = df.loc[df["urban"] == 1, "wfood"].dropna()
    axes[1].boxplot([rural_wfood, urban_wfood], tick_labels=["rural", "urban"], patch_artist=True)
    axes[1].set_title("Outcome by Treatment")
    axes[1].set_ylabel("wfood")

    axes[2].hist(df.loc[df["urban"] == 0, "age"].dropna(), bins=30, alpha=0.75, label="rural", color="#33658A")
    axes[2].hist(df.loc[df["urban"] == 1, "age"].dropna(), bins=30, alpha=0.75, label="urban", color="#F26419")
    axes[2].set_title("Age by Treatment")
    axes[2].set_xlabel("age")
    axes[2].legend()

    axes[3].hist(df.loc[df["urban"] == 0, "size"].dropna(), bins=np.arange(0.5, df["size"].max() + 1.5, 1), alpha=0.75, label="rural", color="#33658A")
    axes[3].hist(df.loc[df["urban"] == 1, "size"].dropna(), bins=np.arange(0.5, df["size"].max() + 1.5, 1), alpha=0.75, label="urban", color="#F26419")
    axes[3].set_title("Household Size by Treatment")
    axes[3].set_xlabel("size")
    axes[3].legend()

    sex_by_group = (
        df.groupby("urban")["sex_male"]
        .mean()
        .reindex([0.0, 1.0])
    )
    axes[4].bar(["rural", "urban"], sex_by_group.values, color=["#33658A", "#F26419"])
    axes[4].set_title("Male Share by Treatment")
    axes[4].set_ylabel("share")

    axes[5].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "eda_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Run the descriptive diagnostics workflow and save outputs.
def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    data_csv = project_root / "data" / "budgetfood.csv"
    out_dir = project_root / "outputs" / "eda_output"

    raw = load_data(data_csv)
    df = prepare_features(raw, town_threshold=4)
    analysis_df = df[["wfood", "totexp", "log_totexp", "age", "size", "town", "sex_male", "urban"]].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    save_plots(analysis_df, out_dir)

    print("Saved EDA outputs to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
