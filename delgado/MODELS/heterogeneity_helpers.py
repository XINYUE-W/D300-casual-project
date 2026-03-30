from __future__ import annotations

# Helper functions for repeated-split heterogeneity analysis.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


CLAN_LABELS = {
    "Y": "Food share",
    "log_totexp": "Log total expenditure",
    "age": "Age",
    "size": "Household size",
    "sex_male": "Male share",
    "p_hat": "Propensity score",
}


# Apply the shared paper-style plot theme.
def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFA",
            "axes.edgecolor": "#D8D8D2",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#ECECE4",
            "grid.linewidth": 0.9,
            "grid.alpha": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
        }
    )


# Return a display label for grouped heterogeneity plots.
def pretty_variable_name(variable: str) -> str:
    mapping = {
        "log_totexp": "Log Total Expenditure",
        "age": "Age",
        "size": "Household Size",
        "sex_male": "Sex of Reference Person",
    }
    return mapping.get(variable, variable)


# Return concise x-axis labels for grouped plots.
def pretty_group_labels(tmp: pd.DataFrame, variable: str) -> list[str]:
    if variable == "sex_male":
        return ["Female" if g == "0" else "Male" for g in tmp["group"].tolist()]
    return [f"Q{i+1}" for i in range(len(tmp))]


# Force the forest size to a multiple of four for EconML stability.
def normalize_cf_trees(n_trees: int) -> int:
    n_trees = max(40, int(n_trees))
    remainder = n_trees % 4
    if remainder == 0:
        return n_trees
    return n_trees + (4 - remainder)


# Profile treatment effects across quantile bins of one variable.
def profile_by_quantile(df: pd.DataFrame, column: str, q: int = 5) -> pd.DataFrame:
    tmp = df[[column, "tau_hat"]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[column], q=q, duplicates="drop")
    out = (
        tmp.groupby("bin", observed=True)
        .agg(
            variable_mean=(column, "mean"),
            tau_mean=("tau_hat", "mean"),
            tau_std=("tau_hat", "std"),
            n_obs=("tau_hat", "size"),
        )
        .reset_index()
    )
    out["se"] = out["tau_std"] / np.sqrt(out["n_obs"])
    out["ci95_low"] = out["tau_mean"] - 1.96 * out["se"]
    out["ci95_high"] = out["tau_mean"] + 1.96 * out["se"]
    out["group"] = out["bin"].astype(str)
    out["variable"] = column
    return out[["variable", "group", "variable_mean", "tau_mean", "se", "ci95_low", "ci95_high", "n_obs"]]


# Profile treatment effects across categories of one variable.
def profile_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    tmp = df[[column, "tau_hat"]].dropna().copy()
    out = (
        tmp.groupby(column, observed=True)
        .agg(
            variable_mean=(column, "mean"),
            tau_mean=("tau_hat", "mean"),
            tau_std=("tau_hat", "std"),
            n_obs=("tau_hat", "size"),
        )
        .reset_index()
    )
    out["se"] = out["tau_std"] / np.sqrt(out["n_obs"])
    out["ci95_low"] = out["tau_mean"] - 1.96 * out["se"]
    out["ci95_high"] = out["tau_mean"] + 1.96 * out["se"]
    out["group"] = out[column].map(lambda x: str(int(x)) if float(x).is_integer() else str(x))
    out["variable"] = column
    return out[["variable", "group", "variable_mean", "tau_mean", "se", "ci95_low", "ci95_high", "n_obs"]]


# Build a sorted-GATE table from estimated treatment effects.
def build_sorted_gate_table(df: pd.DataFrame, q: int = 4) -> pd.DataFrame:
    tmp = df.copy()
    labels = [f"Q{i}" for i in range(1, q + 1)]
    tmp["effect_group"] = pd.qcut(tmp["tau_hat"], q=q, labels=labels, duplicates="drop")
    out = (
        tmp.groupby("effect_group", observed=True)["tau_hat"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"effect_group": "group", "mean": "tau_mean", "count": "n_obs", "std": "tau_std"})
    )
    out["se"] = out["tau_std"] / np.sqrt(out["n_obs"])
    out["ci95_low"] = out["tau_mean"] - 1.96 * out["se"]
    out["ci95_high"] = out["tau_mean"] + 1.96 * out["se"]
    return out


# Compare the most- and least-affected groups on observed covariates.
def build_clan_table(df: pd.DataFrame, columns: list[str], q: int = 4) -> pd.DataFrame:
    tmp = df.copy()
    labels = [f"Q{i}" for i in range(1, q + 1)]
    tmp["effect_group"] = pd.qcut(tmp["tau_hat"], q=q, labels=labels, duplicates="drop")
    low_group = labels[0]
    high_group = labels[-1]

    rows = []
    for column in columns:
        s = pd.to_numeric(tmp[column], errors="coerce")
        low = s[tmp["effect_group"] == low_group].dropna()
        high = s[tmp["effect_group"] == high_group].dropna()
        low_mean = float(low.mean())
        high_mean = float(high.mean())
        diff = float(high_mean - low_mean)
        se_diff = float(np.sqrt((high.var(ddof=1) / len(high)) + (low.var(ddof=1) / len(low)))) if len(high) > 1 and len(low) > 1 else np.nan
        rows.append(
            {
                "variable": column,
                f"{low_group}_mean": low_mean,
                f"{high_group}_mean": high_mean,
                "difference_high_minus_low": diff,
                "se_difference": se_diff,
                "ci95_low": diff - 1.96 * se_diff if np.isfinite(se_diff) else np.nan,
                "ci95_high": diff + 1.96 * se_diff if np.isfinite(se_diff) else np.nan,
                f"{low_group}_n": int(low.shape[0]),
                f"{high_group}_n": int(high.shape[0]),
            }
        )
    return pd.DataFrame(rows)


# Estimate split-specific scores used in repeated-split diagnostics.
def fit_split_scores(
    df: pd.DataFrame,
    x_cols: list[str],
    seed: int,
    cf_trees: int,
) -> pd.DataFrame:
    # Split once to learn scores on an auxiliary sample and evaluate them on a separate main sample.
    aux_df, main_df = train_test_split(df, test_size=0.50, random_state=seed, stratify=df["D"])

    X_aux = aux_df[x_cols].to_numpy()
    Y_aux = aux_df["Y"].to_numpy(dtype=float)
    D_aux = aux_df["D"].to_numpy(dtype=int)
    X_main = main_df[x_cols].to_numpy()

    cf = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=seed + 11,
            n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=seed + 17,
            n_jobs=-1,
        ),
        n_estimators=normalize_cf_trees(cf_trees),
        min_samples_leaf=20,
        max_depth=10,
        discrete_treatment=True,
        random_state=seed,
        cv=3,
        n_jobs=-1,
    )
    # S_hat is the learned ranking signal from the auxiliary sample.
    cf.fit(Y_aux, D_aux, X=X_aux)
    s_hat = cf.effect(X_main)

    ctrl_aux = aux_df[aux_df["D"] == 0].copy()
    b_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        random_state=seed + 23,
        n_jobs=-1,
    )
    b_model.fit(ctrl_aux[x_cols], ctrl_aux["Y"])
    b_hat = b_model.predict(main_df[x_cols])

    p_model = LogisticRegression(max_iter=2000, solver="lbfgs")
    p_model.fit(X_aux, D_aux)
    p_hat = np.clip(p_model.predict_proba(X_main)[:, 1], 0.01, 0.99)

    out = main_df.copy().reset_index(drop=True)
    out["S_hat"] = s_hat
    out["B_hat"] = b_hat
    out["p_hat"] = p_hat
    # The pseudo-outcome turns the held-out sample into an approximately unbiased treatment-effect target.
    out["pseudo_dr"] = (out["Y"] - out["B_hat"]) * (out["D"] - out["p_hat"]) / (out["p_hat"] * (1.0 - out["p_hat"]))
    return out


# Compute the repeated-split BLP coefficients and 95% intervals.
def blp_from_scores(split_df: pd.DataFrame) -> dict[str, float]:
    p = split_df["p_hat"].to_numpy(dtype=float)
    y = split_df["Y"].to_numpy(dtype=float)
    d = split_df["D"].to_numpy(dtype=float)
    s = split_df["S_hat"].to_numpy(dtype=float)
    b = split_df["B_hat"].to_numpy(dtype=float)
    d_resid = d - p
    s_centered = s - float(np.mean(s))
    # beta_1 captures the average effect; beta_2 tests whether the score ranks heterogeneity correctly.
    X = sm.add_constant(
        pd.DataFrame(
            {
                "baseline_B": b,
                "ate_signal": d_resid,
                "heterogeneity_signal": d_resid * s_centered,
            }
        ),
        has_constant="add",
    )
    w = 1.0 / np.clip(p * (1.0 - p), 1e-6, None)
    model = sm.WLS(y, X, weights=w).fit(cov_type="HC3")
    ci = model.conf_int(alpha=0.05)
    return {
        "ate_coef": float(model.params["ate_signal"]),
        "ate_ci95_low": float(ci.loc["ate_signal", 0]),
        "ate_ci95_high": float(ci.loc["ate_signal", 1]),
        "het_coef": float(model.params["heterogeneity_signal"]),
        "het_ci95_low": float(ci.loc["heterogeneity_signal", 0]),
        "het_ci95_high": float(ci.loc["heterogeneity_signal", 1]),
    }


# Compute sorted-GATE summaries and the Q4-Q1 gap for one split.
def sorted_gate_from_scores(split_df: pd.DataFrame, q: int = 4) -> tuple[pd.DataFrame, dict[str, float]]:
    tmp = split_df.copy()
    labels = [f"Q{i}" for i in range(1, q + 1)]
    tmp["effect_group"] = pd.qcut(tmp["S_hat"], q=q, labels=labels, duplicates="drop")
    rows = []
    # Group the held-out pseudo-outcomes by the learned score to obtain sorted GATEs.
    for group in labels:
        g = tmp[tmp["effect_group"] == group]["pseudo_dr"].dropna()
        est = float(g.mean())
        se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
        rows.append(
            {
                "group": group,
                "tau_mean": est,
                "ci95_low": est - 1.96 * se if np.isfinite(se) else np.nan,
                "ci95_high": est + 1.96 * se if np.isfinite(se) else np.nan,
                "n_obs": int(len(g)),
            }
        )
    q1 = tmp[tmp["effect_group"] == labels[0]]["pseudo_dr"].dropna()
    q4 = tmp[tmp["effect_group"] == labels[-1]]["pseudo_dr"].dropna()
    diff = float(q4.mean() - q1.mean())
    se_diff = float(np.sqrt(q4.var(ddof=1) / len(q4) + q1.var(ddof=1) / len(q1))) if len(q4) > 1 and len(q1) > 1 else np.nan
    diff_row = {
        "group": f"{labels[-1]}_minus_{labels[0]}",
        "tau_mean": diff,
        "ci95_low": diff - 1.96 * se_diff if np.isfinite(se_diff) else np.nan,
        "ci95_high": diff + 1.96 * se_diff if np.isfinite(se_diff) else np.nan,
        "n_obs": int(len(q1) + len(q4)),
    }
    return pd.DataFrame(rows), diff_row


# Compute CLAN summaries for the most- and least-affected groups.
def clan_from_scores(split_df: pd.DataFrame, columns: list[str], q: int = 4) -> pd.DataFrame:
    tmp = split_df.copy()
    labels = [f"Q{i}" for i in range(1, q + 1)]
    tmp["effect_group"] = pd.qcut(tmp["S_hat"], q=q, labels=labels, duplicates="drop")
    low_group = labels[0]
    high_group = labels[-1]
    rows = []
    # CLAN compares observed characteristics of the bottom and top score groups.
    for column in columns:
        s = pd.to_numeric(tmp[column], errors="coerce")
        low = s[tmp["effect_group"] == low_group].dropna()
        high = s[tmp["effect_group"] == high_group].dropna()
        low_mean = float(low.mean())
        high_mean = float(high.mean())
        low_se = float(low.std(ddof=1) / np.sqrt(len(low))) if len(low) > 1 else np.nan
        high_se = float(high.std(ddof=1) / np.sqrt(len(high))) if len(high) > 1 else np.nan
        diff = float(high_mean - low_mean)
        se_diff = float(np.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low))) if len(high) > 1 and len(low) > 1 else np.nan
        rows.append(
            {
                "variable": column,
                f"{low_group}_mean": low_mean,
                f"{low_group}_ci95_low": low_mean - 1.96 * low_se if np.isfinite(low_se) else np.nan,
                f"{low_group}_ci95_high": low_mean + 1.96 * low_se if np.isfinite(low_se) else np.nan,
                f"{high_group}_mean": high_mean,
                f"{high_group}_ci95_low": high_mean - 1.96 * high_se if np.isfinite(high_se) else np.nan,
                f"{high_group}_ci95_high": high_mean + 1.96 * high_se if np.isfinite(high_se) else np.nan,
                "difference_high_minus_low": diff,
                "diff_ci95_low": diff - 1.96 * se_diff if np.isfinite(se_diff) else np.nan,
                "diff_ci95_high": diff + 1.96 * se_diff if np.isfinite(se_diff) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# Collapse repeated-split outputs to column-wise medians.
def aggregate_median_interval(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, as_index=False)[value_cols].median().reset_index(drop=True)


# Format the repeated-split CLAN summary for CSV and LaTeX output.
def format_repeated_clan_table(clan_summary: pd.DataFrame) -> pd.DataFrame:
    tmp = clan_summary.copy()
    tmp["Characteristic"] = tmp["variable"].map(CLAN_LABELS).fillna(tmp["variable"])
    tmp["Most affected (Q1)"] = tmp["Q1_mean"].map(lambda x: f"{x:.3f}")
    tmp["Least affected (Q4)"] = tmp["Q4_mean"].map(lambda x: f"{x:.3f}")
    tmp["Difference (Q4 - Q1)"] = tmp["difference_high_minus_low"].map(lambda x: f"{x:.3f}")
    tmp["95% CI for difference"] = tmp.apply(
        lambda row: f"[{row['diff_ci95_low']:.3f}, {row['diff_ci95_high']:.3f}]",
        axis=1,
    )
    order = [
        "Food share",
        "Log total expenditure",
        "Age",
        "Household size",
        "Male share",
        "Propensity score",
    ]
    tmp["Characteristic"] = pd.Categorical(tmp["Characteristic"], categories=order, ordered=True)
    return (
        tmp.sort_values("Characteristic")
        [
            [
                "Characteristic",
                "Most affected (Q1)",
                "Least affected (Q4)",
                "Difference (Q4 - Q1)",
                "95% CI for difference",
            ]
        ]
        .reset_index(drop=True)
    )


# Build the score used in permutation-based importance checks.
def build_dr_heterogeneity_score(
    df: pd.DataFrame,
    x_cols: list[str],
    seed: int,
    cf_trees: int,
    permute_score: bool = False,
) -> np.ndarray:
    X = df[x_cols].to_numpy()
    Y = df["Y"].to_numpy(dtype=float)
    D = df["D"].to_numpy(dtype=int)

    cf = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=10,
            random_state=seed + 101,
            n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=10,
            random_state=seed + 107,
            n_jobs=-1,
        ),
        n_estimators=normalize_cf_trees(cf_trees),
        min_samples_leaf=20,
        max_depth=10,
        discrete_treatment=True,
        random_state=seed,
        cv=3,
        n_jobs=-1,
    )
    cf.fit(Y, D, X=X)
    tau_hat = np.asarray(cf.effect(X), dtype=float)

    treated_mask = D == 1
    control_mask = D == 0

    m1_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=seed + 211,
        n_jobs=-1,
    )
    m1_model.fit(X[treated_mask], Y[treated_mask])
    m1_hat = np.asarray(m1_model.predict(X), dtype=float)

    m0_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=seed + 223,
        n_jobs=-1,
    )
    m0_model.fit(X[control_mask], Y[control_mask])
    m0_hat = np.asarray(m0_model.predict(X), dtype=float)

    e_model = LogisticRegression(max_iter=2000, solver="lbfgs")
    e_model.fit(X, D)
    e_hat = np.clip(e_model.predict_proba(X)[:, 1], 0.01, 0.99)

    gamma_hat = tau_hat + (D * (Y - m1_hat) / e_hat) - ((1 - D) * (Y - m0_hat) / (1.0 - e_hat))
    if permute_score:
        rng = np.random.default_rng(seed)
        gamma_hat = rng.permutation(gamma_hat)
    return np.asarray(gamma_hat, dtype=float)


# Fit the auxiliary forest used for score-based importance.
def fit_score_forest_importance(
    df: pd.DataFrame,
    x_cols: list[str],
    score: np.ndarray,
    seed: int,
) -> np.ndarray:
    X = df[x_cols].to_numpy()
    score_forest = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        random_state=seed + 307,
        n_jobs=-1,
    )
    score_forest.fit(X, score)
    return np.asarray(score_forest.feature_importances_, dtype=float)


# Run the permutation benchmark for score-based importance.
def run_permutation_importance(
    df: pd.DataFrame,
    x_cols: list[str],
    n_perm: int,
    base_seed: int,
    cf_trees: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Start from the observed heterogeneity score, then compare it with permuted score draws.
    observed_score = build_dr_heterogeneity_score(
        df,
        x_cols=x_cols,
        seed=base_seed,
        cf_trees=cf_trees,
        permute_score=False,
    )
    observed = fit_score_forest_importance(df, x_cols=x_cols, score=observed_score, seed=base_seed)
    perm_rows = []
    for perm_id in range(n_perm):
        permuted_score = build_dr_heterogeneity_score(
            df,
            x_cols=x_cols,
            seed=base_seed + 1000 + perm_id,
            cf_trees=cf_trees,
            permute_score=True,
        )
        perm_imp = fit_score_forest_importance(
            df,
            x_cols=x_cols,
            score=permuted_score,
            seed=base_seed + 2000 + perm_id,
        )
        perm_rows.append({"perm_id": perm_id, **{col: float(val) for col, val in zip(x_cols, perm_imp)}})

    perm_df = pd.DataFrame(perm_rows)
    summary_rows = []
    for i, col in enumerate(x_cols):
        obs = float(observed[i])
        exceed_count = int((perm_df[col] >= obs).sum())
        perm_p_value = float((1 + exceed_count) / (n_perm + 1))
        summary_rows.append(
            {
                "feature": col,
                "observed_importance": obs,
                "perm_median": float(perm_df[col].median()),
                "perm_p95": float(perm_df[col].quantile(0.95)),
                "perm_p_value": perm_p_value,
                "adjusted_importance": float(-np.log(max(perm_p_value, 1e-12))),
                "n_permutations": int(n_perm),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("observed_importance", ascending=False).reset_index(drop=True)
    return summary, perm_df


# Run repeated sample splitting and aggregate the main heterogeneity outputs.
def run_repeated_sample_splitting(
    df: pd.DataFrame,
    x_cols: list[str],
    n_splits: int,
    base_seed: int,
    cf_trees: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blp_rows: list[dict[str, float]] = []
    gate_rows: list[dict[str, float | str]] = []
    clan_rows: list[pd.DataFrame] = []

    # Aggregate across many splits so the final tables reflect stable medians rather than one random partition.
    for split_id in range(n_splits):
        split_seed = base_seed + split_id
        split_scores = fit_split_scores(df, x_cols, seed=split_seed, cf_trees=cf_trees)

        blp_row = blp_from_scores(split_scores)
        blp_row["split_id"] = split_id
        blp_rows.append(blp_row)

        gate_df, diff_row = sorted_gate_from_scores(split_scores, q=4)
        gate_df["split_id"] = split_id
        gate_rows.extend(gate_df.to_dict("records"))
        diff_row["split_id"] = split_id
        gate_rows.append(diff_row)

        clan_df = clan_from_scores(split_scores, columns=["Y", "log_totexp", "age", "size", "sex_male", "p_hat"], q=4)
        clan_df["split_id"] = split_id
        clan_rows.append(clan_df)

    blp_all = pd.DataFrame(blp_rows)
    gate_all = pd.DataFrame(gate_rows)
    clan_all = pd.concat(clan_rows, ignore_index=True)

    blp_summary = pd.DataFrame(
        [
            {
                "term": "ATE_beta1",
                "estimate_median": float(blp_all["ate_coef"].median()),
                "ci95_low_median": float(blp_all["ate_ci95_low"].median()),
                "ci95_high_median": float(blp_all["ate_ci95_high"].median()),
                "n_splits": int(n_splits),
            },
            {
                "term": "HET_beta2",
                "estimate_median": float(blp_all["het_coef"].median()),
                "ci95_low_median": float(blp_all["het_ci95_low"].median()),
                "ci95_high_median": float(blp_all["het_ci95_high"].median()),
                "n_splits": int(n_splits),
            },
        ]
    )

    gate_summary = aggregate_median_interval(
        gate_all,
        group_cols=["group"],
        value_cols=["tau_mean", "ci95_low", "ci95_high", "n_obs"],
    )
    clan_summary = aggregate_median_interval(
        clan_all,
        group_cols=["variable"],
        value_cols=[
            "Q1_mean",
            "Q1_ci95_low",
            "Q1_ci95_high",
            "Q4_mean",
            "Q4_ci95_low",
            "Q4_ci95_high",
            "difference_high_minus_low",
            "diff_ci95_low",
            "diff_ci95_high",
        ],
    )
    return blp_summary, gate_summary, clan_summary


# Plot the single-split GATE figure from a saved CSV.
def plot_gates(gates_path, out_path) -> None:
    apply_plot_style()
    gates = pd.read_csv(gates_path)
    gates = gates[gates["group"].isin(["q1", "q2", "q3", "q4", "q5"])].copy()

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    x = np.arange(len(gates))
    y = gates["mean"].to_numpy()
    yerr = 1.96 * gates["se"].to_numpy()
    ax.fill_between(x, y - yerr, y + yerr, color="#FFD9CC", alpha=0.9, zorder=1)
    ax.plot(x, y, color="#D1495B", lw=3.2, zorder=3)
    ax.scatter(x, y, s=110, color="#D1495B", edgecolor="white", linewidth=1.6, zorder=4)
    ax.axhline(0.0, color="#4B4B4B", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_ylabel("Average Predicted Treatment Effect")
    ax.set_xlabel("Predicted CATE Quintile")
    ax.set_title("GATES Across Predicted Effect Groups", pad=14)
    ax.set_ylim(min(y - yerr) - 0.004, max(y + yerr) + 0.004)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.0012, f"{yi:.3f}", ha="center", va="bottom", fontsize=10, color="#7A2E3A")
    ax.text(0.98, 0.06, "Shaded band: 95% CI", transform=ax.transAxes, ha="right", va="bottom", color="#6A6A6A", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Plot the repeated-split sorted GATE figure.
def plot_sorted_gates(sorted_gate_df: pd.DataFrame, out_path) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_facecolor("none")
    x = np.arange(len(sorted_gate_df))
    y = sorted_gate_df["tau_mean"].to_numpy()
    lower = sorted_gate_df["ci95_low"].to_numpy()
    upper = sorted_gate_df["ci95_high"].to_numpy()

    ci_half = (upper - lower) / 2.0
    main_color = "#1F4E79"
    band_color = "#C7D3DD"
    accent_color = "#8C2D19"

    ax.fill_between(x, lower, upper, color=band_color, alpha=0.35, zorder=1)
    ax.errorbar(
        x,
        y,
        yerr=ci_half,
        fmt="none",
        ecolor="#5C6770",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=2,
    )
    ax.plot(x, y, color=main_color, lw=2.0, zorder=3)
    ax.scatter(x, y, s=46, color=main_color, edgecolor="white", linewidth=0.9, zorder=4)
    ax.axhline(0.0, color="#6E6E6E", lw=0.9, linestyle="--", alpha=0.9, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_gate_df["group"].tolist())
    ax.set_ylabel("Estimated Group Average Treatment Effect")
    ax.set_xlabel("Sorted effect group")
    ax.set_ylim(min(lower) - 0.006, max(upper) + 0.006)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    for xi, yi in zip(x, y):
        ax.text(
            xi,
            yi + 0.0011,
            f"{yi:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=main_color,
        )

    if len(sorted_gate_df) >= 2:
        gap = float(y[-1] - y[0])
        bracket_y = max(y[0], y[-1]) + 0.0035
        ax.plot([x[0], x[-1]], [bracket_y, bracket_y], color=accent_color, lw=1.2, zorder=5)
        ax.plot([x[0], x[0]], [bracket_y - 0.0007, bracket_y], color=accent_color, lw=1.2, zorder=5)
        ax.plot([x[-1], x[-1]], [bracket_y - 0.0007, bracket_y], color=accent_color, lw=1.2, zorder=5)
        ax.text(
            (x[0] + x[-1]) / 2.0,
            bracket_y + 0.0009,
            f"Q4 - Q1 = {gap:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=accent_color,
        )

    ax.text(
        0.98,
        0.04,
        "Vertical bars denote 95% confidence intervals",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=8.8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# Plot covariate profiles against treatment effects.
def plot_profiles(profile_df: pd.DataFrame, out_path) -> None:
    apply_plot_style()
    variables = profile_df["variable"].unique().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.8))
    axes = axes.flatten()

    color_map = {
        "log_totexp": "#2F6690",
        "age": "#4F772D",
        "size": "#E07A1F",
        "sex_male": "#B56576",
    }

    for ax, variable in zip(axes, variables):
        tmp = profile_df[profile_df["variable"] == variable].reset_index(drop=True)
        x = np.arange(len(tmp))
        y = tmp["tau_mean"].to_numpy()
        yerr = 1.96 * tmp["se"].to_numpy()
        color = color_map.get(variable, "#333333")
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.18, zorder=1)
        ax.plot(x, y, color=color, lw=3.0, zorder=3)
        ax.scatter(x, y, s=95, color=color, edgecolor="white", linewidth=1.3, zorder=4)
        ax.axhline(0.0, color="#4B4B4B", lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_group_labels(tmp, variable))
        ax.tick_params(axis="x", pad=6)
        ax.set_title(pretty_variable_name(variable), pad=10)
        ax.set_ylabel("Average Predicted Treatment Effect")
        ax.set_xlabel("Group")
        ax.set_ylim(min(y - yerr) - 0.004, max(y + yerr) + 0.004)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.0011, f"{yi:.3f}", ha="center", va="bottom", fontsize=9, color=color)
        if variable != "sex_male":
            means = tmp["variable_mean"].round(2).tolist()
            mean_text = "Group means: " + " | ".join([f"{label}={mean_val}" for label, mean_val in zip(pretty_group_labels(tmp, variable), means)])
            ax.text(
                0.02,
                0.04,
                mean_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                color="#6A6A6A",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
            )

    fig.suptitle("Predicted Treatment Effects by Covariate Group", y=0.98, fontsize=17, fontweight="bold")
    fig.subplots_adjust(top=0.90, hspace=0.40, wspace=0.24, bottom=0.10)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Plot observed and permutation-based variable importance.
def plot_permutation_importance(perm_summary: pd.DataFrame, out_path) -> None:
    apply_plot_style()
    tmp = perm_summary.sort_values("observed_importance", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    y = np.arange(len(tmp))
    ax.barh(y, tmp["observed_importance"], color="#355C7D", alpha=0.92, label="Observed importance")
    ax.scatter(tmp["perm_p95"], y, color="#E07A1F", s=70, label="95th percentile under permutation", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_variable_name(v) for v in tmp["feature"]])
    ax.set_xlabel("Feature importance")
    ax.set_title("Permutation-Adjusted Variable Importance", pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
