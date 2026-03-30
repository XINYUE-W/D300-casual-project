from __future__ import annotations

# Helper functions for the main estimation pipeline.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import CausalForestDML, LinearDML
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

TAU_QUINTILE_LABELS = ["q1", "q2", "q3", "q4", "q5"]
BASE_X_COLS = ["log_totexp", "size", "age", "sex_male"]


# Match Matplotlib figures to the paper's serif style.
def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CFCFC7",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E8E8E0",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
        }
    )


# Summarize estimated treatment effects by causal-forest quintile.
def build_gate_table(test_pred: pd.DataFrame) -> pd.DataFrame:
    tmp = test_pred.copy()
    tmp["tau_quintile"] = pd.qcut(tmp["tau_hat"], q=5, labels=TAU_QUINTILE_LABELS, duplicates="drop")
    out = tmp.groupby("tau_quintile", observed=True)["tau_hat"].agg(["mean", "count"]).reset_index()
    return out.rename(columns={"tau_quintile": "group", "mean": "gate_tau_hat", "count": "n_obs"})


# Run the single-split BLP test on the held-out sample.
def fit_blp_heterogeneity_test(test_pred: pd.DataFrame) -> pd.DataFrame:
    p = np.clip(test_pred["ehat"].to_numpy(dtype=float), 0.01, 0.99)
    y = test_pred["Y"].to_numpy(dtype=float)
    d = test_pred["D"].to_numpy(dtype=float)
    s = test_pred["tau_hat"].to_numpy(dtype=float)
    s_centered = s - float(np.mean(s))
    d_resid = d - p
    interaction = d_resid * s_centered

    # Estimate the baseline outcome surface on controls only, as in the BLP-style decomposition.
    ctrl = test_pred[test_pred["D"] == 0]
    b_model = sm.OLS(ctrl["Y"], sm.add_constant(ctrl[BASE_X_COLS], has_constant="add")).fit()
    b_hat = b_model.predict(sm.add_constant(test_pred[BASE_X_COLS], has_constant="add")).to_numpy()

    X = sm.add_constant(
        pd.DataFrame(
            {
                "baseline_B": b_hat,
                "d_resid": d_resid,
                "heterogeneity_signal": interaction,
            }
        ),
        has_constant="add",
    )
    # Reweight by inverse treatment variance so the residualized treatment term is well scaled.
    w = 1.0 / np.clip(p * (1.0 - p), 1e-6, None)
    model = sm.WLS(y, X, weights=w).fit(cov_type="HC3")

    terms = ["const", "baseline_B", "d_resid", "heterogeneity_signal"]
    out = pd.DataFrame(
        [
            {
                "term": term,
                "coef": float(model.params.get(term, np.nan)),
                "se": float(model.bse.get(term, np.nan)),
                "p_value": float(model.pvalues.get(term, np.nan)),
            }
            for term in terms
        ]
    )
    out["is_heterogeneity_term"] = out["term"].eq("heterogeneity_signal")
    return out


# Build single-split GATE summaries and the Q5-Q1 contrast.
def build_gates_test_table(test_pred: pd.DataFrame) -> pd.DataFrame:
    tmp = test_pred.copy()
    tmp["tau_quintile"] = pd.qcut(tmp["tau_hat"], q=5, labels=TAU_QUINTILE_LABELS, duplicates="drop")
    grp = tmp.groupby("tau_quintile", observed=True)["tau_hat"]
    out = grp.agg(["mean", "count", "std"]).reset_index().rename(columns={"tau_quintile": "group"})
    out["se"] = out["std"] / np.sqrt(out["count"])
    out["ci95_low"] = out["mean"] - 1.96 * out["se"]
    out["ci95_high"] = out["mean"] + 1.96 * out["se"]

    # Report the end-to-end spread because that is the cleanest summary of ranked heterogeneity.
    q1 = tmp.loc[tmp["tau_quintile"] == TAU_QUINTILE_LABELS[0], "tau_hat"]
    q5 = tmp.loc[tmp["tau_quintile"] == TAU_QUINTILE_LABELS[-1], "tau_hat"]
    if len(q1) > 1 and len(q5) > 1:
        diff = float(q5.mean() - q1.mean())
        se_diff = float(np.sqrt(q5.var(ddof=1) / len(q5) + q1.var(ddof=1) / len(q1)))
        z = diff / se_diff if se_diff > 0 else np.nan
        p_value = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    else:
        diff, se_diff, z, p_value = np.nan, np.nan, np.nan, np.nan

    summary = pd.DataFrame(
        [
            {
                "group": "q5_minus_q1",
                "mean": diff,
                "count": int(len(q1) + len(q5)),
                "std": np.nan,
                "se": se_diff,
                "ci95_low": diff - 1.96 * se_diff if np.isfinite(se_diff) else np.nan,
                "ci95_high": diff + 1.96 * se_diff if np.isfinite(se_diff) else np.nan,
                "z_stat": z,
                "p_value": p_value,
            }
        ]
    )
    return pd.concat([out, summary], ignore_index=True)


# Fit the GWL-style benchmark and return the test-sample ATE.
def fit_gwl_parametric_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interactions: bool = False,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, float, float, float]:
    train = train_df.copy()
    test = test_df.copy()

    train["log_totexp_sq"] = train["log_totexp"] ** 2
    test["log_totexp_sq"] = test["log_totexp"] ** 2

    base_terms = ["D", "log_totexp", "log_totexp_sq", "size", "age", "sex_male"]
    interaction_terms: list[str] = []
    if interactions:
        for c in ["log_totexp", "log_totexp_sq", "size", "age", "sex_male"]:
            train[f"D_x_{c}"] = train["D"] * train[c]
            test[f"D_x_{c}"] = test["D"] * test[c]
            interaction_terms.append(f"D_x_{c}")
        base_terms.extend(interaction_terms)

    X_train = sm.add_constant(train[base_terms], has_constant="add")
    gwl = sm.OLS(train["Y"], X_train).fit(cov_type="HC3")

    # Predict the same test sample twice, once treated and once untreated, then average the difference.
    X1 = test.copy()
    X0 = test.copy()
    X1["D"] = 1.0
    X0["D"] = 0.0
    if interactions:
        for c in ["log_totexp", "log_totexp_sq", "size", "age", "sex_male"]:
            X1[f"D_x_{c}"] = X1[c]
            X0[f"D_x_{c}"] = 0.0

    X1 = sm.add_constant(X1[base_terms], has_constant="add")
    X0 = sm.add_constant(X0[base_terms], has_constant="add")

    delta = (X1 - X0).to_numpy()
    g = delta.mean(axis=0)
    beta = gwl.params.reindex(X1.columns).to_numpy()
    cov = gwl.cov_params().reindex(index=X1.columns, columns=X1.columns).to_numpy()

    ate = float(g @ beta)
    se = float(np.sqrt(g @ cov @ g))
    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se
    return gwl, ate, ci_low, ci_high


# Build the nuisance models shared by the DML stages.
def build_nuisance_models(seed: int) -> tuple[RandomForestRegressor, RandomForestClassifier]:
    model_y = RandomForestRegressor(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=20,
        random_state=seed + 11,
        n_jobs=-1,
    )
    model_t = RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=20,
        random_state=seed + 17,
        n_jobs=-1,
    )
    return model_y, model_t


# Fit LinearDML and return test-sample treatment effects and ATE.
def fit_linear_dml_stage(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    D_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[LinearDML, np.ndarray, float, tuple[float, float]]:
    model_y, model_t = build_nuisance_models(seed)
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        random_state=seed,
        cv=5,
    )
    # ATE is the mean of the estimated conditional effects on the held-out sample.
    dml.fit(Y_train, D_train, X=X_train)
    tau_dml_test = dml.effect(X_test)
    ate_dml_test = float(np.mean(tau_dml_test))
    ate_dml_ci = dml.ate_interval(X=X_test, alpha=0.05)
    return dml, tau_dml_test, ate_dml_test, (float(ate_dml_ci[0]), float(ate_dml_ci[1]))


# Fit CausalForestDML and return test-sample treatment effects and ATE.
def fit_causal_forest_stage(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    D_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[CausalForestDML, np.ndarray, float, tuple[float, float]]:
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=10, random_state=seed + 23, n_jobs=-1),
        model_t=RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10, random_state=seed + 29, n_jobs=-1),
        n_estimators=1200,
        min_samples_leaf=20,
        max_depth=10,
        discrete_treatment=True,
        random_state=seed,
        cv=3,
        n_jobs=-1,
    )
    # This is the flexible model used later for heterogeneity ranking and CLAN summaries.
    cf.fit(Y_train, D_train, X=X_train)
    tau_cf_test = cf.effect(X_test)
    ate_cf_test = float(np.mean(tau_cf_test))
    ate_cf_ci = cf.ate_interval(X=X_test, alpha=0.05)
    return cf, tau_cf_test, ate_cf_test, (float(ate_cf_ci[0]), float(ate_cf_ci[1]))


# Assemble the main comparison table.
def build_main_results_table(
    ate_gwl_base_test: float,
    ate_gwl_base_ci_low: float,
    ate_gwl_base_ci_high: float,
    ate_gwl_ext_test: float,
    ate_gwl_ext_ci_low: float,
    ate_gwl_ext_ci_high: float,
    ate_dml_test: float,
    ate_dml_ci: tuple[float, float],
    ate_cf_test: float,
    ate_cf_ci: tuple[float, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "GWL_Baseline_HC3",
                "ate_test": ate_gwl_base_test,
                "ate_ci_low": ate_gwl_base_ci_low,
                "ate_ci_high": ate_gwl_base_ci_high,
            },
            {
                "model": "GWL_Extended_HC3",
                "ate_test": ate_gwl_ext_test,
                "ate_ci_low": ate_gwl_ext_ci_low,
                "ate_ci_high": ate_gwl_ext_ci_high,
            },
            {
                "model": "LinearDML",
                "ate_test": ate_dml_test,
                "ate_ci_low": ate_dml_ci[0],
                "ate_ci_high": ate_dml_ci[1],
            },
            {
                "model": "CausalForestDML",
                "ate_test": ate_cf_test,
                "ate_ci_low": ate_cf_ci[0],
                "ate_ci_high": ate_cf_ci[1],
            },
        ]
    )


# Write the main CSV and text outputs.
def save_pipeline_outputs(
    out_dir: Path,
    overlap_summary: pd.DataFrame,
    gate_table: pd.DataFrame,
    blp_table: pd.DataFrame,
    gates_test: pd.DataFrame,
    importance: pd.DataFrame,
    test_pred: pd.DataFrame,
    full_df: pd.DataFrame,
    main_results: pd.DataFrame,
    gwl_base_model: sm.regression.linear_model.RegressionResultsWrapper,
    gwl_ext_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap_summary.to_csv(out_dir / "overlap_summary.csv", index=False)
    main_results.to_csv(out_dir / "main_results.csv", index=False)
    save_propensity_overlap_figure(full_df, out_dir / "propensity_overlap_plot.png")


# Plot treated and control propensity-score densities without overlaid bars.
def save_propensity_overlap_figure(df: pd.DataFrame, out_path: Path) -> None:
    apply_plot_style()
    treated = df.loc[df["D"] == 1, "ehat"].to_numpy(dtype=float)
    control = df.loc[df["D"] == 0, "ehat"].to_numpy(dtype=float)
    bins = np.linspace(0, 1, 41)
    mids = 0.5 * (bins[1:] + bins[:-1])

    # Use line densities rather than overlapping bars to keep the figure closer to journal style.
    control_density, _ = np.histogram(control, bins=bins, density=True)
    treated_density, _ = np.histogram(treated, bins=bins, density=True)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.set_facecolor("none")
    ax.plot(mids, control_density, color="#8FA8BF", lw=2.0, linestyle="--", label="Rural")
    ax.plot(mids, treated_density, color="#1F4E79", lw=2.2, label="Urban")
    ax.scatter(mids[::2], control_density[::2], s=18, color="#8FA8BF", zorder=3)
    ax.scatter(mids[::2], treated_density[::2], s=18, color="#1F4E79", zorder=3)
    ax.set_xlabel("Estimated propensity score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Overlap")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color="#E6E6E0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# Write the report-ready LaTeX tables used outside the main paper block.
def save_main_result_figures(
    out_dir: Path,
    main_results: pd.DataFrame,
    blp_table: pd.DataFrame,
    n_obs: int,
    treated_share: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    het_row = blp_table.loc[blp_table["term"] == "heterogeneity_signal"].iloc[0]
    pretty = main_results.copy()
    pretty["Model"] = pretty["model"].map(
        {
            "GWL_Baseline_HC3": "GWL baseline (HC3)",
            "GWL_Extended_HC3": "GWL extended (HC3)",
            "LinearDML": "LinearDML",
            "CausalForestDML": "CausalForestDML",
        }
    )
    pretty["ATE"] = pretty["ate_test"].map(lambda x: f"{x:.4f}")
    pretty["95% CI"] = pretty.apply(lambda r: f"[{r['ate_ci_low']:.4f}, {r['ate_ci_high']:.4f}]", axis=1)
    table_df = pretty[["Model", "ATE", "95% CI"]]
    blp_text = pd.DataFrame(
        [
            {
                "Test": "BLP heterogeneity signal",
                "Coefficient": f"{het_row['coef']:.4f}",
                "p-value": f"{het_row['p_value']:.4f}",
            }
        ]
    )
    caption = f"Main estimation results. Sample used: {n_obs:,} observations; treated share = {treated_share:.3f}."
    main_latex = table_df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label="tab:main_results",
    )
    blp_latex = blp_text.to_latex(
        index=False,
        escape=False,
        caption="BLP heterogeneity test summary.",
        label="tab:blp_summary",
    )
    (out_dir / "main_results_table.tex").write_text(main_latex, encoding="utf-8")
    (out_dir / "blp_summary_table.tex").write_text(blp_latex, encoding="utf-8")

    plot_df = main_results.copy().reset_index(drop=True)
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['ate_test']:.4f}\n[{r['ate_ci_low']:.4f}, {r['ate_ci_high']:.4f}]",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    y = np.arange(len(plot_df))
    xerr = np.vstack(
        [
            plot_df["ate_test"] - plot_df["ate_ci_low"],
            plot_df["ate_ci_high"] - plot_df["ate_test"],
        ]
    )
    ax.errorbar(
        plot_df["ate_test"],
        y,
        xerr=xerr,
        fmt="o",
        color="#1f4e79",
        ecolor="#7aa6c2",
        elinewidth=2.2,
        capsize=4,
        markersize=7,
    )
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlabel("ATE estimate")
    ax.set_title("Average Treatment Effect Across Estimators")
    xmin = min(plot_df["ate_ci_low"].min(), -0.05)
    xmax = max(plot_df["ate_ci_high"].max(), 0.005)
    ax.set_xlim(xmin - 0.01, xmax + 0.03)
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["ate_ci_high"] + 0.002,
            yi,
            row["label"],
            va="center",
            ha="left",
            fontsize=9.5,
            color="#2b2b2b",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f7f7f7", edgecolor="#dddddd"),
        )
    fig.tight_layout()
    fig.savefig(out_dir / "ate_comparison_figure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
