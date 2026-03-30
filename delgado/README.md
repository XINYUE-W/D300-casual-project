# D300 Project - Delgado (BudgetFood)

This repo studies whether urban residence affects the household food expenditure share and whether that effect is heterogeneous across households. The project uses the `BudgetFood` data background from Delgado and Mora (1998), but the research question here is different: the original paper studies Engel-curve specification, while this repo studies an observational treatment-effect problem.

## Research design

- Outcome `Y`: `wfood`
- Treatment `D`: `urban = 1(town >= 4)`
- Covariates `X`: `log_totexp`, `age`, `size`, `sex_male`

The empirical strategy combines:

- a GWL-style parametric baseline
- an extended parametric specification with treatment interactions
- `LinearDML`
- `CausalForestDML`
- repeated-split heterogeneity analysis (`BLP`, sorted `GATE`, `CLAN`)

## Folder structure

- `data_management.py`
  - creates and compares frozen and fetched copies of the data
- `MODELS/00_descriptive_diagnostics.py`
  - descriptive tables and overlap diagnostics
- `MODELS/01_engel_curve_context.py`
  - Delgado / Engel-curve background plots
- `MODELS/02_main_model_building.py`
  - main ATE estimation pipeline
- `MODELS/03_heterogeneity_analysis.py`
  - repeated-split heterogeneity outputs (default: `100` sample splits)
- `MODELS/main_model_helpers.py`
  - helper functions for the main estimation pipeline
- `MODELS/heterogeneity_helpers.py`
  - helper functions for repeated-split heterogeneity analysis
- `notebooks/00_project_eda.ipynb`
  - project EDA and diagnostics
- `notebooks/01_main_model_results.ipynb`
  - main estimation results and LaTeX tables
- `notebooks/02_heterogeneity_review.ipynb`
  - repeated-split heterogeneity results
- `notebooks/03_delgado_context.ipynb`
  - Delgado context notebook
- `latex/`
  - paper-ready result blocks and tables

## Data workflow

Run:

```bash
python3 data_management.py
```

This maintains four files in `data/`:

- `budgetfood_direct.csv`
  - frozen local copy
- `budgetfood_fetched.csv`
  - re-fetched copy
- `budgetfood.csv`
  - active analysis copy used by the scripts
- `budgetfood_comparison.csv`
  - comparison between the frozen and fetched versions

## Reproducible run

1. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate d300-delgado
```

2. Prepare data:

```bash
python3 data_management.py
```

3. Run the analysis:

```bash
python3 MODELS/00_descriptive_diagnostics.py
python3 MODELS/01_engel_curve_context.py
python3 MODELS/02_main_model_building.py
python3 MODELS/03_heterogeneity_analysis.py
```

## Outputs

- `outputs/eda_output/`
  - `eda_diagnostics.png`
- `outputs/context_results/`
  - `engel_curve_gwl_vs_lowess.png`
- `outputs/main_model_results/`
  - `main_results.csv`, `overlap_summary.csv`, `propensity_overlap_plot.png`
- `outputs/heterogeneity_results/`
  - `repeated_split_blp_summary.csv`
  - `repeated_split_sorted_gate_summary.csv`
  - `repeated_split_sorted_gate_plot.png`
  - `repeated_split_clan_summary.csv`
  - `repeated_split_clan_table.csv`
  - `repeated_split_settings.csv`
  - `permutation_importance_summary.csv`
  - `permutation_importance_plot.png`

## LaTeX results

The `latex/` folder contains paper-ready blocks.

If your main `.tex` file is in the repo root, add:

```latex
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{caption}
\usepackage{adjustbox}
\usepackage{graphicx}
```

Then insert:

```latex
\input{latex/results_blocks.tex}
```

If your document numbers tables within sections but you want global numbering (for example, `Table 3`, `Table 4` rather than `Table 4.1`), also add:

```latex
\usepackage{chngcntr}
\counterwithout{table}{section}
```

## Notes

- This is an observational treatment-effects design.
- Causal interpretation requires selection on observables and overlap.
- The main heterogeneity results are the repeated-split results, not the earlier single-split exploratory outputs.
- The repeated-split heterogeneity summaries are reported using the `100`-split default setting; the BLP, sorted GATE, and CLAN summaries now use `95%` confidence intervals.
