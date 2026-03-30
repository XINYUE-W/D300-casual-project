Use these files from your main LaTeX document with:

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

If you want finer control, insert the individual files instead:

- `latex/tbl_results_34.tex`
- `latex/tbl_main_results.tex`
- `latex/tbl_blp_results.tex`
- `latex/tbl_clan_results.tex`
- `latex/tbl_permutation_results.tex`

`latex/tbl_results_34.tex` is the preferred combined block for the side-by-side Table 3 / Table 4 layout.

The figure files used by `latex/results_blocks.tex` are stored directly in this folder:

- `latex/repeated_split_sorted_gate_plot.png`
- `latex/permutation_importance_plot.png`
- `latex/eda_diagnostics.png`
- `latex/propensity_overlap_plot.png`
