#!/usr/bin/env python3
"""Keep frozen and fetched copies of BudgetFood."""

from __future__ import annotations

from pathlib import Path
import sys
import hashlib
from io import StringIO

import pandas as pd


def _try_direct_csv_urls() -> pd.DataFrame | None:
    urls = [
        "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/BudgetFood.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if "rownames" in df.columns:
                df = df.drop(columns=["rownames"])
            if not df.empty:
                print(f"[OK] Loaded BudgetFood from URL: {url}")
                return df
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] direct URL fetch failed for {url}: {exc}")
    return None


def _try_statsmodels_fetch() -> pd.DataFrame | None:
    try:
        from statsmodels.datasets import get_rdataset
    except Exception:
        return None

    for pkg in ("Ecdat",):
        try:
            ds = get_rdataset("BudgetFood", package=pkg)
            df = ds.data.copy()
            if not df.empty:
                print(f"[OK] Loaded BudgetFood via statsmodels from package={pkg}")
                return df
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] statsmodels fetch failed for package={pkg}: {exc}")
    return None


def _try_rpy2_fetch() -> pd.DataFrame | None:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr, isinstalled
        from rpy2.robjects.vectors import StrVector
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] rpy2 unavailable: {exc}")
        return None

    ro.r('options(repos = c(CRAN = "https://cloud.r-project.org"))')
    ro.r("options(ask = FALSE)")
    utils = importr("utils")
    user_lib = (Path(__file__).resolve().parents[1] / ".r_libs").as_posix()
    Path(user_lib).mkdir(parents=True, exist_ok=True)
    ro.r(f'Sys.setenv(R_LIBS_USER="{user_lib}")')

    pkg = "Ecdat"
    try:
        if not isinstalled(pkg):
            utils.install_packages(StrVector([pkg]), lib=user_lib, quiet=True)
        importr(pkg)
        ro.r(f'data("BudgetFood", package="{pkg}")')
        r_df = ro.r["BudgetFood"]
        with localconverter(default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(r_df)
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"[OK] Loaded BudgetFood via rpy2 from package={pkg}")
            return df
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] rpy2 fetch failed for package={pkg}: {exc}")

    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for clean comparisons."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _frame_hash(df: pd.DataFrame) -> str:
    """Stable content hash for quick version checks."""
    normalized = pd.read_csv(StringIO(df.reset_index(drop=True).to_csv(index=False)))
    payload = normalized.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _comparison_table(direct_df: pd.DataFrame, fetched_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize differences between frozen and fetched copies."""
    direct_cols = list(direct_df.columns)
    fetched_cols = list(fetched_df.columns)
    same_columns = direct_cols == fetched_cols
    rows_equal = len(direct_df) == len(fetched_df)
    # Ignore harmless dtype differences after normalization.
    direct_cmp = pd.read_csv(StringIO(direct_df.reset_index(drop=True).to_csv(index=False)))
    fetched_cmp = pd.read_csv(StringIO(fetched_df.reset_index(drop=True).to_csv(index=False)))
    if same_columns:
        fetched_cmp = fetched_cmp[direct_cols]
    values_equal = same_columns and direct_cmp.equals(fetched_cmp)
    return pd.DataFrame(
        [
            {"metric": "direct_rows", "value": len(direct_df)},
            {"metric": "fetched_rows", "value": len(fetched_df)},
            {"metric": "same_row_count", "value": rows_equal},
            {"metric": "direct_col_count", "value": len(direct_cols)},
            {"metric": "fetched_col_count", "value": len(fetched_cols)},
            {"metric": "same_columns", "value": same_columns},
            {"metric": "exact_equal_after_standardization", "value": values_equal},
            {"metric": "direct_hash", "value": _frame_hash(direct_df)},
            {"metric": "fetched_hash", "value": _frame_hash(fetched_df)},
            {
                "metric": "direct_only_columns",
                "value": ", ".join(sorted(set(direct_cols) - set(fetched_cols))) or "(none)",
            },
            {
                "metric": "fetched_only_columns",
                "value": ", ".join(sorted(set(fetched_cols) - set(direct_cols))) or "(none)",
            },
        ]
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    active_csv = out_dir / "budgetfood.csv"
    direct_csv = out_dir / "budgetfood_direct.csv"
    fetched_csv = out_dir / "budgetfood_fetched.csv"
    comparison_csv = out_dir / "budgetfood_comparison.csv"

    # Freeze the current local copy the first time this runs.
    if not direct_csv.exists() and active_csv.exists():
        legacy_df = pd.read_csv(active_csv)
        legacy_df = _standardize_columns(legacy_df)
        legacy_df.to_csv(direct_csv, index=False)
        print(f"[OK] Created frozen direct copy from existing active file: {direct_csv}")

    df = _try_direct_csv_urls()
    if df is None:
        df = _try_statsmodels_fetch()
    if df is None:
        df = _try_rpy2_fetch()

    if df is None or df.empty:
        print("[ERROR] Could not fetch BudgetFood data from known sources.", file=sys.stderr)
        print("Try checking internet/R package availability and rerun.", file=sys.stderr)
        return 1

    # Keep canonical lowercase columns.
    df = _standardize_columns(df)
    df.to_csv(fetched_csv, index=False)
    # Keep the active analysis file aligned with the fetched copy.
    df.to_csv(active_csv, index=False)

    if not direct_csv.exists():
        df.to_csv(direct_csv, index=False)
        print(f"[OK] No frozen direct copy existed; initialized it from current fetch: {direct_csv}")

    direct_df = _standardize_columns(pd.read_csv(direct_csv))
    comparison = _comparison_table(direct_df, df)
    comparison.to_csv(comparison_csv, index=False)

    print(f"[OK] Saved fetched copy to: {fetched_csv}")
    print(f"[OK] Updated active analysis copy to: {active_csv}")
    print(f"[OK] Frozen direct copy kept at: {direct_csv}")
    print(f"[OK] Saved dataset comparison report to: {comparison_csv}")
    print(f"[OK] Columns: {', '.join(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
