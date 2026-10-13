"""
Merge EDGAR quarterly fundamentals + manual cloud segment revenue + FRED fed
funds rate into one quarterly panel, and engineer the (lagged, leakage-free)
features used by model.py.

Inputs:
  data/processed/edgar_quarterly.csv    (from pull_edgar.py)
  data/cloud_segment_manual.csv         (manually sourced, see CLOUD_SEGMENT_README.md)
  data/processed/fed_funds_quarterly.csv(from pull_fred.py)

Output:
  data/panel.csv

Design notes (do not change without asking -- see project brief):
  - Target = cloud_yoy: YoY growth of the manually-sourced cloud SEGMENT revenue
    (Intelligent Cloud / Google Cloud / AWS), computed on each ticker's own
    period_end sequence (handles MSFT's non-calendar fiscal quarters fine since
    we always compare period_end to period_end - 1 year, not position - 4).
  - capex here = capex_cash + finance_lease_additions (falls back to capex_cash
    alone when finance-lease data isn't available that quarter -- flagged).
  - Every feature used to predict cloud_yoy at quarter t must be knowable by the
    end of quarter t-1 or t's prediction time: capex_yoy etc. are given as
    lag1..lag4 (i.e. already shifted), fed_funds_chg_yoy is also lagged by one
    quarter (shift(1)) since the model should not see the *current* quarter's
    rate when predicting the current quarter's cloud growth, even though the
    rate itself is technically public mid-quarter -- treat consistently as
    lagged macro info.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROC_DIR = BASE_DIR / "data" / "processed"
DATA_DIR = BASE_DIR / "data"

CLOUD_TARGET_TICKERS = {"MSFT", "GOOGL", "AMZN"}  # META = capex/FCF only, not a target


def load_edgar() -> pd.DataFrame:
    path = PROC_DIR / "edgar_quarterly.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path} -- run pull_edgar.py first.")
    df = pd.read_csv(path, parse_dates=["period_end"])
    df["capex_cash"] = df.get("capex_cash", np.nan)
    df["finance_lease_additions"] = df.get("finance_lease_additions", np.nan)
    df["capex_total"] = df["capex_cash"].fillna(0) + df["finance_lease_additions"].fillna(0)
    df["capex_total_has_lease"] = df["finance_lease_additions"].notna()
    # if capex_cash itself is missing, capex_total should be NaN not just the lease part
    df.loc[df["capex_cash"].isna(), "capex_total"] = np.nan
    df["fcf"] = df["operating_cash_flow"] - df["capex_total"]
    df["fcf_margin"] = df["fcf"] / df["revenue"]
    df["capex_to_rev"] = df["capex_total"] / df["revenue"]
    return df


def load_cloud_segment() -> pd.DataFrame:
    path = DATA_DIR / "cloud_segment_manual.csv"
    df = pd.read_csv(path, parse_dates=["period_end"])
    df = df.dropna(subset=["cloud_revenue_usd"])
    if df.empty:
        print("WARNING: cloud_segment_manual.csv has no filled rows yet.")
    return df


def load_fed_funds() -> pd.DataFrame:
    path = PROC_DIR / "fed_funds_quarterly.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path} -- run pull_fred.py first.")
    return pd.read_csv(path, parse_dates=["quarter_end"])


def yoy_by_ticker(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    """YoY growth matched by (ticker, period_end - 1 year) rather than a fixed
    4-row shift, so it's robust to any missing quarter in between."""
    df = df.sort_values(["ticker", "period_end"]).copy()
    lut = df.set_index(["ticker", "period_end"])[value_col]

    def _yoy(row):
        key = (row["ticker"], row["period_end"] - pd.DateOffset(years=1))
        prior = lut.get(key, np.nan)
        if pd.isna(prior) or prior == 0:
            return np.nan
        return (row[value_col] - prior) / abs(prior)

    df[out_col] = df.apply(_yoy, axis=1)
    return df


def add_lags(df: pd.DataFrame, col: str, max_lag: int = 4) -> pd.DataFrame:
    df = df.sort_values(["ticker", "period_end"]).copy()
    for lag in range(1, max_lag + 1):
        df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    return df


def main():
    edgar = load_edgar()
    cloud = load_cloud_segment()
    fed = load_fed_funds()

    edgar = yoy_by_ticker(edgar, "capex_total", "capex_yoy")
    edgar = yoy_by_ticker(edgar, "long_term_debt", "debt_yoy")
    edgar = add_lags(edgar, "capex_yoy", max_lag=4)
    edgar = add_lags(edgar, "debt_yoy", max_lag=1)  # only lag1 needed per feature list
    # capex_to_rev / fcf_margin are contemporaneous ratios (same-quarter capex & revenue,
    # which are only known once that quarter's results are released -- the same moment
    # cloud_yoy for that quarter becomes known). Using them un-lagged as predictors of
    # that same quarter's cloud_yoy would leak target-quarter information, so lag them.
    edgar = add_lags(edgar, "capex_to_rev", max_lag=1)
    edgar = add_lags(edgar, "fcf_margin", max_lag=1)

    cloud = cloud.rename(columns={"cloud_revenue_usd": "cloud_revenue"})
    cloud_agg = cloud.groupby(["ticker", "period_end"], as_index=False)["cloud_revenue"].first()
    cloud_agg = yoy_by_ticker(cloud_agg, "cloud_revenue", "cloud_yoy")
    cloud_agg = add_lags(cloud_agg, "cloud_yoy", max_lag=1)
    cloud_agg = cloud_agg.rename(columns={"cloud_yoy_lag1": "cloud_yoy_lag1_feat"})

    fed["fed_funds_chg_yoy_lag"] = fed["fed_funds_chg_yoy"].shift(1)
    fed_small = fed[["quarter_end", "fed_funds_chg_yoy_lag"]].rename(columns={"quarter_end": "period_end"})

    panel = cloud_agg.merge(
        edgar[
            [
                "ticker", "period_end", "revenue", "capex_cash", "finance_lease_additions",
                "capex_total", "capex_total_has_lease", "operating_cash_flow", "long_term_debt",
                "fcf", "fcf_margin", "capex_to_rev",
                "capex_yoy", "capex_yoy_lag1", "capex_yoy_lag2", "capex_yoy_lag3", "capex_yoy_lag4",
                "debt_yoy_lag1", "capex_to_rev_lag1", "fcf_margin_lag1",
            ]
        ],
        on=["ticker", "period_end"],
        how="left",
    )
    panel = panel.merge(fed_small, on="period_end", how="left")

    panel = panel[panel["ticker"].isin(CLOUD_TARGET_TICKERS)].sort_values(["ticker", "period_end"])

    feature_cols = [
        "capex_yoy_lag1", "capex_yoy_lag2", "capex_yoy_lag3", "capex_yoy_lag4",
        "fed_funds_chg_yoy_lag", "capex_to_rev_lag1", "fcf_margin_lag1", "debt_yoy_lag1",
        "cloud_yoy_lag1_feat",
    ]
    n_complete = panel[["cloud_yoy"] + feature_cols].dropna().groupby(panel["ticker"]).size() if not panel.empty else {}

    out_path = DATA_DIR / "panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(panel)} rows)")
    print("\nComplete-case rows (target + all features non-null) per ticker:")
    print(n_complete)
    for t in CLOUD_TARGET_TICKERS:
        n = int(n_complete.get(t, 0)) if hasattr(n_complete, "get") else 0
        if n < 8:
            print(f"  STOP/REVIEW: {t} has only {n} complete-case quarters (<8). "
                  f"Walk-forward validation + pre/post-2023 split will not be reliable.")


if __name__ == "__main__":
    main()
