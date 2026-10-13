"""
Pull the effective federal funds rate from FRED and build a quarterly series.

Uses the public `fredgraph.csv` endpoint, which does NOT require a FRED API key
(unlike the JSON /fred/series/observations endpoint). This keeps the pipeline
runnable without asking for another credential.

Series: FEDFUNDS (Effective Federal Funds Rate, monthly, percent)
https://fred.stlouisfed.org/series/FEDFUNDS

Output: data/processed/fed_funds_quarterly.csv
  columns: quarter_end (date), fed_funds_avg (quarterly avg of monthly rate, %),
           fed_funds_chg_yoy (percentage-point change vs same quarter last year)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"


def main():
    print(f"GET {FRED_CSV_URL}")
    resp = requests.get(FRED_CSV_URL, timeout=30)
    resp.raise_for_status()
    raw_path = RAW_DIR / "FEDFUNDS.csv"
    raw_path.write_bytes(resp.content)

    df = pd.read_csv(raw_path)
    # fredgraph.csv columns are typically: observation_date, FEDFUNDS
    date_col = df.columns[0]
    val_col = df.columns[1]
    df = df.rename(columns={date_col: "date", val_col: "fed_funds"})
    df["date"] = pd.to_datetime(df["date"])
    df["fed_funds"] = pd.to_numeric(df["fed_funds"], errors="coerce")
    df = df.dropna(subset=["fed_funds"]).sort_values("date")

    df["quarter"] = df["date"].dt.to_period("Q")
    q = df.groupby("quarter")["fed_funds"].mean().reset_index()
    q["quarter_end"] = q["quarter"].dt.end_time.dt.normalize()
    q = q[["quarter_end", "fed_funds"]].rename(columns={"fed_funds": "fed_funds_avg"})
    q = q.sort_values("quarter_end").reset_index(drop=True)

    q["fed_funds_chg_yoy"] = q["fed_funds_avg"] - q["fed_funds_avg"].shift(4)

    out_path = PROC_DIR / "fed_funds_quarterly.csv"
    q.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(q)} quarters, {q['quarter_end'].min().date()} to {q['quarter_end'].max().date()})")


if __name__ == "__main__":
    main()
