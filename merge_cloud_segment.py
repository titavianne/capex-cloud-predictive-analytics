"""One-off merge of the per-ticker cloud segment research CSVs (produced by
three separate research passes) into the single data/cloud_segment_manual.csv
used by build_panel.py. Run once after all three per-ticker files exist."""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PARTS = ["cloud_segment_MSFT.csv", "cloud_segment_GOOGL.csv", "cloud_segment_AMZN.csv"]

COLUMNS = [
    "ticker", "segment_name", "fiscal_year_label", "fiscal_quarter_label", "period_end",
    "cloud_revenue_usd", "segment_definition_note", "source_url", "release_date",
    "filled_by", "verified",
]


def main():
    frames = []
    for p in PARTS:
        path = DATA_DIR / p
        if not path.exists():
            print(f"WARNING: {p} not found, skipping")
            continue
        df = pd.read_csv(path)
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise SystemExit(f"{p} is missing columns: {missing}")
        frames.append(df[COLUMNS])

    if not frames:
        raise SystemExit("No part files found.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["ticker", "period_end"])
    dupes = merged.duplicated(subset=["ticker", "period_end"], keep=False)
    if dupes.any():
        print("WARNING: duplicate ticker/period_end rows found:")
        print(merged[dupes])

    out_path = DATA_DIR / "cloud_segment_manual.csv"
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(merged)} rows)")
    print(merged.groupby("ticker").agg(n=("period_end", "size"), first=("period_end", "min"), last=("period_end", "max")))


if __name__ == "__main__":
    main()
