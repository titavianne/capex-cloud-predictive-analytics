"""
Pull quarterly fundamentals (revenue, capex, finance-lease additions, operating
cash flow, long-term debt) for hyperscaler parents from SEC EDGAR companyfacts.

Cloud SEGMENT revenue is NOT available here -- SEC companyfacts only exposes
consolidated (company-wide) XBRL facts. Segment data must be filled manually
in data/cloud_segment_manual.csv (see fill_cloud_segment.py / README).

Requires env var SEC_USER_AGENT = "Your Name your.email@domain.com"
(SEC fair-access policy: https://www.sec.gov/os/accessing-edgar-data)

Output:
  data/raw/companyfacts_<TICKER>.json   (cached raw API response)
  data/raw/edgar_facts_long.csv         (every raw duration/instant fact pulled,
                                          for audit)
  data/processed/edgar_quarterly.csv    (de-cumulated, tag-stitched quarterly panel)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()
if not SEC_USER_AGENT or "@" not in SEC_USER_AGENT:
    sys.exit(
        "ERROR: set SEC_USER_AGENT env var first, e.g.\n"
        '  export SEC_USER_AGENT="Your Name your.email@domain.com"   (bash)\n'
        '  $env:SEC_USER_AGENT = "Your Name your.email@domain.com"   (PowerShell)'
    )

HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
SLEEP_BETWEEN_REQUESTS = 0.3  # be polite to SEC (<=10 req/s allowed, we go much slower)

# CIKs (zero-padded to 10 digits), confirmed against SEC EDGAR full text search.
TICKERS = {
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",  # Alphabet Inc. (parent of Google)
    "META": "0001326801",
}

# Candidate XBRL us-gaap tags per concept, in priority order (first available wins
# for a given quarter; the tag actually used is recorded for audit).
TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "capex_cash": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        # Amazon-specific tag used in some filings for the same cash-capex line;
        # flagged by the user as a risk point -- do NOT silently sum this with
        # PaymentsToAcquirePropertyPlantAndEquipment, treat as alternate tag for
        # the SAME line item (priority fallback only).
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
    ],
    # Equipment/data-center capacity obtained under finance leases (non-cash additions).
    # Needed because cash capex alone understates real infrastructure build-out,
    # especially for AMZN/MSFT/META who lease substantial data-center capacity.
    "finance_lease_additions": [
        "RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability",
        "FinanceLeaseRightOfUseAssetObtainedInExchangeForLeaseLiability",
        "FinanceLeaseObligationsIncurred",
        "CapitalLeaseObligationsIncurred",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
        "DebtLongtermAndShorttermCombinedAmount",
    ],
}

DURATION_CONCEPTS = {"revenue", "capex_cash", "finance_lease_additions", "operating_cash_flow"}
INSTANT_CONCEPTS = {"long_term_debt"}


def fetch_companyfacts(ticker: str, cik: str) -> dict:
    cache_path = RAW_DIR / f"companyfacts_{ticker}.json"
    if cache_path.exists():
        print(f"[{ticker}] using cached {cache_path.name}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    print(f"[{ticker}] GET {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return data


def extract_tag_facts(facts: dict, tag: str) -> pd.DataFrame:
    """Pull every USD fact for one us-gaap tag into a flat frame."""
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if node is None:
        return pd.DataFrame()
    units = node.get("units", {})
    usd = units.get("USD", [])
    if not usd:
        return pd.DataFrame()
    df = pd.DataFrame(usd)
    df["tag"] = tag
    keep = [c for c in ["start", "end", "val", "fy", "fp", "form", "filed", "accn", "frame", "tag"] if c in df.columns]
    df = df[keep].copy()
    # Only 10-Q / 10-K sourced facts (skip 8-K/other ad hoc disclosures, which are
    # inconsistent in period framing).
    df = df[df["form"].isin(["10-Q", "10-Q/A", "10-K", "10-K/A"])]
    return df


def dedupe_prefer_latest_filed(df: pd.DataFrame) -> pd.DataFrame:
    """For facts covering the exact same (start,end), a later filing may restate
    the value. Keep the most-recently-filed value; record how many distinct
    values existed (restatement count) for audit."""
    if df.empty:
        return df
    df = df.sort_values("filed")
    g = df.groupby(["start", "end"], as_index=False)
    out = g.agg(
        val=("val", "last"),
        n_versions=("val", "nunique"),
        filed=("filed", "last"),
        fy=("fy", "last"),
        fp=("fp", "last"),
        form=("form", "last"),
        accn=("accn", "last"),
        tag=("tag", "last"),
    )
    return out


def decumulate(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a set of (start,end,val) duration facts into pure-quarter values.

    Group facts that share the same `start` date (= same fiscal-year anchor).
    Within a group sorted by `end` date ascending:
      - 1st entry (shortest duration, ~1 quarter) IS the Q1 quarterly value.
      - each later entry's quarterly value = val - previous cumulative val
        (so the 4th entry, the FY/10-K annual figure, yields Q4 = FY - 9M).
    Flags a gap when consecutive `end` dates are >110 days apart (missing filing
    or non-standard period) instead of guessing.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["period_end", "period_start", "quarter_val", "cum_val", "fy", "fp", "form",
                     "filed", "accn", "tag", "n_versions", "gap_flag", "seq_in_group"]
        )
    df = df.copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    rows = []
    for start, g in df.groupby("start"):
        g = g.sort_values("end").reset_index(drop=True)
        # Drop noisy single-member "chains" that are not a genuine standalone
        # quarter (e.g. ad hoc trailing-twelve-month comparatives some filers
        # tag with an odd start date). A group is only trusted if it either
        # (a) has >=2 members, i.e. a real YTD cumulative chain anchored at a
        # recurring fiscal-year start, or (b) is a single member whose own
        # duration is already ~1 quarter (75-105 days).
        if len(g) == 1 and not (75 <= (g.loc[0, "end"] - start).days <= 105):
            continue
        prev_end = None
        prev_val = 0.0
        for i, r in g.iterrows():
            dur_days = (r["end"] - start).days
            gap_flag = False
            if prev_end is not None:
                gap_days = (r["end"] - prev_end).days
                if gap_days > 110:
                    gap_flag = True
            quarter_val = r["val"] - prev_val
            rows.append(
                {
                    "period_end": r["end"],
                    "period_start": prev_end if prev_end is not None else start,
                    "quarter_val": quarter_val,
                    "cum_val": r["val"],
                    "fy": r.get("fy"),
                    "fp": r.get("fp"),
                    "form": r.get("form"),
                    "filed": r.get("filed"),
                    "accn": r.get("accn"),
                    "tag": r.get("tag"),
                    "n_versions": r.get("n_versions", 1),
                    "gap_flag": gap_flag,
                    "seq_in_group": i + 1,
                    "duration_days": dur_days,
                    "chain_len": len(g),
                }
            )
            prev_end = r["end"]
            prev_val = r["val"]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Resolve conflicts: the same period_end can appear from more than one
    # start-group (e.g. a discrete 3-month fact AND a YTD-chain-derived value
    # for the same quarter). Prefer the longer chain (more corroborated by
    # surrounding quarters), tie-broken by duration closest to 91 days.
    out["_dist91"] = (out["duration_days"] - 91).abs()
    out = out.sort_values(["period_end", "chain_len", "_dist91"], ascending=[True, False, True])
    dup_check = out.groupby("period_end")["quarter_val"]
    conflict = dup_check.transform(lambda s: (s.max() - s.min()) > 0.01 * s.abs().max() if len(s) > 1 else False)
    if conflict.any():
        n = int(out.loc[conflict, "period_end"].nunique())
        print(f"    WARNING: {n} period(s) had disagreeing quarterly values across chains (kept the better-corroborated one) -- see audit CSV")
    out = out.drop_duplicates(subset=["period_end"], keep="first").drop(columns=["_dist91"])
    return out.sort_values("period_end").reset_index(drop=True)


def build_concept_series(facts: dict, concept: str) -> pd.DataFrame:
    """Try each candidate tag in priority order, de-cumulate each, then stitch
    into one quarterly series (first tag with coverage for a given quarter wins).
    Returns long frame: period_end, value, tag_used, concept, + audit columns.
    """
    is_instant = concept in INSTANT_CONCEPTS
    all_tagged = []
    for tag in TAGS[concept]:
        raw = extract_tag_facts(facts, tag)
        if raw.empty:
            continue
        if is_instant:
            raw = raw.copy()
            raw["end"] = pd.to_datetime(raw["end"])
            raw = raw.sort_values("filed")
            dd = raw.groupby("end", as_index=False).agg(
                val=("val", "last"),
                n_versions=("val", "nunique"),
                filed=("filed", "last"),
                fy=("fy", "last"),
                fp=("fp", "last"),
                form=("form", "last"),
                accn=("accn", "last"),
            )
            dd["tag"] = tag
            dd["quarter_val"] = dd["val"]
            dd["period_end"] = dd["end"]
            dd["gap_flag"] = False
            dd["duration_days"] = 0
        else:
            dd_raw = dedupe_prefer_latest_filed(raw)
            dd = decumulate(dd_raw)
        if dd.empty:
            continue
        dd["tag_priority"] = TAGS[concept].index(tag)
        all_tagged.append(dd)

    if not all_tagged:
        return pd.DataFrame()

    combined = pd.concat(all_tagged, ignore_index=True)
    combined["period_end"] = pd.to_datetime(combined["period_end"])
    # Note: decumulate() already filtered out noisy chains before diffing, so
    # `duration_days` here is just the raw fact's own span (91/183/273/365 for
    # a Q1/H1/9M/FY-derived quarter_val) -- it is NOT a quality signal at this
    # point and must not be used to drop rows.
    combined = combined.sort_values(["period_end", "tag_priority"])
    stitched = combined.drop_duplicates(subset=["period_end"], keep="first").copy()
    stitched["concept"] = concept
    return stitched


def main():
    long_rows = []
    quarterly_frames = []
    for ticker, cik in TICKERS.items():
        facts = fetch_companyfacts(ticker, cik)
        for concept in TAGS:
            s = build_concept_series(facts, concept)
            if s.empty:
                print(f"  [{ticker}] WARNING: no data found for concept '{concept}'")
                continue
            s["ticker"] = ticker
            quarterly_frames.append(s[["ticker", "concept", "period_end", "quarter_val", "tag", "fy", "fp",
                                        "form", "filed", "accn", "n_versions", "gap_flag"]])
            n_gaps = int(s["gap_flag"].sum())
            print(f"  [{ticker}] {concept}: {len(s)} quarters, tags used={sorted(s['tag'].unique())}, gaps={n_gaps}")

    if not quarterly_frames:
        sys.exit("No data extracted at all -- check SEC_USER_AGENT / network access.")

    long_df = pd.concat(quarterly_frames, ignore_index=True)
    long_df.to_csv(RAW_DIR / "edgar_facts_long.csv", index=False)

    wide = long_df.pivot_table(
        index=["ticker", "period_end"], columns="concept", values="quarter_val", aggfunc="first"
    ).reset_index()

    # tag/audit columns alongside, per concept (so a human can see which tag + filing
    # backs every number without re-opening edgar_facts_long.csv)
    for concept in TAGS:
        sub = long_df[long_df["concept"] == concept][["ticker", "period_end", "tag", "form", "filed", "accn", "n_versions", "gap_flag"]]
        sub = sub.rename(columns={c: f"{concept}_{c}" for c in ["tag", "form", "filed", "accn", "n_versions", "gap_flag"]})
        wide = wide.merge(sub, on=["ticker", "period_end"], how="left")

    wide = wide.sort_values(["ticker", "period_end"])
    wide.to_csv(PROC_DIR / "edgar_quarterly.csv", index=False)
    print(f"\nWrote {PROC_DIR / 'edgar_quarterly.csv'} ({len(wide)} rows)")
    print(f"Wrote {RAW_DIR / 'edgar_facts_long.csv'} ({len(long_df)} rows, audit trail)")


if __name__ == "__main__":
    main()
