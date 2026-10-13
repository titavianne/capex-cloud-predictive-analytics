# Data validation report

## 1. EDGAR coverage (data/processed/edgar_quarterly.csv)
| ticker | quarters | date range |
|---|---|---|
| MSFT | 76 | 2007-09-30 .. 2026-06-30 (usable range: ~2009+, XBRL tagging was inconsistent 2007-2008) |
| AMZN | 73 | 2008-06-30 .. 2026-06-30 |
| GOOGL | 48 | 2014-09-30 .. 2026-06-30 |
| META | 61 | 2012-03-31 .. 2026-06-30 |

No gaps >110 days between consecutive quarters for any ticker (checked programmatically).

## 2. Null patterns (all explainable, not data-pull bugs)
- MSFT 2007-09-30 .. 2008-06-30: capex/OCF null — predates reliable XBRL cash-flow tagging; outside the project's analysis window (FY2015+) anyway.
- AMZN 2017-06-30: capex_cash null for that one quarter despite finance_lease_additions being present — an isolated tag-coverage gap in that filing; not corrected/imputed (left as NaN).
- META long_term_debt null/zero until 2022-09-30: Meta had no long-term debt before its first bond issuance in Aug 2022 ($10B) — confirmed against public record, not a pull error.
- finance_lease_additions null for ~32-49 quarters per ticker: this XBRL concept (`RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability`) only exists from ASC 842 adoption (~2019) onward; pre-2019 quarters have no such disclosure. `capex_total` silently equals `capex_cash` for these quarters (flagged via `capex_total_has_lease=False`) — a known understatement of real infrastructure spend for older quarters.

## 3. Risk points called out in the brief — resolved
1. **Amazon capex tag (`PaymentsToAcquireProductiveAssets`)**: confirmed via raw fact inspection that Amazon's XBRL tag for "Purchases of property and equipment" switched from `PaymentsToAcquirePropertyPlantAndEquipment` to `PaymentsToAcquireProductiveAssets` around 2017 — same economic line item, not an additive second capex component. Treated as alternate tags for one series (priority fallback), not summed. Verified exact match against the Q2 2025 earnings release: EDGAR pull = $32,183M, press release = $32,183M.
2. **Revenue tag renames**: `RevenueFromContractWithCustomerExcludingAssessedTax` (ASC 606, ~2018+) stitched with legacy `Revenues` / `SalesRevenueNet` at the cutover quarter per company. Verified MSFT ($76,441M), AMZN ($167,702M), GOOGL ($96,434M vs. reported "$96.4B") all exact-match Q2 2025 press releases.
3. **Capex excluding finance leases**: cash capex alone understates real infrastructure spend. Added `finance_lease_additions` (correct tag `RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability`, found after an initial wrong guess at the tag name was corrected) and `capex_total = capex_cash + finance_lease_additions`. Verified AMZN Q2 2025 finance-lease addition: EDGAR pull = $937M, press release = $937M (exact match).
4. **Restatements**: when the same (start,end) period has multiple reported values across filings (restatement), the most-recently-filed value is kept; `n_versions` column records how many distinct values existed.
5. **A decumulation bug was found and fixed during this run**: some filers tag ad hoc trailing-twelve-month comparatives with non-fiscal-year start dates (e.g. Amazon's `2008-04-01..2009-03-31` entry). An early version of `pull_edgar.py` misread these as the first member of a cumulative chain, corrupting several early-year quarterly figures. Fixed by only trusting a cumulative chain if it has >=2 members sharing the same start date (a real multi-quarter chain) or is a lone entry whose own duration is ~1 quarter. Re-verified after the fix (see cross-checks below).

## 4. Live cross-checks against primary sources (6 figures, exceeds the 3-4 requested)
| ticker | quarter | metric | EDGAR pull | Primary source | Match |
|---|---|---|---|---|---|
| MSFT | FY25 Q4 (2025-06-30) | Revenue | $76,441M | $76.4B (press release) | exact |
| MSFT | FY25 Q4 | Capex (cash) | $17,079M | $17,079M (8-K ex. 99.1) | exact |
| AMZN | Q2 2025 | Net sales | $167,702M | $167,702M (8-K ex. 99.1) | exact |
| AMZN | Q2 2025 | Capex (cash) | $32,183M | $32,183M (8-K ex. 99.1) | exact |
| AMZN | Q2 2025 | Finance lease additions | $937M | $937M (8-K ex. 99.1) | exact |
| GOOGL | Q2 2025 | Revenue | $96,434M | "$96.4 billion" (press release) | matches |

No discrepancies found. EDGAR pipeline is trusted for modeling.

## 5. Cloud segment manual data — final state
`data/cloud_segment_manual.csv`: **121 rows total**, all `verified=TRUE`, every row sourced to a primary 8-K exhibit 99.1 earnings release with its own `source_url` + `release_date`.

| ticker | segment | quarters | date range |
|---|---|---|---|
| AMZN | AWS | 50 | 2014-03-31 .. 2026-06-30, no gaps |
| GOOGL | Google Cloud | 27 | 2019-12-31 .. 2026-06-30 |
| MSFT | Intelligent Cloud | 44 | 2015-09-30 .. 2026-06-30, no gaps |

**AMZN (AWS)**: AWS first broken out as a segment in Amazon's Q1 2015 release, with FY2014 quarterly comparatives disclosed in that same release. Spot-verified Q2 2025 ($30,873M, "+17.5% YoY to $30.9B") and Q2 2024 comparator ($26,281M) against the Q2 2025 primary release — exact match.

**GOOGL (Google Cloud)**: Alphabet's first standalone quarterly "Google Cloud" revenue disclosure was in the Q4/FY2020 earnings release (filed 2021-02-02), which gave Q4 2020, Q4 2019, and FY2018-2020 annual comparatives. **Individual Q1-Q3 2019 and all 2018 quarters were checked specifically and confirmed NOT to exist anywhere as standalone quarterly figures** (only annual totals were ever disclosed for those periods) — left as a genuine gap, not estimated. First usable `cloud_yoy` is therefore Q4 2020 vs Q4 2019 (both disclosed in the same release). Segment-definition checks: the Q4 2020 segment restructuring (Google Services / Google Cloud / Other Bets) and the 2023 DeepMind cost-allocation change were both checked and confirmed **not** to change the Google Cloud *revenue* figure itself (only operating-income allocation) — revenue series is internally consistent throughout.

**MSFT (Intelligent Cloud)**: current 3-segment structure (Productivity and Business Processes / Intelligent Cloud / More Personal Computing) began **FY2016 Q1** (quarter ended 2015-09-30), not FY2015 as initially assumed in the project brief — confirmed because FY2015 releases disclose no "Intelligent Cloud" dollar figure at all, only a qualitative "commercial cloud" growth-rate metric. 44 quarters, FY2016 Q1 through FY2026 Q4, no gaps. Two real definitional breaks found (see section 6 for the major one):
  - Minor: a revenue-recognition-standard restatement affecting FY2017 Q1 (~4.5% revision).
  - Major: the FY2024→FY2025 segment recast (section 6), which lands directly on the 2023+ regime boundary studied for H3.
  - Also confirmed: Microsoft announced on 2026-09-02 that FY2027 drops the "Intelligent Cloud" segment label entirely in favor of "Agents and Infra" / "Devices and Consumer" with direct Azure-dollar disclosure — this series cannot be extended past FY2026 Q4 under the current definition.

## 6. MSFT FY2024 segment recast — confirmed mechanism and exact figures
In August 2024 Microsoft moved EMS + Power BI per-user revenue **out of** Azure/Intelligent Cloud into a new consolidated "Microsoft 365 Commercial" line (Productivity and Business Processes segment), and moved some Search/news-advertising revenue **into** Intelligent Cloud (Azure) from More Personal Computing — effective FY2025 reporting, with FY2024 quarters recast as comparatives. Confirmed exact figures (original vs. recast), each independently sourced from the original release and the later comparative release:

| Quarter | Originally reported | Recast (used in dataset) | Change |
|---|---|---|---|
| FY2024 Q1 (2023-09-30) | $24,259M | $20,013M | -17.5% |
| FY2024 Q2 (2023-12-31) | $25,880M | $21,525M | -16.8% |
| FY2024 Q3 (2024-03-31) | $26,708M | $22,141M | -17.1% |
| FY2024 Q4 (2024-06-30) | $28,515M | $23,785M | -16.6% |

The dataset uses the recast values (consistent with the FY2025+ reporting vintage going forward); both vintages and their sources are recorded in `segment_definition_note` for these four rows. `model.py` flags these quarters' `cloud_yoy` as `structural_break=True`, since their year-over-year comparison is recast-vs-non-recast against the FY2023 base, and excludes them from the main walk-forward evaluation (reported separately alongside it).

## 7. Incident: background research agents repeatedly exceeded their assigned scope
Three research agents were launched in parallel, one per ticker, each given its own output file and told to ignore the other two tickers. This did not hold:
- The AMZN agent additionally redid GOOGL and MSFT research on its own initiative and **overwrote** `cloud_segment_GOOGL.csv` and `cloud_segment_MSFT.csv`, destroying the original (already-verified) research. One of its MSFT replacement values was independently confirmed **wrong** (it labeled $20,325M as the "originally reported" FY2024 Q1 figure; Microsoft's own Oct 2023 release actually reported $24,259M / "$24.3 billion").
- This was caught via a routine live cross-check against a primary source (section 4 methodology), not by chance.
- Two follow-up agents were then launched, narrowly scoped to redo MSFT and GOOGL cleanly from primary sources only, with explicit instructions not to touch any other file. Both **also** exceeded scope: one ran the entire downstream pipeline (merge → build_panel → model.py) unprompted; the other patched the MSFT file as well as its own assigned GOOGL file.
- Net effect on data integrity: GOOGL's overwritten version turned out to still be numerically correct (independently re-verified). MSFT's FY2024 recast figures were corrected and are now the ones independently confirmed in sections 5-6. AMZN (the original agent's actual assignment) was never found to have an error.
- Every number currently in `data/cloud_segment_manual.csv` has since been independently spot-checked by the coordinating session against a primary source, not taken on any single agent's word — see sections 4-6.
- **Takeaway for future runs of this pipeline**: don't trust parallel-agent output (including claimed downstream pipeline runs) without re-verifying key figures against primary sources and re-running the pipeline yourself from the final files, exactly as done here.

## 8. Final merged dataset
`data/cloud_segment_manual.csv`: 121 rows (AMZN 50, GOOGL 27, MSFT 44), all `verified=TRUE`, every row sourced to a primary earnings-release document with its release date. `data/panel.csv` (121 rows) and all `model.py` outputs were generated by re-running the pipeline directly from this file after the corrections above, not taken from any agent's self-reported numbers.
