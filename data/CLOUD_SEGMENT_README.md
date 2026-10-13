# cloud_segment_manual.csv — sourcing rules

Cloud segment revenue is **not** available from SEC companyfacts (it only exposes
consolidated entity-level XBRL facts, not segment-dimensioned ones). Every row in
this file must be manually sourced from a primary document: the company's own
earnings press release (8-K exhibit), investor-relations quarterly results page,
or 10-Q/10-K segment footnote.

## Rules (do not violate)
- **Never fill `cloud_revenue_usd` from memory.** If you cannot find a primary
  source for a quarter, leave the row's value/source/release_date blank and list
  it as a gap in your report. Do not estimate or interpolate.
- `source_url` must point to the specific document (press release, 8-K exhibit,
  10-Q/10-K) that states the number, not a search results page or aggregator
  unless the aggregator itself cites and link to the primary filing and you
  verified the number there matches.
- `release_date` = the date that document was issued/filed (not today's date).
- `segment_definition_note`: record here any change in what the segment includes
  (e.g. a product line moved in/out of the segment) effective that quarter, with
  a brief description. Known risk points to check explicitly:
  - MSFT has regrouped segment contents within "Intelligent Cloud" over time
    (e.g. Nuance, GitHub placement) — verify against each 10-K's segment note.
  - GOOGL "Google Cloud" segment definition (GCP + Google Workspace) and the
    2023 Alphabet reorganization — verify whether this changed what's included.
- `verified` = TRUE only once you've opened the source and matched the number.

## Columns
| column | meaning |
|---|---|
| ticker | MSFT / AMZN / GOOGL |
| segment_name | as reported, e.g. "Intelligent Cloud", "AWS", "Google Cloud" |
| fiscal_year_label | company's own fiscal year label, e.g. "FY2023" |
| fiscal_quarter_label | company's own fiscal quarter label, e.g. "Q2" |
| period_end | ISO date of the quarter end (calendar date, for joining to EDGAR data which is calendar-dated) |
| cloud_revenue_usd | segment revenue in whole USD (not millions) |
| segment_definition_note | free text, see above |
| source_url | link to primary source document |
| release_date | ISO date the source document was issued |
| filled_by | who/what filled this row |
| verified | TRUE/FALSE |

## Target coverage (fill as much as verifiable; report gaps, do not force)
- AMZN "AWS": segment first broken out ~Q1 2015 (10-Q filed Apr 2015), with 2014
  comparatives often available in that filing. Target back to Q1 2014 if sourced,
  else start at first verifiable quarter.
- GOOGL "Google Cloud": segment first broken out ~Q1 2019 (10-Q filed Apr 2019),
  with 2018 comparatives. Target back to Q1 2018 if sourced.
- MSFT "Intelligent Cloud": current 3-segment structure began FY2015 Q1 (quarter
  ended Sep 30, 2014). Target back to FY2015 Q1 if sourced.
- All three: extend forward to the most recently *released* quarter as of the
  research date. Do not guess at a quarter that hasn't been reported yet.
