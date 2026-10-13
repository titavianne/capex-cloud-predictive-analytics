"""
Predictive model for cloud_yoy (quarterly YoY growth of hyperscaler cloud
SEGMENT revenue: MSFT Intelligent Cloud, GOOGL Google Cloud, AMZN AWS).

Design (fixed by project brief, do not change without asking):
  - One-step-ahead expanding-window walk-forward validation (NOT random split).
  - Features are all lagged (<=t-1 information) -- see build_panel.py.
  - Models: naive (cloud_yoy_lag1), AR(1) per ticker, Ridge/Lasso (standardized,
    TimeSeriesSplit-tuned alpha), small XGBoost (max_depth 2-3).
  - Metrics: RMSE, MAE (MAPE only if explicitly discussed -- growth rates near
    zero make MAPE unstable/explosive).
  - Regime test (H3): train through 2022-12-31, test 2023-01-01 onward. Also
    fit Lasso separately on pre-/post-2023 data and compare capex-lag coefficients.
  - Stationarity (ADF) on features/target; cross-correlation (CCF) across lags.
  - Every model is compared to the naive baseline; a model that does not beat
    the baseline is reported as such -- that is a legitimate finding, not a bug.

Known data caveats carried into this analysis (see data/cloud_segment_manual.csv
segment_definition_note and VALIDATION_REPORT.md):
  - MSFT recast its segment definition effective FY2025 reporting, moving part
    of Windows Commercial / M365 Commercial into Productivity & Business
    Processes; FY2024 quarters (period_end 2023-09-30 .. 2024-06-30) are
    recast-vs-original comparisons for cloud_yoy and are flagged as a
    structural break overlapping the 2023+ regime window studied for H3.
  - GOOGL Q1-Q3 2018 individual quarters are unavailable (only FY2018 total and
    Q4 2018 were ever disclosed) -- cloud_yoy for GOOGL only starts Q4 2019
    (first quarter with a true prior-year-quarter comparator).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, ccf
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PANEL_PATH = BASE_DIR / "data" / "panel.csv"
REGIME_CUTOFF = pd.Timestamp("2022-12-31")

FEATURES = [
    "capex_yoy_lag1", "capex_yoy_lag2", "capex_yoy_lag3", "capex_yoy_lag4",
    "fed_funds_chg_yoy_lag", "capex_to_rev_lag1", "fcf_margin_lag1",
    "debt_yoy_lag1", "cloud_yoy_lag1_feat",
]
TARGET = "cloud_yoy"

# MSFT quarters where cloud_yoy spans the FY2025 segment recast (recast current
# quarter vs. non-recast year-ago quarter) -- see docstring. Excluded from the
# main walk-forward evaluation; reported separately.
MSFT_STRUCTURAL_BREAK_QUARTERS = pd.to_datetime(
    ["2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]
)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH} -- run build_panel.py first.")
    df = pd.read_csv(PANEL_PATH, parse_dates=["period_end"])
    df["structural_break"] = (df["ticker"] == "MSFT") & (df["period_end"].isin(MSFT_STRUCTURAL_BREAK_QUARTERS))
    return df


def complete_case(df: pd.DataFrame) -> pd.DataFrame:
    cc = df.dropna(subset=[TARGET] + FEATURES).copy()
    return cc.sort_values(["ticker", "period_end"])


# ---------------------------------------------------------------------------
# Stationarity + cross-correlation diagnostics
# ---------------------------------------------------------------------------
def run_adf(series: pd.Series, label: str) -> dict:
    s = series.dropna()
    if len(s) < 8:
        return {"label": label, "n": len(s), "adf_stat": np.nan, "p_value": np.nan, "note": "too few obs"}
    stat, pval, *_ = adfuller(s, autolag="AIC")
    return {"label": label, "n": len(s), "adf_stat": stat, "p_value": pval}


def stationarity_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("period_end")
        rows.append(run_adf(g[TARGET], f"{ticker}:{TARGET}"))
        for f in ["capex_yoy_lag1", "fed_funds_chg_yoy_lag", "capex_to_rev_lag1", "fcf_margin_lag1", "debt_yoy_lag1"]:
            rows.append(run_adf(g[f], f"{ticker}:{f}"))
    return pd.DataFrame(rows)


def ccf_report(df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """Cross-correlation of capex_yoy (unlagged, from edgar-derived column if
    present) against cloud_yoy at lags 0..max_lag, per ticker."""
    rows = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("period_end")
        x = g["capex_yoy_lag1"].dropna()
        y = g[TARGET]
        common = g.dropna(subset=["capex_yoy_lag1", TARGET])
        if len(common) < 8:
            continue
        x = common["capex_yoy_lag1"].to_numpy()
        y = common[TARGET].to_numpy()
        c = ccf(y, x, adjusted=False)[: max_lag + 1]
        for lag, val in enumerate(c):
            rows.append({"ticker": ticker, "lag": lag, "ccf_cloud_vs_capexlag1": val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-forward (expanding window, one-step-ahead) evaluation
# ---------------------------------------------------------------------------
def walk_forward_indices(n: int, min_train: int):
    for i in range(min_train, n):
        yield list(range(i)), i


def fit_predict_naive(y_train, X_train_row, y_true_next):
    # naive = last observed cloud_yoy (the lag1 feature IS this value)
    return X_train_row["cloud_yoy_lag1_feat"]


def fit_predict_ar1(y_train):
    try:
        model = ARIMA(y_train.to_numpy(), order=(1, 0, 0))
        fit = model.fit()
        fc = fit.forecast(steps=1)
        return float(fc[0])
    except Exception:
        return float(y_train.iloc[-1]) if len(y_train) else np.nan


def tune_alpha(X_train, y_train, model_cls, alphas=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)):
    n = len(y_train)
    n_splits = max(2, min(4, n - 1))
    if n < n_splits + 2:
        return alphas[len(alphas) // 2]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_score = alphas[0], np.inf
    for a in alphas:
        errs = []
        for tr_idx, va_idx in tscv.split(X_train):
            if len(tr_idx) < 3 or len(va_idx) == 0:
                continue
            scaler = StandardScaler().fit(X_train[tr_idx])
            Xtr = scaler.transform(X_train[tr_idx])
            Xva = scaler.transform(X_train[va_idx])
            m = model_cls(alpha=a, max_iter=20000)
            m.fit(Xtr, y_train[tr_idx])
            pred = m.predict(Xva)
            errs.append(mean_squared_error(y_train[va_idx], pred))
        if errs and np.mean(errs) < best_score:
            best_score = np.mean(errs)
            best_alpha = a
    return best_alpha


def walk_forward_ticker(g: pd.DataFrame, min_train: int = 8) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    X_all = g[FEATURES].to_numpy(dtype=float)
    y_all = g[TARGET].to_numpy(dtype=float)
    results = []

    for train_idx, test_i in walk_forward_indices(len(g), min_train):
        y_train = g.loc[train_idx, TARGET]
        X_train = X_all[train_idx]
        X_test = X_all[test_i : test_i + 1]
        y_true = y_all[test_i]
        row = g.loc[test_i]

        pred_naive = row["cloud_yoy_lag1_feat"]
        pred_ar1 = fit_predict_ar1(y_train)

        alpha_r = tune_alpha(X_train, y_train.to_numpy(), Ridge)
        scaler = StandardScaler().fit(X_train)
        ridge = Ridge(alpha=alpha_r).fit(scaler.transform(X_train), y_train)
        pred_ridge = float(ridge.predict(scaler.transform(X_test))[0])

        alpha_l = tune_alpha(X_train, y_train.to_numpy(), Lasso)
        lasso = Lasso(alpha=alpha_l, max_iter=20000).fit(scaler.transform(X_train), y_train)
        pred_lasso = float(lasso.predict(scaler.transform(X_test))[0])

        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        )
        xgb_model.fit(X_train, y_train)
        pred_xgb = float(xgb_model.predict(X_test)[0])

        results.append({
            "ticker": row["ticker"], "period_end": row["period_end"],
            "y_true": y_true, "structural_break": row["structural_break"],
            "pred_naive": pred_naive, "pred_ar1": pred_ar1,
            "pred_ridge": pred_ridge, "pred_lasso": pred_lasso, "pred_xgb": pred_xgb,
        })
    return pd.DataFrame(results)


def score_predictions(df: pd.DataFrame, exclude_breaks: bool = True) -> pd.DataFrame:
    d = df[~df["structural_break"]] if exclude_breaks else df
    models = ["naive", "ar1", "ridge", "lasso", "xgb"]
    rows = []
    for ticker in list(d["ticker"].unique()) + ["ALL"]:
        sub = d if ticker == "ALL" else d[d["ticker"] == ticker]
        if sub.empty:
            continue
        for m in models:
            col = f"pred_{m}"
            valid = sub.dropna(subset=["y_true", col])
            if valid.empty:
                continue
            rows.append({
                "ticker": ticker, "model": m, "n": len(valid),
                "rmse": rmse(valid["y_true"], valid[col]),
                "mae": mae(valid["y_true"], valid[col]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime split (H3): train <=2022-12-31, test 2023+; separate Lasso fits
# ---------------------------------------------------------------------------
def regime_lasso_comparison(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["period_end"] <= REGIME_CUTOFF]
    post = df[df["period_end"] > REGIME_CUTOFF]
    rows = []
    for label, sub in [("pre_2023", pre), ("post_2023", post)]:
        sub = sub.dropna(subset=[TARGET] + FEATURES)
        if len(sub) < 8:
            rows.append({"period": label, "n": len(sub), "note": "too few obs (<8), coefficients not reliable"})
            continue
        X = sub[FEATURES].to_numpy(dtype=float)
        y = sub[TARGET].to_numpy(dtype=float)
        scaler = StandardScaler().fit(X)
        alpha = tune_alpha(X, y, Lasso)
        lasso = Lasso(alpha=alpha, max_iter=20000).fit(scaler.transform(X), y)
        coefs = dict(zip(FEATURES, lasso.coef_))
        coefs.update({"period": label, "n": len(sub), "alpha": alpha})
        rows.append(coefs)
    return pd.DataFrame(rows)


def regime_train_test(df: pd.DataFrame) -> pd.DataFrame:
    """Train through 2022-12-31 (per ticker), test on 2023+ one-step-ahead
    (reusing the already-lagged features; not a walk-forward refit per step
    here -- single train fit, multiple test points, since this is specifically
    testing regime STABILITY of one fitted model, not re-tuning each quarter)."""
    rows = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("period_end").dropna(subset=[TARGET] + FEATURES)
        train = g[g["period_end"] <= REGIME_CUTOFF]
        test = g[g["period_end"] > REGIME_CUTOFF]
        if len(train) < 8 or test.empty:
            rows.append({"ticker": ticker, "n_train": len(train), "n_test": len(test), "note": "insufficient data for regime test"})
            continue
        Xtr, ytr = train[FEATURES].to_numpy(dtype=float), train[TARGET].to_numpy(dtype=float)
        Xte, yte = test[FEATURES].to_numpy(dtype=float), test[TARGET].to_numpy(dtype=float)
        scaler = StandardScaler().fit(Xtr)
        alpha = tune_alpha(Xtr, ytr, Lasso)
        lasso = Lasso(alpha=alpha, max_iter=20000).fit(scaler.transform(Xtr), ytr)
        pred = lasso.predict(scaler.transform(Xte))
        naive_pred = test["cloud_yoy_lag1_feat"].to_numpy()
        rows.append({
            "ticker": ticker, "n_train": len(train), "n_test": len(test),
            "lasso_rmse": rmse(yte, pred), "lasso_mae": mae(yte, pred),
            "naive_rmse": rmse(yte, naive_pred), "naive_mae": mae(yte, naive_pred),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2030 scenario projection (bear/base/bull) -- explicitly labeled assumptions,
# NOT a validated forecast. Widening intervals, no claim of accuracy.
# ---------------------------------------------------------------------------
def scenario_2030(df: pd.DataFrame) -> pd.DataFrame:
    """Simple compounding scenario from the last observed cloud revenue level
    per ticker, using explicit assumed annual growth rates for bear/base/bull,
    with linearly widening uncertainty bands. This is a labeled assumption
    exercise, not a model-validated projection -- see report."""
    assumptions = {"bear": 0.05, "base": 0.15, "bull": 0.30}  # assumed ANNUAL growth
    rows = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("period_end")
        last = g.iloc[-1]
        last_rev = last.get("cloud_revenue")
        last_date = last["period_end"]
        if pd.isna(last_rev):
            continue
        years_to_2030 = 2030 - last_date.year
        for scenario, annual_g in assumptions.items():
            for y in range(1, max(years_to_2030, 1) + 1):
                proj = last_rev * ((1 + annual_g) ** y)
                band_pct = 0.05 * y  # widening band, illustrative only
                rows.append({
                    "ticker": ticker, "scenario": scenario, "year": last_date.year + y,
                    "projected_annual_cloud_revenue": proj,
                    "low": proj * (1 - band_pct), "high": proj * (1 + band_pct),
                    "assumption_annual_growth": annual_g,
                })
    return pd.DataFrame(rows)


def main():
    df = load_panel()
    cc = complete_case(df)

    print("=== Complete-case sample sizes per ticker ===")
    print(cc.groupby("ticker").size())
    for t in ["MSFT", "GOOGL", "AMZN"]:
        n = (cc["ticker"] == t).sum()
        if n < 8:
            print(f"STOP/REVIEW: {t} has only {n} complete-case quarters (<8) -- "
                  f"walk-forward + regime split will not be reliable. Consider the "
                  f"documented fallback (total-revenue-growth / capex-intensity only).")

    print("\n=== ADF stationarity ===")
    adf_df = stationarity_report(cc)
    print(adf_df.to_string(index=False))

    print("\n=== CCF: cloud_yoy vs capex_yoy_lag1, lags 0-4 ===")
    print(ccf_report(cc).to_string(index=False))

    print("\n=== Walk-forward (expanding window, one-step-ahead) ===")
    wf_frames = [walk_forward_ticker(g) for _, g in cc.groupby("ticker") if len(g) >= 9]
    if not wf_frames:
        print("Not enough data for walk-forward on any ticker.")
        wf = pd.DataFrame()
    else:
        wf = pd.concat(wf_frames, ignore_index=True)
        scores = score_predictions(wf, exclude_breaks=True)
        print(scores.to_string(index=False))
        if wf["structural_break"].any():
            print("\n(MSFT FY2024 structural-break quarters excluded above; shown separately if present:)")
            scores_incl = score_predictions(wf, exclude_breaks=False)
            print(scores_incl.to_string(index=False))

    print("\n=== Regime comparison (train<=2022-12-31, test 2023+) ===")
    print(regime_train_test(cc).to_string(index=False))

    print("\n=== Lasso coefficients, pre- vs post-2023 (H3) ===")
    print(regime_lasso_comparison(cc).to_string(index=False))

    print("\n=== 2030 scenarios (bear/base/bull, explicit assumptions, NOT validated) ===")
    scen = scenario_2030(df)
    print(scen.to_string(index=False))

    out_dir = BASE_DIR / "data" / "processed"
    if not wf.empty:
        wf.to_csv(out_dir / "walk_forward_predictions.csv", index=False)
    scen.to_csv(out_dir / "scenario_2030.csv", index=False)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
