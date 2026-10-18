# Hyperscaler Capital Expenditure & Cloud Segment Revenue Growth: A Predictive Analytics Study

> **Course:** Predictive Analytics  
> **Institution:** Cakrawala University  
> **Academic Year:** 2026/2027  
> **Repository:** [titavianne/capex-cloud-predictive-analytics](https://github.com/titavianne/capex-cloud-predictive-analytics)  
> **Dataset Registry:** [s.id/listdatapred02](https://s.id/listdatapred02)

---

## 👥 Group Members

| Full Name | Student ID | Primary Role |
|---|---|---|
| **TITA NOVIANA** | `24120500011` | Lead Data Engineering & Econometric Modeling |
| **SUCI FRANSISCA SISILIA R** | `24120500008` | Lead Machine Learning & Hyperparameter Tuning |
| **NADIRA DIKA KAIVALYA** | `24110500023` | Lead Business Translation, Regime Analysis & Governance |

---

## 📌 Executive Summary & Research Problem

Hyperscalers (**Amazon AWS**, **Alphabet Google Cloud**, and **Microsoft Intelligent Cloud/Azure**) are currently deploying an unprecedented **$30B–$50B+ each in annual capital expenditures (Capex)**, predominantly directed toward AI datacenter infrastructure and specialized accelerators. 

A central assumption across Wall Street and corporate strategy is that aggressive infrastructure investment leads directly to accelerated cloud segment revenue growth over a 1–4 quarter horizon. 

This project subjects this assumption to rigorous empirical time-series testing using econometric and machine learning models under a **strictly causal, one-step-ahead walk-forward validation framework (zero data leakage)**.

### Core Hypotheses Tested:
- **$H_1$ (Capex Transmission):** Hyperscaler capex growth at lags 1–4 quarters contains predictive information for short-term cloud revenue growth ($y_{t}$).
- **$H_2$ (ML Superiority):** Regularized linear models and gradient-boosted decision trees outperform simple autoregressive benchmarks.
- **$H_3$ (Structural Regime Shift):** The relationship between capital deployment and revenue generation shifted fundamentally with the post-2023 Generative AI boom.

---

## 📊 Dataset Overview

The panel dataset covers **121 quarterly observations across 12+ years (2014–2026)**:

| Hyperscaler | Cloud Segment Analyzed | Usable Quarters | Date Range | Primary Source |
|---|---|:---:|:---:|---|
| **Amazon (AMZN)** | AWS | 50 | 2014-Q1 → 2026-Q2 | SEC 10-K/Q & Form 8-K Ex. 99.1 |
| **Microsoft (MSFT)** | Intelligent Cloud | 44 | 2015-Q3 → 2026-Q2 | SEC 10-K/Q & Form 8-K Ex. 99.1 |
| **Alphabet (GOOGL)** | Google Cloud | 27 | 2019-Q4 → 2026-Q2 | SEC 10-K/Q & Form 8-K Ex. 99.1 |
| **Macroeconomic** | Fed Funds Effective Rate | — | Quarterly | Federal Reserve Bank of St. Louis (FRED) |

- **Zero Data Leakage:** All engineered features are lagged $\le t-1$.
- **Structural Break Accounting:** Microsoft's FY2024 segment recast (-17% revenue reallocation) is explicitly audited and isolated during model training and evaluation.

---

## 🔬 Methodology & Evaluation Protocol

Because financial time-series exhibit temporal dependency and structural drift, **standard K-Fold cross-validation is strictly avoided** (as random shuffling causes severe look-ahead leakage). 

Instead, models are evaluated via **One-Step-Ahead Expanding-Window Walk-Forward Validation**:
1. Initial minimum 8-quarter training burn-in window.
2. Models fit on historical data $[0, \dots, t-1]$ and forecast quarter $t$.
3. Preprocessing scalers are fitted solely on training partitions.
4. Evaluation Metrics (Continuous Regression Target):
   - **RMSE** (Root Mean Squared Error) — Primary loss metric.
   - **MAE** (Mean Absolute Error) — Percentage-point unit error.
   - **Directional Accuracy (%)** — Percentage of correct quarter-over-quarter trend direction predictions.
   - *(Classification metrics such as F1/Precision/Recall are mathematically inapplicable for continuous regression).*

---

## 🏆 Key Empirical Findings

### 1. Cross-Model Performance Summary (Out-of-Sample RMSE)

| Ticker (N) | Naive Baseline | AR(1) ARIMA | Ridge (Tuned) | Lasso (Tuned) | XGBoost (Tuned) | Model Winner |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AMZN** (29) | **0.0345** | 0.0381 | 0.0975 | 0.1583 | 0.1089 | 🏆 **Naive** |
| **GOOGL** (14) | **0.0816** | 0.0902 | 0.0887 | 0.0944 | 0.1303 | 🏆 **Naive** |
| **MSFT** (27) | 0.0550 | 0.0541 | 0.0614 | **0.0523** | 0.0561 | 🏆 **Lasso** |
| **OVERALL** (70) | **0.0547** | 0.0579 | 0.0835 | 0.1150 | 0.0976 | 🏆 **Naive** |

### 2. Core Takeaways:
- **Capex Lags Have Zero Predictive Power ($H_1$ Rejected):** Lasso regularizes all capex growth features (`capex_yoy_lag1..4`) to exactly **0.000**. The only surviving predictor is the autoregressive term (`cloud_yoy_lag1`).
- **Persistence Baseline Outperforms Machine Learning ($H_2$ Rejected):** Adding 9 engineered capex and macro features does not beat a simple naive persistence forecast.
- **Regime Shift Post-2023 ($H_3$ Confirmed):** Models trained exclusively on pre-2023 data experience a 2–3× error inflation when tested on post-2023 quarters due to macro volatility and internal AI R&D absorption.

---

## 💼 Strategic C-Level Recommendations

1. **Decouple Cloud Revenue Guidance from Capex Announcements:** CFOs should cease relying on quarterly capex as a short-term proxy for cloud revenue acceleration. Guidance should be pegged to Remaining Performance Obligations (RPO) and organic workload commitments.
2. **Implement Modular, Phased Infrastructure Commitments:** Capacity planning should use staged GPU procurement tranches and lease flexibility to mitigate downside risk under fluctuating interest rates.

---

## 📁 Repository Deliverables

```text
├── Predictive_Analytics_Hyperscaler_Capex_Cloud_Revenue.ipynb   # Master notebook with complete execution outputs
├── Presentation_Predictive_Analytics.pptx                      # 12-slide executive presentation
├── contribution_statement.pdf                                  # Signed contribution statement (LaTeX/ReportLab)
├── contribution_statement.tex                                  # LaTeX source for contribution statement
├── data/
│   ├── panel.csv                                               # Complete financial panel (121 rows, 24 features)
│   └── cloud_segment_manual.csv                                # Curated and verified 8-K cloud revenues
├── generate_slides.py                                          # Script to generate PPTX presentation
├── generate_contribution_pdf.py                                # Script to compile contribution statement PDF
├── update_and_run_notebook.py                                  # Script to build and execute notebook
└── requirements.txt                                            # Python environment dependencies
```

---

## 🚀 How to Reproduce

1. **Clone the repository:**
   ```bash
   git clone https://github.com/titavianne/capex-cloud-predictive-analytics.git
   cd capex-cloud-predictive-analytics
   ```

2. **Set up virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook Predictive_Analytics_Hyperscaler_Capex_Cloud_Revenue.ipynb
   ```
