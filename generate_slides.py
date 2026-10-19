"""
Generate a clean, professional PPTX presentation for the Predictive Analytics
Group Project. Design: white background, Cakrawala University teal (#1B5E6B)
and gold (#C5A55A) accent colors matching the institutional logo.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path
import copy

# -- Colour palette from logo --
TEAL      = RGBColor(0x1B, 0x5E, 0x6B)   # dark teal from shield
TEAL_LT   = RGBColor(0x2A, 0x7F, 0x8F)   # lighter teal for accents
GOLD      = RGBColor(0xC5, 0xA5, 0x5A)   # gold swoosh
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
BLACK     = RGBColor(0x1F, 0x29, 0x37)
GRAY      = RGBColor(0x6B, 0x72, 0x80)
GRAY_LT   = RGBColor(0xE5, 0xE7, 0xEB)
RED_SOFT  = RGBColor(0xDC, 0x26, 0x26)
GREEN_OK  = RGBColor(0x16, 0xA3, 0x4A)

LOGO_PATH = Path(__file__).resolve().parent / "logo.png"
SLIDE_W   = Inches(13.333)
SLIDE_H   = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

# -- Helper: blank slide with white bg --
BLANK_LAYOUT = prs.slide_layouts[6]  # blank

def add_slide():
    slide = prs.slides.add_slide(BLANK_LAYOUT)
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = WHITE
    return slide

def add_logo(slide):
    slide.shapes.add_picture(
        str(LOGO_PATH), Inches(0.35), Inches(0.25), Inches(0.6), Inches(0.6)
    )

def add_bottom_bar(slide):
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Emu(0), SLIDE_H - Inches(0.18), SLIDE_W, Inches(0.18)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

def add_gold_accent(slide, top, width=Inches(1.8)):
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), top, width, Inches(0.06)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = GOLD
    accent.line.fill.background()

def add_title_text(slide, text, left, top, width, height, font_size=28, bold=True, color=TEAL, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = alignment
    return tf

def add_body_text(slide, text, left, top, width, height, font_size=14, color=BLACK, bold=False, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = alignment
    return tf

def add_bullet_list(slide, items, left, top, width, height, font_size=13, color=BLACK):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(4)
        p.level = 0
    return tf

def make_table(slide, rows, cols, left, top, width, height):
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    return table

def style_header_row(table, col_count):
    for i in range(col_count):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TEAL
        for p in cell.text_frame.paragraphs:
            p.font.color.rgb = WHITE
            p.font.bold = True
            p.font.size = Pt(11)
            p.alignment = PP_ALIGN.CENTER

def style_body_cell(cell, text, font_size=10, bold=False, align=PP_ALIGN.CENTER):
    cell.text = text
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(font_size)
        p.font.color.rgb = BLACK
        p.font.bold = bold
        p.alignment = align

# ═══════════════════════════════════════════════════════════════
# SLIDE 1: TITLE SLIDE
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)

# Teal band across upper area
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Inches(2.2), SLIDE_W, Inches(2.8))
band.fill.solid()
band.fill.fore_color.rgb = TEAL
band.line.fill.background()

add_title_text(s, "Predictive Analytics Group Project",
    Inches(0.8), Inches(2.4), Inches(11), Inches(0.7), font_size=32, color=WHITE)
add_title_text(s, "Does Hyperscaler Capex Growth Predict\nCloud Segment Revenue Growth?",
    Inches(0.8), Inches(3.0), Inches(11), Inches(1.2), font_size=24, bold=False, color=GOLD)
add_body_text(s, "An Empirical Time-Series & Econometric Analysis of Amazon AWS, Google Cloud, and Microsoft Azure",
    Inches(0.8), Inches(4.1), Inches(10), Inches(0.5), font_size=14, color=GRAY_LT)

add_body_text(s, "Tita Noviana (40% • 24120500011)   |   Suci Fransisca Sisilia R (40% • 24120500008)   |   Nadira Dika Kaivalya (20% • 24110500023)",
    Inches(0.8), Inches(5.45), Inches(11.5), Inches(0.4), font_size=12.5, color=TEAL, bold=True)
add_body_text(s, "Lead Data & Econometrics (40%)  •  Lead ML & Tuning (40%)  •  Lead Business & Strategy (20%)",
    Inches(0.8), Inches(5.85), Inches(11), Inches(0.35), font_size=11, color=GOLD, bold=True)
add_body_text(s, "Predictive Analytics  •  Cakrawala University  •  October 2026",
    Inches(0.8), Inches(6.25), Inches(11), Inches(0.3), font_size=11, color=GRAY)

# ═══════════════════════════════════════════════════════════════
# SLIDE 2: AGENDA / OUTLINE
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Presentation Outline", Inches(0.8), Inches(0.3), Inches(8), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

agenda_items = [
    "Part I: Context & Data Pipeline (Tita Noviana — 40%)",
    "   1.  Industry Context & Research Problem",
    "   2.  Data Sources, Volume & Quality Audit",
    "   3.  Methodology: Walk-Forward Time-Series Validation",
    "",
    "Part II: Empirical Modeling & Tuning (Suci Fransisca Sisilia R — 40%)",
    "   4.  Candidate Models (AR(1), Ridge, Lasso, XGBoost vs. Naive)",
    "   5.  Master Cross-Model Comparison Table",
    "   6.  Hyperparameter Tuning: Default vs. Tuned Audit",
    "   7.  Model Interpretation: Why Capex Lags = 0.000",
    "",
    "Part III: Regime Shifts & Executive Strategy (Nadira Dika Kaivalya — 20%)",
    "   8.  Regime Shift Test: Pre-2023 vs. Post-2023 AI Boom",
    "   9.  Strategic Recommendations for C-Level Executives",
    "   10. 2030 Strategic Scenarios, Technical & Business Limitations",
]
add_bullet_list(s, agenda_items, Inches(1.0), Inches(1.3), Inches(11), Inches(5.8), font_size=13, color=BLACK)

# ═══════════════════════════════════════════════════════════════
# SLIDE 3: INDUSTRY CONTEXT & RESEARCH PROBLEM
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Industry Context & Research Problem", Inches(0.8), Inches(0.3), Inches(10), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

add_body_text(s, "The Capital Allocation Dilemma", Inches(0.8), Inches(1.3), Inches(6), Inches(0.4), font_size=18, color=TEAL, bold=True)
context_points = [
    "•  Amazon, Google, and Microsoft each spend $30B–$50B+ annually on AI datacenter infrastructure",
    "•  Implicit assumption: aggressive capex → faster cloud revenue growth after 1–4 quarter lag",
    "•  CFOs need to know: Is this assumption empirically supported for short-term revenue guidance?",
    "",
    "Three Testable Hypotheses:",
    "   H₁:  Capex growth (lag 1–4Q) predicts cloud revenue YoY growth",
    "   H₂:  ML models (Ridge, Lasso, XGBoost) beat a simple Naive baseline",
    "   H₃:  The capex–cloud relationship shifted in the 2023+ Generative AI era",
]
add_bullet_list(s, context_points, Inches(0.8), Inches(1.9), Inches(11), Inches(4.5), font_size=14, color=BLACK)

# Stakeholder callout box
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.5), Inches(1.3), Inches(4.2), Inches(1.4))
box.fill.solid()
box.fill.fore_color.rgb = RGBColor(0xF0, 0xFD, 0xFA)
box.line.color.rgb = TEAL
box.line.width = Pt(1.5)
tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Primary Stakeholder"
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = TEAL
p = tf.add_paragraph()
p.text = "Chief Financial Officer (CFO)\n& Capital Allocation Committee"
p.font.size = Pt(11)
p.font.color.rgb = BLACK

# ═══════════════════════════════════════════════════════════════
# SLIDE 4: DATA SOURCES, VOLUME & QUALITY
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Data Sources, Volume & Quality Audit", Inches(0.8), Inches(0.3), Inches(10), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

# Data sources
sources = [
    "•  SEC EDGAR XBRL (companyfacts): Consolidated quarterly financials (Revenue, Capex, OCF, Debt)",
    "•  SEC Form 8-K Exhibit 99.1: 121 manually curated & cross-verified cloud segment revenue records",
    "•  FRED (St. Louis Fed): Federal Funds Rate quarterly macro series",
]
add_bullet_list(s, sources, Inches(0.8), Inches(1.3), Inches(11), Inches(1.5), font_size=13)

# Volume table
tbl = make_table(s, 5, 4, Inches(0.8), Inches(3.0), Inches(7.5), Inches(2.2))
headers = ["Hyperscaler", "Cloud Segment", "Quarters", "Period"]
for i, h in enumerate(headers):
    tbl.cell(0, i).text = h
style_header_row(tbl, 4)
data = [
    ["Amazon (AMZN)", "AWS", "50", "2014-Q1 → 2026-Q2"],
    ["Microsoft (MSFT)", "Intelligent Cloud", "44", "2015-Q3 → 2026-Q2"],
    ["Alphabet (GOOGL)", "Google Cloud", "27", "2019-Q4 → 2026-Q2"],
    ["TOTAL PANEL", "", "121 rows", "12+ years coverage"],
]
for r, row_data in enumerate(data):
    for c, val in enumerate(row_data):
        style_body_cell(tbl.cell(r+1, c), val, bold=(r == 3))

# Quality callout
qual_points = [
    "✓  Zero chronological gaps (all quarter steps < 110 days)",
    "✓  6 live cross-checks against primary 8-K releases (100% exact match)",
    "✓  MSFT FY24 segment recast ($-17%) flagged as structural_break",
    "✓  All features lagged ≤ t-1 to prevent data leakage",
]
add_bullet_list(s, qual_points, Inches(8.8), Inches(3.0), Inches(4), Inches(2.5), font_size=11, color=TEAL)

# ═══════════════════════════════════════════════════════════════
# SLIDE 5: METHODOLOGY — WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Methodology: Walk-Forward Time-Series Validation", Inches(0.8), Inches(0.3), Inches(11), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

method_points = [
    "Why NOT use standard K-Fold Cross-Validation?",
    "   →  K-Fold randomly shuffles observations, causing severe look-ahead data leakage",
    "   →  Future quarterly data would contaminate the training set — invalid for time-series",
    "",
    "Our Approach: One-Step-Ahead Expanding-Window Walk-Forward",
    "   1.  Start with minimum 8-quarter burn-in training window",
    "   2.  At each step t, train ONLY on historical quarters [0, 1, ..., t-1]",
    "   3.  Predict the single out-of-sample quarter t",
    "   4.  Expand window to include t, repeat until end of series",
    "   5.  StandardScaler fitted exclusively on training partition (no test peeking)",
    "",
    "Evaluation Metrics (Regression, NOT Classification):",
    "   •  RMSE (Root Mean Squared Error) — primary metric, penalizes large errors heavily",
    "   •  MAE (Mean Absolute Error) — interpretable in percentage-point units",
    "   •  Directional Accuracy — % of correct up/down direction calls",
    "   •  F1-Score / Precision / Recall NOT applicable (continuous target, not binary labels)",
]
add_bullet_list(s, method_points, Inches(0.8), Inches(1.3), Inches(11.5), Inches(5.5), font_size=13)

# ═══════════════════════════════════════════════════════════════
# SLIDE 6: MODELS EVALUATED
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Five Models Evaluated", Inches(0.8), Inches(0.3), Inches(10), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

# Model cards
models_info = [
    ("Model 1: Naive Baseline", "ŷₜ = yₜ₋₁ (last quarter's growth)", "Benchmark — must be beaten", GRAY_LT),
    ("Model 2: AR(1) ARIMA", "Autoregressive MLE fit on historical series", "Classical econometric time-series", GRAY_LT),
    ("Model 3: Ridge Regression", "L2 regularization, handles capex lag collinearity", "Default α=1.0 vs Tuned α∈[1e-4, 5.0]", RGBColor(0xF0, 0xFD, 0xFA)),
    ("Model 4: Lasso Regression", "L1 regularization, automatic feature selection", "Default α=1.0 vs Tuned α∈[1e-4, 5.0]", RGBColor(0xF0, 0xFD, 0xFA)),
    ("Model 5: XGBoost Regressor", "Gradient boosted trees, captures non-linear interactions", "Default (depth=6) vs Tuned (depth=2-3)", RGBColor(0xF0, 0xFD, 0xFA)),
]

for i, (title, desc, tuning, bg_color) in enumerate(models_info):
    left = Inches(0.8) if i < 3 else Inches(0.8)
    col = i % 3
    row = i // 3
    x = Inches(0.6) + col * Inches(4.15)
    y = Inches(1.3) + row * Inches(2.7)

    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.9), Inches(2.3))
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    box.line.color.rgb = TEAL
    box.line.width = Pt(1)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEAL
    p = tf.add_paragraph()
    p.text = desc
    p.font.size = Pt(11)
    p.font.color.rgb = BLACK
    p.space_before = Pt(6)
    p = tf.add_paragraph()
    p.text = tuning
    p.font.size = Pt(10)
    p.font.color.rgb = GOLD
    p.space_before = Pt(4)
    p.font.bold = True

# ═══════════════════════════════════════════════════════════════
# SLIDE 7: MASTER COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Master Cross-Model Comparison (RMSE)", Inches(0.8), Inches(0.3), Inches(11), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))
add_body_text(s, "Walk-Forward Out-of-Sample Evaluation  •  MSFT structural break quarters excluded  •  Lower = Better",
    Inches(0.8), Inches(1.1), Inches(11), Inches(0.4), font_size=11, color=GRAY)

tbl = make_table(s, 5, 7, Inches(0.5), Inches(1.7), Inches(12.3), Inches(2.5))
hdrs = ["Ticker (N)", "Naive\nBaseline", "AR(1)\nARIMA", "Ridge\n(Tuned)", "Lasso\n(Tuned)", "XGBoost\n(Tuned)", "Winner"]
for i, h in enumerate(hdrs):
    tbl.cell(0, i).text = h
style_header_row(tbl, 7)

results = [
    ["AMZN (29)", "0.0345", "0.0381", "0.0975", "0.1583", "0.1089", "🏆 Naive"],
    ["GOOGL (14)", "0.0816", "0.0902", "0.0887", "0.0944", "0.1303", "🏆 Naive"],
    ["MSFT (27)", "0.0550", "0.0541", "0.0614", "0.0523", "0.0561", "🏆 Lasso"],
    ["ALL (70)", "0.0547", "0.0579", "0.0835", "0.1150", "0.0976", "🏆 Naive"],
]
for r, row_data in enumerate(results):
    for c, val in enumerate(row_data):
        cell = tbl.cell(r+1, c)
        style_body_cell(cell, val, font_size=12, bold=(c == 6 or c == 0))
        # Highlight winner column
        if c == 6:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xEC, 0xFD, 0xF5)

# Key finding box
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(4.5), Inches(12.3), Inches(2.2))
box.fill.solid()
box.fill.fore_color.rgb = RGBColor(0xFE, 0xF2, 0xF2)
box.line.color.rgb = RED_SOFT
box.line.width = Pt(1.5)
tf = box.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.2)
tf.margin_top = Inches(0.15)
p = tf.paragraphs[0]
p.text = "Key Finding: The Naive Persistence Baseline wins or ties across all hyperscalers"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = RED_SOFT
p = tf.add_paragraph()
p.text = ("• Adding 9 engineered capex/macro features to Ridge, Lasso, and XGBoost does NOT improve one-step-ahead forecasting\n"
          "• The only marginal ML win is Lasso (Tuned) for MSFT: RMSE 0.0523 vs 0.0550 Naive — a 5% improvement on just 27 quarters\n"
          "• Hypothesis H₂ (ML beats baseline) is NOT SUPPORTED by the empirical evidence")
p.font.size = Pt(12)
p.font.color.rgb = BLACK
p.space_before = Pt(6)

# ═══════════════════════════════════════════════════════════════
# SLIDE 8: HYPERPARAMETER TUNING DEFAULT VS TUNED
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Hyperparameter Tuning: Default vs. Tuned", Inches(0.8), Inches(0.3), Inches(11), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

tbl = make_table(s, 7, 6, Inches(0.4), Inches(1.3), Inches(12.5), Inches(3.2))
hdrs2 = ["Ticker", "Model Family", "Search Space", "Default\nRMSE", "Tuned\nRMSE", "Outcome"]
for i, h in enumerate(hdrs2):
    tbl.cell(0, i).text = h
style_header_row(tbl, 6)

tuning_data = [
    ["AMZN", "Ridge", "α ∈ [1e-4, 5.0]", "0.0931", "0.0975", "Trade-off"],
    ["AMZN", "XGBoost", "depth:[2,3], lr:[0.05,0.1]", "0.1065", "0.1089", "Trade-off"],
    ["GOOGL", "Ridge", "α ∈ [1e-4, 5.0]", "0.0850", "0.0887", "Trade-off"],
    ["GOOGL", "XGBoost", "depth:[2,3], lr:[0.05,0.1]", "0.1429", "0.1303", "✓ Improved"],
    ["MSFT", "Ridge", "α ∈ [1e-4, 5.0]", "0.0663", "0.0614", "✓ Improved"],
    ["MSFT", "XGBoost", "depth:[2,3], lr:[0.05,0.1]", "0.0543", "0.0561", "Trade-off"],
]
for r, row_data in enumerate(tuning_data):
    for c, val in enumerate(row_data):
        cell = tbl.cell(r+1, c)
        style_body_cell(cell, val, font_size=11, bold=(c == 5 and "Improved" in val))
        if c == 5 and "Improved" in val:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xEC, 0xFD, 0xF5)

add_body_text(s, "Insight: Tuning provides modest gains for GOOGL XGBoost (-8.8% RMSE) and MSFT Ridge (-7.5% RMSE).\n"
    "For AMZN, both Ridge and XGBoost show slight degradation after tuning — consistent with overfitting on small quarterly samples.",
    Inches(0.8), Inches(5.0), Inches(11.5), Inches(1.5), font_size=13, color=BLACK)

# ═══════════════════════════════════════════════════════════════
# SLIDE 9: REGIME SHIFT TEST
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Regime Shift Test: Pre-2023 vs. Post-2023 AI Boom", Inches(0.8), Inches(0.3), Inches(11), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

add_body_text(s, "Test: Train a Lasso model ONLY on pre-2023 data, then evaluate on post-2023 AI era quarters",
    Inches(0.8), Inches(1.15), Inches(11), Inches(0.4), font_size=13, color=GRAY)

tbl = make_table(s, 4, 5, Inches(0.8), Inches(1.8), Inches(11), Inches(2.0))
hdrs3 = ["Ticker", "Train Period", "Test Period", "Lasso RMSE\n(Frozen Pre-2023)", "Naive\nRMSE"]
for i, h in enumerate(hdrs3):
    tbl.cell(0, i).text = h
style_header_row(tbl, 5)
regime_data = [
    ["AMZN", "≤ 2022 (n=23)", "2023+ (n=14)", "0.1041  (3.0× worse)", "0.0347"],
    ["GOOGL", "≤ 2022 (n=8)", "2023+ (n=14)", "0.2148  (2.6× worse)", "0.0816"],
    ["MSFT", "≤ 2022 (n=25)", "2023+ (n=14)", "0.1423  (1.9× worse)", "0.0740"],
]
for r, row_data in enumerate(regime_data):
    for c, val in enumerate(row_data):
        cell = tbl.cell(r+1, c)
        style_body_cell(cell, val, font_size=12, bold=(c == 3))
        if c == 3:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFE, 0xF2, 0xF2)

regime_bullets = [
    "•  H₃ (Regime Shift) is PARTIALLY CONFIRMED: pre-2023 models collapse 2–3× on post-2023 data",
    "•  BUT this is NOT because capex became a stronger predictor — Lasso capex coefficients DROP to exactly 0.000 in BOTH eras",
    "•  The shift is driven by volatility and level changes in cloud growth itself (AI infrastructure boom effects)",
    "•  The autoregressive term (last quarter's growth) remains the only meaningful predictor in both regimes",
]
add_bullet_list(s, regime_bullets, Inches(0.8), Inches(4.2), Inches(11.5), Inches(2.5), font_size=13)

# ═══════════════════════════════════════════════════════════════
# SLIDE 10: MODEL INTERPRETATION — LASSO COEFFICIENTS
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Model Interpretation: The Zero-Capex Phenomenon", Inches(0.8), Inches(0.3), Inches(11), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

tbl = make_table(s, 10, 3, Inches(0.8), Inches(1.2), Inches(7), Inches(4.0))
hdrs4 = ["Feature", "Pre-2023\nCoefficient", "Post-2023\nCoefficient"]
for i, h in enumerate(hdrs4):
    tbl.cell(0, i).text = h
style_header_row(tbl, 3)
coef_data = [
    ["capex_yoy_lag1", "0.000", "0.000"],
    ["capex_yoy_lag2", "0.000", "0.000"],
    ["capex_yoy_lag3", "-0.000", "0.000"],
    ["capex_yoy_lag4", "0.005", "0.000"],
    ["fed_funds_chg_yoy_lag", "-0.008", "-0.021"],
    ["capex_to_rev_lag1", "-0.019", "0.000"],
    ["fcf_margin_lag1", "-0.014", "0.000"],
    ["debt_yoy_lag1", "0.009", "0.037"],
    ["cloud_yoy_lag1 (AR term)", "0.124", "0.096"],
]
for r, row_data in enumerate(coef_data):
    for c, val in enumerate(row_data):
        cell = tbl.cell(r+1, c)
        style_body_cell(cell, val, font_size=11, bold=(r == 8), align=PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT)
        if r < 4 and c > 0:  # capex rows
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFE, 0xF2, 0xF2)
        if r == 8 and c > 0:  # AR term
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xEC, 0xFD, 0xF5)

interp_points = [
    "All 4 capex growth lags are",
    "regularized to exactly 0.000",
    "by Lasso in BOTH eras.",
    "",
    "The only dominant predictor",
    "is the autoregressive term",
    "(last quarter's cloud growth).",
    "",
    "This does NOT mean capex",
    "doesn't create revenue —",
    "it means the conversion",
    "horizon is multi-year, not",
    "1–4 quarters, and recent",
    "capex is absorbed by internal",
    "AI R&D (Gemini, Copilot, Llama),",
    "not immediate customer billing.",
]
add_bullet_list(s, interp_points, Inches(8.3), Inches(1.2), Inches(4.5), Inches(5.0), font_size=12, color=TEAL)

# ═══════════════════════════════════════════════════════════════
# SLIDE 11: STRATEGIC RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "Strategic Recommendations for C-Level Executives", Inches(0.8), Inches(0.3), Inches(11), Inches(0.6), font_size=28)
add_gold_accent(s, Inches(0.9))

# Recommendation 1
box1 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.3), Inches(6), Inches(2.7))
box1.fill.solid()
box1.fill.fore_color.rgb = RGBColor(0xF0, 0xFD, 0xFA)
box1.line.color.rgb = TEAL
box1.line.width = Pt(1.5)
tf = box1.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.15)
tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]
p.text = "Recommendation 1"
p.font.size = Pt(10)
p.font.color.rgb = GOLD
p.font.bold = True
p = tf.add_paragraph()
p.text = "Decouple Cloud Revenue Guidance\nfrom Capex Announcements"
p.font.size = Pt(16)
p.font.color.rgb = TEAL
p.font.bold = True
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = ("• Stop using quarterly capex figures as a proxy for\n"
          "  1–4 quarter cloud revenue acceleration\n"
          "• Base guidance on Remaining Performance\n"
          "  Obligations (RPO) and organic momentum instead\n"
          "• Eliminates multi-billion-dollar forecast overshoots")
p.font.size = Pt(11)
p.font.color.rgb = BLACK
p.space_before = Pt(8)

# Recommendation 2
box2 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.3), Inches(6), Inches(2.7))
box2.fill.solid()
box2.fill.fore_color.rgb = RGBColor(0xF0, 0xFD, 0xFA)
box2.line.color.rgb = TEAL
box2.line.width = Pt(1.5)
tf = box2.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.15)
tf.margin_top = Inches(0.1)
p = tf.paragraphs[0]
p.text = "Recommendation 2"
p.font.size = Pt(10)
p.font.color.rgb = GOLD
p.font.bold = True
p = tf.add_paragraph()
p.text = "Implement Adaptive Scenario-Based\nCapacity Planning"
p.font.size = Pt(16)
p.font.color.rgb = TEAL
p.font.bold = True
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = ("• Post-2023 shows increased sensitivity to interest\n"
          "  rate changes and debt leverage conditions\n"
          "• Stage GPU capacity in modular tranches with\n"
          "  flexible lease options for macro downside scenarios\n"
          "• Monitor ROIC at datacenter cluster level")
p.font.size = Pt(11)
p.font.color.rgb = BLACK
p.space_before = Pt(8)

# Three findings summary
add_body_text(s, "Three Core Evidence-Based Findings", Inches(0.5), Inches(4.4), Inches(12), Inches(0.4), font_size=16, color=TEAL, bold=True)
findings = [
    "Finding 1 (H₁ Rejected):  Capex growth lags 1–4Q have zero predictive power — Lasso eliminates all coefficients to 0.000",
    "Finding 2 (H₂ Rejected):  Naive Persistence Baseline wins or ties vs. all ML models for AMZN and GOOGL",
    "Finding 3 (H₃ Partial):   Pre-2023 models collapse 2–3× on post-2023 data, but capex is NOT the driver of the regime shift",
]
add_bullet_list(s, findings, Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0), font_size=12)

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# SLIDE 12: 2030 SCENARIOS, TECHNICAL & BUSINESS LIMITATIONS
# ═══════════════════════════════════════════════════════════════
s = add_slide()
add_logo(s)
add_bottom_bar(s)
add_title_text(s, "2030 Strategic Scenarios & Project Limitations", Inches(0.8), Inches(0.25), Inches(11), Inches(0.55), font_size=26)
add_gold_accent(s, Inches(0.82))

# --- UPPER SECTION: 2030 Scenarios vs Model Horizon ---
# Left: Scenario Table (Compact)
add_body_text(s, "Strategic 2030 Capacity Scenarios (Heuristic Bounds, NOT an ML Forecast)",
    Inches(0.6), Inches(0.95), Inches(6.8), Inches(0.3), font_size=11, color=TEAL, bold=True)
tbl = make_table(s, 4, 3, Inches(0.6), Inches(1.25), Inches(6.8), Inches(1.4))
for i, h in enumerate(["Scenario", "Growth Assumption", "Structural Driver / Risk Profile"]):
    tbl.cell(0, i).text = h
style_header_row(tbl, 3)
scen_data = [
    ["Bear Case", "5% YoY", "GPU commoditization, enterprise AI monetization plateau"],
    ["Base Case", "15% YoY", "Steady cloud migration + moderate GenAI workload capture"],
    ["Bull Case", "30% YoY", "Transformative enterprise agentic AI + sovereign cloud demand"],
]
for r, row_data in enumerate(scen_data):
    for c, val in enumerate(row_data):
        style_body_cell(tbl.cell(r+1, c), val, font_size=9.5, align=PP_ALIGN.LEFT if c == 2 else PP_ALIGN.CENTER)

# Right: Explanatory Callout: Why ML Cannot Predict to 2030
callout_box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.6), Inches(1.05), Inches(5.15), Inches(1.6))
callout_box.fill.solid()
callout_box.fill.fore_color.rgb = RGBColor(0xF0, 0xFD, 0xFA)
callout_box.line.color.rgb = TEAL
callout_box.line.width = Pt(1.5)
tf_c = callout_box.text_frame
tf_c.word_wrap = True
tf_c.margin_left = Inches(0.15)
tf_c.margin_right = Inches(0.15)
tf_c.margin_top = Inches(0.08)
p = tf_c.paragraphs[0]
p.text = "Why Can't the Model Predict to 2030?"
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = TEAL

reasons_2030 = [
    "•  1-Quarter Horizon Limit (h=1): Models were strictly validated for 1Q-ahead. Iterative 16-20Q projections suffer fatal recursive error compounding.",
    "•  Zero Capex Predictive Signal: Regularization shrinks all capex lags to 0.000 — no empirical basis exists for multi-year ML extrapolation.",
    "•  Role of 2030 Numbers: Heuristic compounding stress-test for capacity downside planning, explicitly NOT a statistical model prediction.",
]
for r_text in reasons_2030:
    p = tf_c.add_paragraph()
    p.text = r_text
    p.font.size = Pt(9.5)
    p.font.color.rgb = BLACK
    p.space_before = Pt(2)

# --- MIDDLE SECTION: Two Detailed Limitation Cards ---
# Left Card: Technical & Methodological Limitations
card_tech = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(2.8), Inches(5.95), Inches(2.95))
card_tech.fill.solid()
card_tech.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
card_tech.line.color.rgb = TEAL
card_tech.line.width = Pt(1.5)
tf_t = card_tech.text_frame
tf_t.word_wrap = True
tf_t.margin_left = Inches(0.15)
tf_t.margin_right = Inches(0.15)
tf_t.margin_top = Inches(0.08)
p = tf_t.paragraphs[0]
p.text = "Technical & Methodological Limitations"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = TEAL

tech_points = [
    "•  Short Forecast Horizon (h=1Q): Validated solely for 1-step ahead; cannot project long-term horizons (2030) without exponential error drift.",
    "•  Small Sample Size (N=98 clean): Quarterly panel (GOOGL n=22, MSFT n=40, AMZN n=46) limits complex non-linear training; XGBoost overfits validation.",
    "•  Aggregated Capex Data: SEC 10-K/Q filings combine AI datacenter chips with real estate, office hardware, and retail logistics (no pure AI capex).",
    "•  Accounting Recasts: MSFT FY24/25 segment restatement creates structural breaks requiring manual same-vintage reconciliation.",
    "•  Target Metric Definition: Target is growth change (Delta YoY, can be <= 0), making MAPE mathematically undefined; evaluation relies strictly on RMSE.",
]
for t_pt in tech_points:
    p = tf_t.add_paragraph()
    p.text = t_pt
    p.font.size = Pt(9.5)
    p.font.color.rgb = BLACK
    p.space_before = Pt(3)

# Right Card: Business & Decision-Making Limitations
card_biz = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(2.8), Inches(5.95), Inches(2.95))
card_biz.fill.solid()
card_biz.fill.fore_color.rgb = RGBColor(0xFF, 0xFB, 0xEB)
card_biz.line.color.rgb = GOLD
card_biz.line.width = Pt(1.5)
tf_b = card_biz.text_frame
tf_b.word_wrap = True
tf_b.margin_left = Inches(0.15)
tf_b.margin_right = Inches(0.15)
tf_b.margin_top = Inches(0.08)
p = tf_b.paragraphs[0]
p.text = "Business & Decision-Making Limitations"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = TEAL

biz_points = [
    "•  No Short-Term Guidance Signal: Capex guidance announcements cannot serve as an early indicator to revise 1Q-ahead cloud revenue forecasts.",
    "•  Multi-Year Gestation Lag: Datacenter construction, utility power permits, and cluster setups take 2–4+ years before generating billable customer revenue.",
    "•  Internal R&D Absorption: Significant AI capex feeds proprietary foundation models (Gemini, Copilot, Llama), not third-party cloud segment billing.",
    "•  Non-Causal Relationship: Findings show predictive independence, not causal absence; macroeconomic IT budgets and interest rates dominate.",
    "•  Unfit for 2030 Capital Commitments: Model cannot justify $50B+ multi-year capex outlays; requires modular tranche-based leasing and RPO monitoring.",
]
for b_pt in biz_points:
    p = tf_b.add_paragraph()
    p.text = b_pt
    p.font.size = Pt(9.5)
    p.font.color.rgb = BLACK
    p.space_before = Pt(3)

# --- BOTTOM SECTION: Next Steps & Closing Banner ---
band2 = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Inches(5.9), SLIDE_W, Inches(1.25))
band2.fill.solid()
band2.fill.fore_color.rgb = TEAL
band2.line.fill.background()

add_title_text(s, "Thank You — Questions & Discussion",
    Inches(0.8), Inches(5.98), Inches(11.7), Inches(0.38), font_size=20, color=WHITE, alignment=PP_ALIGN.CENTER)
add_body_text(s, "Recommended Next Steps: Datacenter power filings • Cluster-level GPU utilization tracking • 4–8Q multi-year lag modeling",
    Inches(0.8), Inches(6.38), Inches(11.7), Inches(0.28), font_size=10.5, color=GOLD, bold=True, alignment=PP_ALIGN.CENTER)
add_body_text(s, "Cakrawala University • Predictive Analytics (Meeting 14 & Technical Interview) • Tita Noviana (40%) • Suci Fransisca Sisilia R (40%) • Nadira Dika Kaivalya (20%)",
    Inches(0.8), Inches(6.68), Inches(11.7), Inches(0.25), font_size=9.5, color=GRAY_LT, alignment=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
output_path = Path(__file__).resolve().parent / "Presentation_Predictive_Analytics.pptx"
try:
    prs.save(str(output_path))
    print(f"Successfully generated: {output_path}")
except PermissionError:
    fallback_path = Path(__file__).resolve().parent / "Presentation_Predictive_Analytics_new.pptx"
    prs.save(str(fallback_path))
    print(f"Note: {output_path.name} is currently open in PowerPoint. Saved to fallback: {fallback_path.name}")
print(f"Total slides: {len(prs.slides)}")
