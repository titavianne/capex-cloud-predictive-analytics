"""
Generate a professional, LaTeX-styled PDF for the Contribution Statement
using ReportLab.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT

def generate_pdf():
    pdf_filename = "contribution_statement.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom LaTeX-like Typography Styles (Times-Roman based)
    header_title_style = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827")
    )

    header_sub_style = ParagraphStyle(
        'HeaderSub',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#374151")
    )

    sec_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#111827"),
        spaceBefore=12,
        spaceAfter=4
    )

    subsec_heading_style = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=8,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'BodyJustify',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=12.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1F2937")
    )

    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#111827")
    )

    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#374151")
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        leading=12,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1F2937"),
        leftIndent=15,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1F2937")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#111827")
    )

    sign_style = ParagraphStyle(
        'SignStyle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827")
    )

    sign_sub_style = ParagraphStyle(
        'SignSubStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563")
    )

    story = []

    # Title & Header
    story.append(Paragraph("GROUP PROJECT CONTRIBUTION STATEMENT", header_title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Course: Predictive Analytics &bull; Cakrawala University", header_sub_style))
    story.append(Paragraph("Academic Year 2026/2027", header_sub_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F2937"), spaceAfter=8))

    # Metadata Table
    meta_data = [
        [Paragraph("Project Title:", meta_label_style), Paragraph("Hyperscaler Capital Expenditure and Cloud Segment Revenue Growth: An Empirical Time-Series & Econometric Analysis", meta_val_style)],
        [Paragraph("Project Theme:", meta_label_style), Paragraph("Finance and Business Analytics", meta_val_style)],
        [Paragraph("Task Type:", meta_label_style), Paragraph("Time-Series Forecasting (One-Step-Ahead Walk-Forward Validation)", meta_val_style)],
        [Paragraph("Course Registry:", meta_label_style), Paragraph("s.id/listdatapred02", meta_val_style)],
        [Paragraph("Submission Date:", meta_label_style), Paragraph("19 October 2026 (20:00 WIB via RISE)", meta_val_style)]
    ]
    meta_table = Table(meta_data, colWidths=[100, 388])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#9CA3AF"), spaceAfter=8))

    # Section 1
    story.append(Paragraph("1. Group Membership & Shared Responsibility", sec_heading_style))
    story.append(Paragraph(
        "This project was planned, executed, and documented collaboratively. "
        "The analytical, technical coding, interpretation, and presentation responsibilities were distributed as follows: "
        "TITA NOVIANA (40%), SUCI FRANSISCA SISILIA R (40%), and NADIRA DIKA KAIVALYA (20%). "
        "Each member has developed a comprehensive, end-to-end understanding of the entire analytical workflow and is fully prepared "
        "to explain and defend the methodology, implementation choices, and findings during the technical interview.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Table of Members
    table_data = [
        [Paragraph("Full Name", table_header_style), Paragraph("Student ID", table_header_style), Paragraph("Primary Assigned Role", table_header_style), Paragraph("Share", table_header_style)],
        [Paragraph("TITA NOVIANA", table_cell_bold), Paragraph("24120500011", table_cell_style), Paragraph("Lead Data Engineering & Econometric Modeling", table_cell_style), Paragraph("40%", table_cell_style)],
        [Paragraph("SUCI FRANSISCA SISILIA R", table_cell_bold), Paragraph("24120500008", table_cell_style), Paragraph("Lead Machine Learning & Hyperparameter Tuning", table_cell_style), Paragraph("40%", table_cell_style)],
        [Paragraph("NADIRA DIKA KAIVALYA", table_cell_bold), Paragraph("24110500023", table_cell_style), Paragraph("Lead Business Translation, Regime Analysis & Governance", table_cell_style), Paragraph("20%", table_cell_style)]
    ]
    member_table = Table(table_data, colWidths=[140, 75, 218, 55])
    member_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
        ('ALIGN', (3,0), (3,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")])
    ]))
    story.append(member_table)
    story.append(Spacer(1, 10))

    # Section 2
    story.append(Paragraph("2. Detailed Allocation of Analytical, Coding, Interpretation & Presentation Roles", sec_heading_style))

    # Member 1
    story.append(Paragraph("2.1 TITA NOVIANA (Student ID: 24120500011) &mdash; Role: Lead Data Engineering & Econometrics", subsec_heading_style))
    m1_points = [
        "<b>Analytical:</b> Formulated core predictive questions (H1: capex transmission, H2: ML vs. baseline); established one-step-ahead expanding-window walk-forward validation logic; conducted Augmented Dickey-Fuller (ADF) stationarity tests and Cross-Correlation Function (CCF) lag diagnostics.",
        "<b>Coding:</b> Built automated ingestion pipelines for SEC EDGAR XBRL companyfacts API and FRED Federal Reserve macroeconomic series; engineered chronological lag structure (lags 1-4) with strict zero-leakage controls; programmed Naive Persistence Baseline and AR(1) ARIMA maximum-likelihood models.",
        "<b>Interpretation:</b> Evaluated baseline persistence dynamics versus autoregressive models; analyzed downward life-cycle drift in hyperscaler growth rates; quantified sample size constraints across quarterly reporting windows.",
        "<b>Presentation:</b> Delivered Pitch Sections 1 & 2 (Problem Context, CFO Capital Allocation Dilemma, & Zero-Leakage Architecture); prepared for Technical Interview on data auditing and stationarity."
    ]
    for pt in m1_points:
        story.append(Paragraph(f"&bull;&nbsp;&nbsp;{pt}", bullet_style))
    story.append(Spacer(1, 4))

    # Member 2
    story.append(Paragraph("2.2 SUCI FRANSISCA SISILIA R (Student ID: 24120500008) &mdash; Role: Lead Machine Learning & Tuning", subsec_heading_style))
    m2_points = [
        "<b>Analytical:</b> Designed systematic hyperparameter search spaces for the two strongest non-baseline model families (Ridge & XGBoost); established expanding-window TimeSeriesSplit protocols to prevent temporal contamination; evaluated shrinkage mechanics in ill-conditioned financial panels.",
        "<b>Coding:</b> Implemented regularized Ridge (L2) and Lasso (L1) models with dynamic path optimization; built shallow-tree constrained XGBoost Regressor (max_depth=2-3); generated master evaluation tables (RMSE, MAE, Directional Accuracy) and diagnostic visual residual plots.",
        "<b>Interpretation:</b> Executed the Default-versus-Tuned comparative performance audit required by the rubric; interpreted Lasso coefficient shrinkage to zero (0.000) for all lagged capex features; synthesized empirical evidence disproving multi-quarter capex predictive superiority.",
        "<b>Presentation:</b> Delivered Pitch Section 3 (Candidate Model Benchmark, Default vs. Tuned Trade-offs, & Non-Causal Feature Coefficients); prepared for Technical Interview questions on regularized ML and hyperparameter tuning."
    ]
    for pt in m2_points:
        story.append(Paragraph(f"&bull;&nbsp;&nbsp;{pt}", bullet_style))
    story.append(Spacer(1, 4))

    # Member 3
    story.append(Paragraph("2.3 NADIRA DIKA KAIVALYA (Student ID: 24110500023) &mdash; Role: Lead Business Translation & Regime Analysis", subsec_heading_style))
    m3_points = [
        "<b>Analytical:</b> Formulated the structural regime shift test (H3: pre-2023 cloud maturation vs. post-2023 Generative AI boom); audited Microsoft FY2024 segment reclassification (-17% revenue recast); designed the transparent 2030 compounding scenario exercise.",
        "<b>Coding:</b> Curated, cross-checked, and verified 121 quarterly cloud segment revenue disclosures directly from primary Form 8-K Exhibit 99.1 earnings releases; developed pre- vs. post-2023 regime evaluation routines; generated the 2030 fan chart visualization with widening uncertainty bands.",
        "<b>Interpretation:</b> Formulated the 3 Core Evidence-Based Findings and 2 Feasible Strategic Recommendations for Chief Financial Officers; analyzed server depreciation overhang (4-6 year schedules), internal AI R&D absorption, and margin compression risks.",
        "<b>Presentation:</b> Delivered Pitch Section 4 (Strategic C-Level Actionable Recommendations, Capital Budget Decoupling, & 2030 Scenarios); led group defense during stakeholder panel Q&A and technical interview."
    ]
    for pt in m3_points:
        story.append(Paragraph(f"&bull;&nbsp;&nbsp;{pt}", bullet_style))
    story.append(Spacer(1, 8))

    # Section 3: Academic Integrity & Signatures
    story.append(Paragraph("3. Academic Integrity and Individual Ownership Affirmation", sec_heading_style))
    story.append(Paragraph(
        "We hereby affirm that: (1) The submitted predictive analysis represents our original, collaborative work conducted specifically "
        "for the Predictive Analytics course at Cakrawala University; (2) All financial datasets and macroeconomic indicators are authentic, public, "
        "and verifiable against primary SEC EDGAR and Federal Reserve sources; and (3) The individual ownership is allocated as TITA NOVIANA (40%), "
        "SUCI FRANSISCA SISILIA R (40%), and NADIRA DIKA KAIVALYA (20%), with each member ready to independently explain and defend any part of the analytical report.",
        body_style
    ))
    story.append(Spacer(1, 24))

    # Signature Block
    sig_data = [
        [Paragraph("____________________________", sign_style), Paragraph("____________________________", sign_style), Paragraph("____________________________", sign_style)],
        [Paragraph("TITA NOVIANA", sign_style), Paragraph("SUCI FRANSISCA SISILIA R", sign_style), Paragraph("NADIRA DIKA KAIVALYA", sign_style)],
        [Paragraph("Student ID: 24120500011", sign_sub_style), Paragraph("Student ID: 24120500008", sign_sub_style), Paragraph("Student ID: 24110500023", sign_sub_style)],
        [Paragraph("Date: 19 October 2026", sign_sub_style), Paragraph("Date: 19 October 2026", sign_sub_style), Paragraph("Date: 19 October 2026", sign_sub_style)]
    ]
    sig_table = Table(sig_data, colWidths=[162, 162, 164])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(KeepTogether(sig_table))

    doc.build(story)
    print(f"Successfully generated {pdf_filename}")

if __name__ == "__main__":
    generate_pdf()
