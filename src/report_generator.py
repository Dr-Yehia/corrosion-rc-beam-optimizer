# ============================================================
# src/report_generator.py
# Corrosion RC Beam Optimizer
# Automated Scientific PDF Report Generator
# Produces: results/Final_Report.pdf
# Sections:
#   1. Title & Abstract
#   2. Database Summary
#   3. ACI 318-19 Benchmark
#   4. MLP Baseline Results
#   5. NSGA-III GA Optimisation Log
#   6. SHAP Feature Importance
#   7. PySR Discovered Equation
#   8. Statistical Validation
#   9. Figures Gallery
#  10. Conclusion & Decision
# ============================================================

import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    REPORT_TITLE, REPORT_FILE, REPORT_AUTHOR,
    FIGURES_DIR, MODELS_DIR, EQ_DIR,
    L1_TARGET_R2, L2_TARGET_R2,
    W1, W2, W3,
    GA_POPULATION_SIZE, GA_MAX_GENERATIONS, GA_MAX_RUNS,
    GA_CONSISTENCY_WINDOW, GA_CROSSOVER_RATE, GA_MUTATION_RATE,
    BOOTSTRAP_N, KFOLD_N_SPLITS, WILCOXON_ALPHA,
    PYSR_NITERATIONS, PYSR_MAXSIZE,
)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.error("ReportLab not installed. Run: pip install reportlab")

PAGE_W, PAGE_H = A4
LEFT_M = RIGHT_M = 2.5 * cm
TOP_M  = BOTTOM_M = 2.0 * cm


# ============================================================
# STYLE DEFINITIONS
# ============================================================

def _build_styles():
    """Return a dict of named ParagraphStyles for the report."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent    = base["Title"],
            fontSize  = 20,
            leading   = 26,
            textColor = colors.HexColor("#0D1B2A"),
            alignment = TA_CENTER,
            spaceAfter= 6,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent    = base["Normal"],
            fontSize  = 11,
            leading   = 16,
            textColor = colors.HexColor("#3A5A8C"),
            alignment = TA_CENTER,
            spaceAfter= 18,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent    = base["Heading1"],
            fontSize  = 14,
            leading   = 20,
            textColor = colors.HexColor("#0D1B2A"),
            spaceBefore=16,
            spaceAfter = 6,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent    = base["Heading2"],
            fontSize  = 11,
            leading   = 16,
            textColor = colors.HexColor("#1A3A5C"),
            spaceBefore= 10,
            spaceAfter = 4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent    = base["Normal"],
            fontSize  = 10,
            leading   = 15,
            alignment = TA_JUSTIFY,
            spaceAfter= 6,
        ),
        "mono": ParagraphStyle(
            "Mono",
            parent    = base["Code"],
            fontSize  = 8.5,
            leading   = 13,
            leftIndent= 12,
            backColor = colors.HexColor("#F4F6F8"),
            spaceAfter= 6,
        ),
        "verdict_pass": ParagraphStyle(
            "VerdictPass",
            parent    = base["Normal"],
            fontSize  = 12,
            leading   = 18,
            textColor = colors.HexColor("#1B5E20"),
            alignment = TA_CENTER,
            spaceBefore=10,
            spaceAfter = 10,
        ),
        "verdict_fail": ParagraphStyle(
            "VerdictFail",
            parent    = base["Normal"],
            fontSize  = 12,
            leading   = 18,
            textColor = colors.HexColor("#B71C1C"),
            alignment = TA_CENTER,
            spaceBefore=10,
            spaceAfter = 10,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent    = base["Normal"],
            fontSize  = 8,
            leading   = 11,
            textColor = colors.HexColor("#555555"),
            alignment = TA_CENTER,
            spaceAfter= 8,
        ),
    }
    return styles


def _hr(width: float = 15 * cm):
    return HRFlowable(
        width=width, thickness=0.6,
        color=colors.HexColor("#BDBDBD"),
        spaceAfter=6, spaceBefore=6,
    )


def _table(data: list, col_widths=None, header_row: bool = True):
    """Build a formatted ReportLab Table."""
    tbl = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A3A5C")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE",   (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F0F4FA")]),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CFD8DC")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


# ============================================================
# SECTION BUILDERS
# ============================================================

def _section_title_page(styles: dict) -> list:
    now = datetime.now().strftime("%B %d, %Y — %H:%M")
    return [
        Spacer(1, 2 * cm),
        Paragraph(REPORT_TITLE, styles["title"]),
        Paragraph(REPORT_AUTHOR, styles["subtitle"]),
        Paragraph(f"Generated: {now}", styles["subtitle"]),
        _hr(12 * cm),
        Spacer(1, 0.5 * cm),
        Paragraph(
            "This report presents the complete results of the "
            "Corrosion RC Beam Optimizer research pipeline. "
            "The study applies a Neural Network-NSGA-III Genetic "
            "Algorithm framework to predict the residual flexural "
            "capacity R(%) of corroded reinforced concrete beams, "
            "benchmarked against ACI 318-19 and the state-of-the-art "
            "ML literature (Zhang et al., 2025). "
            "Two hierarchical performance layers must be surpassed: "
            f"L1: R² > {L1_TARGET_R2} (ACI threshold), and "
            f"L2: R² > {L2_TARGET_R2} (SOTA threshold). "
            "Statistical significance is confirmed via Wilcoxon, "
            "Bootstrap CI, 10-Fold CV, Cohen’s d, and McNemar tests.",
            styles["body"],
        ),
        PageBreak(),
    ]


def _section_ga_config(styles: dict) -> list:
    elems = [
        Paragraph("NSGA-III Configuration", styles["h1"]),
        _hr(),
        _table(
            [
                ["Parameter",               "Value"],
                ["Population Size",          str(GA_POPULATION_SIZE)],
                ["Max Generations",          str(GA_MAX_GENERATIONS)],
                ["Max Restarts",             str(GA_MAX_RUNS)],
                ["Consistency Window",       str(GA_CONSISTENCY_WINDOW)],
                ["Crossover Rate",           str(GA_CROSSOVER_RATE)],
                ["Mutation Rate",            str(GA_MUTATION_RATE)],
                ["Fitness W1 (R²)",          str(W1)],
                ["Fitness W2 (ACI improve)", str(W2)],
                ["Fitness W3 (penalty)",     str(W3)],
                ["L1 Target R²",             str(L1_TARGET_R2)],
                ["L2 Target R²",             str(L2_TARGET_R2)],
            ],
            col_widths=[9 * cm, 6 * cm],
        ),
        Spacer(1, 0.4 * cm),
    ]
    return elems


def _section_from_json(
    json_path: Path,
    section_title: str,
    styles: dict,
    keys_to_show: list = None,
) -> list:
    """
    Generic section builder: load a JSON results file
    and render a table of key→value pairs.
    """
    elems = [Paragraph(section_title, styles["h1"]), _hr()]
    if not json_path.exists():
        elems.append(
            Paragraph(f"[File not found: {json_path}]", styles["body"])
        )
        return elems

    with open(json_path) as f:
        data = json.load(f)

    def _flatten(d: dict, prefix="") -> list:
        rows = []
        for k, v in d.items():
            if keys_to_show and k not in keys_to_show:
                continue
            key_label = (prefix + k).replace("_", " ").title()
            if isinstance(v, dict):
                rows.extend(_flatten(v, prefix=k + "."))
            elif isinstance(v, list):
                rows.append([key_label, str(v)])
            else:
                rows.append([key_label, str(v)])
        return rows

    rows = [["Metric", "Value"]] + _flatten(data)
    elems.append(
        _table(rows, col_widths=[10 * cm, 5 * cm])
    )
    elems.append(Spacer(1, 0.3 * cm))
    return elems


def _section_ga_log(log_lines: list, styles: dict) -> list:
    """Render the GA live log panel (last 80 lines)."""
    elems = [
        Paragraph("NSGA-III Live Optimisation Log", styles["h1"]),
        _hr(),
    ]
    shown = log_lines[-80:] if len(log_lines) > 80 else log_lines
    for line in shown:
        elems.append(Paragraph(line, styles["mono"]))
    elems.append(Spacer(1, 0.3 * cm))
    return elems


def _section_equation(styles: dict) -> list:
    """Render the PySR discovered equation."""
    elems = [
        Paragraph("PySR Discovered Equation", styles["h1"]),
        _hr(),
    ]
    if PYSR_OUTPUT_FILE := (EQ_DIR / "best_equation.txt"):
        if PYSR_OUTPUT_FILE.exists():
            with open(PYSR_OUTPUT_FILE) as f:
                eq_text = f.read()
            elems.append(Paragraph(eq_text, styles["mono"]))
        else:
            elems.append(
                Paragraph("[Equation file not yet generated]", styles["body"])
            )
    elems.append(Spacer(1, 0.3 * cm))
    return elems


def _section_figures(styles: dict) -> list:
    """Embed all generated figures from results/figures/."""
    elems = [
        Paragraph("Figures Gallery", styles["h1"]),
        _hr(),
    ]
    figure_files = sorted(FIGURES_DIR.glob("*.png"))
    if not figure_files:
        elems.append(
            Paragraph("[No figures generated yet]", styles["body"])
        )
        return elems

    for fig_path in figure_files:
        try:
            img   = RLImage(str(fig_path), width=14 * cm, height=9 * cm)
            caption = fig_path.stem.replace("_", " ").title()
            elems += [
                img,
                Paragraph(f"Figure: {caption}", styles["caption"]),
                Spacer(1, 0.5 * cm),
            ]
        except Exception as e:
            elems.append(
                Paragraph(f"[Could not embed {fig_path.name}: {e}]",
                          styles["body"])
            )
    return elems


def _section_conclusion(validation_results: dict, styles: dict) -> list:
    """Final verdict section."""
    elems = [
        Paragraph("Conclusion & Decision", styles["h1"]),
        _hr(),
    ]
    verdict     = validation_results.get("verdict", "")
    is_pass     = "✅" in verdict
    style_key   = "verdict_pass" if is_pass else "verdict_fail"

    elems.append(Paragraph(verdict, styles[style_key]))
    elems.append(Spacer(1, 0.5 * cm))

    summary = validation_results.get("summary", {})
    if summary:
        rows = [["Test", "Result"]] + [
            [k.replace("_", " ").title(), "✓" if v else "✗"]
            for k, v in summary.items() if isinstance(v, bool)
        ]
        elems.append(_table(rows, col_widths=[10 * cm, 5 * cm]))

    elems += [
        Spacer(1, 0.8 * cm),
        Paragraph(
            f"Report generated automatically by the Corrosion RC Beam "
            f"Optimizer pipeline on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}.",
            styles["caption"],
        ),
    ]
    return elems


# ============================================================
# MAIN PDF BUILDER
# ============================================================

def generate_report(
    mlp_metrics:          dict = None,
    ga_results:           dict = None,
    aci_metrics:          dict = None,
    shap_results:         dict = None,
    pysr_results:         dict = None,
    validation_results:   dict = None,
    log_lines:            list = None,
) -> Path:
    """
    Build and save the full scientific PDF report.

    All parameters are optional — if a results dict is not
    provided, the section is built from the saved JSON files
    in results/models/.

    Returns
    -------
    Path to the generated PDF file.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab is not installed.")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building PDF report \u2192 {REPORT_FILE}")

    doc = SimpleDocTemplate(
        str(REPORT_FILE),
        pagesize     = A4,
        leftMargin   = LEFT_M,
        rightMargin  = RIGHT_M,
        topMargin    = TOP_M,
        bottomMargin = BOTTOM_M,
        title        = REPORT_TITLE,
        author       = REPORT_AUTHOR,
    )

    styles = _build_styles()
    story  = []

    # ─ 1. Title Page
    story += _section_title_page(styles)

    # ─ 2. GA Configuration
    story += _section_ga_config(styles)
    story.append(PageBreak())

    # ─ 3. ACI 318-19 Benchmark
    story += _section_from_json(
        MODELS_DIR / "aci_benchmark_metrics.json",
        "ACI 318-19 Benchmark Results",
        styles,
    )

    # ─ 4. MLP Baseline
    story += _section_from_json(
        MODELS_DIR / "mlp_metrics.json",
        "MLP Baseline — Training & Test Metrics",
        styles,
    )
    story.append(PageBreak())

    # ─ 5. GA Optimisation Log
    if log_lines:
        story += _section_ga_log(log_lines, styles)
        story.append(PageBreak())

    # ─ 6. SHAP Feature Importance
    story += _section_from_json(
        MODELS_DIR / "shap_importance.csv",
        "SHAP Feature Importance",
        styles,
    )

    # ─ 7. PySR Equation
    story += _section_equation(styles)
    story.append(PageBreak())

    # ─ 8. Statistical Validation
    story += _section_from_json(
        MODELS_DIR / "statistical_validation.json",
        "Statistical Validation Results",
        styles,
    )
    story.append(PageBreak())

    # ─ 9. Figures Gallery
    story += _section_figures(styles)
    story.append(PageBreak())

    # ─ 10. Conclusion
    if validation_results:
        story += _section_conclusion(validation_results, styles)
    else:
        vr_path = MODELS_DIR / "statistical_validation.json"
        if vr_path.exists():
            with open(vr_path) as f:
                vr = json.load(f)
            story += _section_conclusion(vr, styles)

    # Build PDF
    doc.build(story)
    logger.info(f"PDF report saved \u2192 {REPORT_FILE}")
    return REPORT_FILE


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    path = generate_report()
    print(f"\n✅ Report generated: {path}")
