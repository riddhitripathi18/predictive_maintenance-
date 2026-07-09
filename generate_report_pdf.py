import os
import datetime
import math
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────────────────────
# 1. PAGE NUMBERING & DECORATIONS CANVAS
# ─────────────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to calculate total page count dynamically.
    Draws professional header and footer decorations on all pages except the cover page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Suppress headers/footers on the cover page
            return
        
        self.saveState()
        
        # ─── HEADER ───
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(HexColor("#1E293B")) # Slate 800
        self.drawString(54, 745, "SYS-CONTROL PREDICTIVE MAINTENANCE TECHNICAL REPORT")
        self.setFont("Helvetica", 8)
        self.setFillColor(HexColor("#64748B")) # Slate 500
        self.drawRightString(558, 745, "POC ANALYSIS & BLUEPRINT")
        
        # Header separator line
        self.setStrokeColor(HexColor("#E2E8F0")) # Slate 200
        self.setLineWidth(0.75)
        self.line(54, 737, 558, 737)
        
        # ─── FOOTER ───
        # Footer separator line
        self.line(54, 55, 558, 55)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(HexColor("#64748B"))
        self.drawString(54, 42, "CONFIDENTIAL - SYS-CONTROL Internal Technical Documentation")
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 42, page_text)
        
        self.restoreState()

# ─────────────────────────────────────────────────────────────
# 2. MAIN PDF GENERATOR FUNCTION
# ─────────────────────────────────────────────────────────────
def build_report():
    pdf_filename = "Predictive_Maintenance_POC_Report.pdf"
    
    # Page dimensions: letter = 612 x 792 points
    # Printable width: 612 - 2 * 54 = 504 points
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # ─── COLOR PALETTE ───
    c_primary = HexColor("#0F172A")    # Deep Slate
    c_secondary = HexColor("#0D9488")  # Teal Accent
    c_dark = HexColor("#1E293B")       # Muted Slate
    c_text = HexColor("#334155")       # Charcoal Body Text
    c_bg_light = HexColor("#F8FAFC")   # Slate 50
    c_border = HexColor("#E2E8F0")     # Slate 200
    
    # ─── CUSTOM STYLES ───
    styles.add(ParagraphStyle(
        name='CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=26,
        leading=32,
        textColor=c_primary,
        alignment=0, # Left-aligned
        spaceAfter=10
    ))
    
    styles.add(ParagraphStyle(
        name='CoverSubtitle',
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        alignment=0,
        spaceAfter=15
    ))

    styles.add(ParagraphStyle(
        name='ReportHeading1',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='ReportHeading2',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_secondary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='ReportBody',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_text,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name='ReportBodyBold',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_text,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name='CodeSnippet',
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=HexColor("#0F172A"),
        backColor=HexColor("#F1F5F9"),
        borderColor=HexColor("#E2E8F0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=5,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name='TableText',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=c_text
    ))

    styles.add(ParagraphStyle(
        name='TableHeaderText',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    ))

    styles.add(ParagraphStyle(
        name='CalloutText',
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=HexColor("#0F766E") # Deep Teal
    ))

    story = []

    # ─────────────────────────────────────────────────────────────
    # COVER PAGE
    # ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 40))
    story.append(Paragraph("SYS-CONTROL", styles['CoverSubtitle']))
    
    # Main Title
    story.append(Paragraph("PREDICTIVE MAINTENANCE &<br/>THERMAL EQUIPMENT ANALYTICS", styles['CoverTitle']))
    
    # Subtitle
    story.append(Paragraph("Proof of Concept (POC) Architecture, Model Calibration, and Technical Specification Report", styles['CoverSubtitle']))
    
    # Teal Thick Rule
    story.append(HRFlowable(width="100%", thickness=4, color=c_secondary, spaceBefore=5, spaceAfter=25))
    
    # Executive Summary brief on Cover Page
    story.append(Paragraph(
        "<b>Document Purpose:</b> This technical specification details the architecture, design choices, physical and "
        "statistical methodologies, and evaluation outcomes of the SYS-CONTROL Predictive Maintenance system. It outlines "
        "the machine learning classifiers trained for mechanical failures (rotary machinery) and the first-principles "
        "thermodynamic calculations implemented for static equipment (heat exchangers).", 
        styles['ReportBody']
    ))
    
    story.append(Spacer(1, 180))
    
    # Metadata block at the bottom
    metadata_data = [
        [Paragraph("<b>Document Version:</b>", styles['TableText']), Paragraph("1.0.0 (Production Blueprint)", styles['TableText'])],
        [Paragraph("<b>Status:</b>", styles['TableText']), Paragraph("Approved POC / Technical Reference", styles['TableText'])],
        [Paragraph("<b>Published Date:</b>", styles['TableText']), Paragraph(datetime.date.today().strftime("%B %d, %Y"), styles['TableText'])],
        [Paragraph("<b>Target Systems:</b>", styles['TableText']), Paragraph("Rotary Fleet (Type L, M, H) & Shell-and-Tube Heat Exchangers", styles['TableText'])],
        [Paragraph("<b>Lead Architects:</b>", styles['TableText']), Paragraph("Antigravity AI & SYS-CONTROL Systems Engineering Group", styles['TableText'])]
    ]
    meta_table = Table(metadata_data, colWidths=[120, 384])
    meta_table.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, HexColor("#F1F5F9")),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # SECTION 1: EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", styles['ReportHeading1']))
    story.append(Paragraph(
        "This project establishes a unified predictive maintenance framework that combines machine learning classifiers "
        "with first-principles thermodynamic analytics. By leveraging the industry-standard AI4I 2020 Predictive "
        "Maintenance dataset and classic heat transfer relations, the system is split into two operating domains:",
        styles['ReportBody']
    ))
    
    summary_bullets = [
        "<b>Domain A - Rotary Machinery (ML-Based):</b> Focuses on predicting sudden mechanical failures in high, medium, "
        "and low-cost production tools. Using advanced machine learning (XGBoost and Random Forest), the system predicts "
        "overall failure probability and isolates 5 specific failure modes (Tool Wear, Heat Dissipation, Power, "
        "Overstrain, and Random Failures) to recommend direct engineering interventions.",
        
        "<b>Domain B - Heat Exchangers (Physics-Based):</b> Evaluates thermal degradation and fouling resistance "
        "using thermodynamic relationships. Using an Effectiveness-NTU approach, it rates the health of static thermal "
        "equipment, ranks them by criticality, and schedules maintenance before units reach warning or critical limits.",
        
        "<b>Operational Flexibility:</b> Models are calibrated with Platt Scaling and support dual operating profiles: "
        "a <i>Balanced Mode</i> which maximizes the F1-Score (operational efficiency) and a <i>Safety-First Mode</i> "
        "which guarantees catching at least 90% of failures (91.18% empirical recall) to protect high-consequence assets."
    ]
    for bullet in summary_bullets:
        story.append(Paragraph(f"• {bullet}", styles['ReportBody']))
        story.append(Spacer(1, 2))
    
    story.append(Spacer(1, 10))
    
    # Callout Box
    callout_data = [[
        Paragraph(
            "<b>Key Outcome:</b> The primary XGBoost classifier achieved a high Area Under the ROC Curve (ROC-AUC) of "
            "<b>0.9776</b>. Through decision threshold calibration, the system caught <b>91.18%</b> of failures in Safety-First "
            "mode, offering a robust safeguard against unscheduled plant downtime.", 
            styles['CalloutText']
        )
    ]]
    callout_table = Table(callout_data, colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#F0FDFA")), # Light Teal
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(callout_table)
    
    # ─────────────────────────────────────────────────────────────
    # SECTION 2: PROBLEM STATEMENT & DOMAIN CONTEXT
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("2. Problem Statement & Domain Context", styles['ReportHeading1']))
    story.append(Paragraph(
        "Industrial plants lose millions annually due to unscheduled maintenance. Traditional reactive 'run-to-failure' "
        "strategies are costly, while conservative time-based preventative maintenance schedules replace expensive parts "
        "too early, leading to wasted useful life. This POC solves these problems for two equipment paradigms:",
        styles['ReportBody']
    ))
    
    story.append(Paragraph("2.1 Rotary Equipment & Mechanical Failure Modes", styles['ReportHeading2']))
    story.append(Paragraph(
        "The system monitors physical variables (Air/Process Temperature, Rotational Speed, Torque, Tool Wear) "
        "of production machines. There are 5 distinct failure mechanisms modeled:",
        styles['ReportBody']
    ))
    
    failures_bullets = [
        "<b>Tool Wear Failure (TWF):</b> Tool wear exceeds physical limits (between 200 and 240 min), causing mechanical failure.",
        "<b>Heat Dissipation Failure (HDF):</b> Occurs if the delta between process and air temperature is too narrow (< 8.6 K) "
        "while rotational speed is low (< 1380 rpm), preventing effective thermal dissipation.",
        "<b>Power Failure (PWF):</b> The actual mechanical power (Torque * Speed) drops below 3500 W or exceeds 9000 W, "
        "causing motor stalls or structural overloading.",
        "<b>Overstrain Failure (OSF):</b> Combined stress of high torque and tool wear. The product of torque and wear "
        "exceeds standard structural thresholds (e.g., > 11,000 Nm·min for Type L, > 12,000 for Type M, > 13,000 for Type H).",
        "<b>Random Failures (RNF):</b> Stochastic process anomalies not fully explained by physical features, occurring "
        "with an empirical probability of 0.1%."
    ]
    for bullet in failures_bullets:
        story.append(Paragraph(f"• {bullet}", styles['ReportBody']))
        story.append(Spacer(1, 1))

    story.append(Paragraph("2.2 Static Thermal Equipment & Fouling", styles['ReportHeading2']))
    story.append(Paragraph(
        "Heat exchangers transfer energy between hot and cold fluid streams. Over time, particulate accumulation, "
        "scaling, and chemical deposition create a solid layer of fouling resistance (R<sub>f</sub>). This fouling "
        "degrades the overall heat transfer coefficient (U) and reduces thermal effectiveness. "
        "Unmonitored fouling leads to thermal pinch bottlenecks, forcing premature plant shutdowns for manual cleaning.",
        styles['ReportBody']
    ))
    
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # SECTION 3: SYSTEM ARCHITECTURE
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("3. System Architecture", styles['ReportHeading1']))
    story.append(Paragraph(
        "The SYS-CONTROL system is designed as a decoupled, modern multi-tier application. The architecture separates "
        "computation-heavy machine learning inference and thermal calculations from the user interface.",
        styles['ReportBody']
    ))
    
    # Architecture Table / Map
    arch_data = [
        [Paragraph("<b>Component / Layer</b>", styles['TableHeaderText']), Paragraph("<b>Technology / Implementation</b>", styles['TableHeaderText']), Paragraph("<b>Description / Role</b>", styles['TableHeaderText'])],
        [
            Paragraph("<b>Frontend Dashboard</b>", styles['TableText']),
            Paragraph("React, Vite, CSS, Chart.js / Plotly", styles['TableText']),
            Paragraph("An operator console displaying real-time equipment statuses, simulation controls, degradation forecasts, and historical csv upload panels.", styles['TableText'])
        ],
        [
            Paragraph("<b>Backend REST API</b>", styles['TableText']),
            Paragraph("FastAPI, Uvicorn, Pydantic", styles['TableText']),
            Paragraph("Exposes routes in main.py for model inference, time-to-failure calculations, and heat exchanger thermal audits. Stores registry database in JSON.", styles['TableText'])
        ],
        [
            Paragraph("<b>ML Inference Pipeline</b>", styles['TableText']),
            Paragraph("XGBoost, Scikit-Learn, Pickle", styles['TableText']),
            Paragraph("Loads serialized XGBoost and RF models. Transforms raw features dynamically and predicts probability scores.", styles['TableText'])
        ],
        [
            Paragraph("<b>TTF Forecasting</b>", styles['TableText']),
            Paragraph("Python, NumPy, ttf_engine.py", styles['TableText']),
            Paragraph("Runs an iterative forward simulation using drift rates to forecast remaining useful life (RUL) and suggest interventions.", styles['TableText'])
        ],
        [
            Paragraph("<b>Static Thermal Analytics</b>", styles['TableText']),
            Paragraph("Python, heat_exchanger_analyzer.py", styles['TableText']),
            Paragraph("Implements first-principles thermodynamics (NTU-Effectiveness) to rate heat exchanger health and rank units.", styles['TableText'])
        ]
    ]
    arch_table = Table(arch_data, colWidths=[110, 140, 254])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
    ]))
    story.append(arch_table)
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("3.1 Key Code Files in the Repository", styles['ReportHeading2']))
    
    files_info = [
        ("main.py", "FastAPI app. Initializes, loads models and scalers, exposes routes like /predict, /simulate-ttf, and /hx/fleet-status."),
        ("1_train_model.py", "Primary ML training pipeline. Preprocesses training sets, performs feature engineering, applies SMOTE, trains classifiers, calibrates thresholds, and writes evaluations."),
        ("ttf_engine.py", "RUL simulator. Predicts the month of failure by forward-stepping parameters and checks model predictions under normal vs. intervened states."),
        ("heat_exchanger_analyzer.py", "Thermodynamic heat transfer model. Evaluates fouling resistance, effectiveness, and calculates estimated days until cleaning is required."),
        ("machine_registry.py / machine_registry.json", "Equipment database management layer. Houses registration schema, metadata, and operational parameters for all tracked physical units.")
    ]
    for filename, desc in files_info:
        story.append(Paragraph(f"• <b>{filename}:</b> {desc}", styles['ReportBody']))
        story.append(Spacer(1, 1))

    # ─────────────────────────────────────────────────────────────
    # SECTION 4: MACHINE FAILURE PREDICTIVE MODELING (ML PIPELINE)
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Machine Failure Predictive Modeling (ML Pipeline)", styles['ReportHeading1']))
    story.append(Paragraph(
        "Machine learning models are trained on the AI4I 2020 dataset (10,000 original rows). Due to the heavy imbalance of "
        "the dataset (~3.39% raw failure rate), specific feature engineering and sampling steps were required.",
        styles['ReportBody']
    ))
    
    story.append(Paragraph("4.1 Feature Engineering (Physics-Informed)", styles['ReportHeading2']))
    story.append(Paragraph(
        "Raw physical measurements are transformed into higher-order features to capture thermodynamic and mechanical stress relations:",
        styles['ReportBody']
    ))
    
    story.append(Paragraph(
        "1. <b>Temperature Delta (temp_delta)</b>: Tracks heat accumulation. Process Temp minus Air Temp [K].<br/>"
        "2. <b>Mechanical Power (power_W)</b>: Formulated from torque and speed: <i>Power = Torque * Speed * (2 * pi / 60)</i> [Watts].<br/>"
        "3. <b>Torque-Wear Interaction (torque_x_wear)</b>: Represents cumulative stress on the cutting tools: <i>Torque * Tool Wear</i>.<br/>"
        "4. <b>Tool Wear Percentage (wear_pct)</b>: Ratio of current tool wear to maximum tool wear (253.0 min).<br/>"
        "5. <b>Stress Indicators (high_torque, high_wear)</b>: Binary indicators for torque exceeding 55 Nm or wear exceeding 200 min.",
        styles['ReportBody']
    ))
    
    story.append(Paragraph("4.2 Handling Severe Class Imbalance", styles['ReportHeading2']))
    story.append(Paragraph(
        "To prevent classifiers from ignoring the minority class (failures) and predicting 'no failure' universally, "
        "the training pipeline applies two techniques:",
        styles['ReportBody']
    ))
    
    imbalance_bullets = [
        "<b>SMOTE (Synthetic Minority Over-sampling Technique):</b> Synthesizes new minority instances (failures) in "
        "the training set using k-nearest neighbors (k=5). This balances the training target (y_train_sm) to a 50/50 ratio, "
        "allowing the model to learn the decision boundaries of failure states effectively.",
        
        "<b>Scale Position Weight (scale_pos_weight):</b> For individual failure mode sub-classifiers (TWF, HDF, PWF, OSF, RNF), "
        "SMOTE is not applied. Instead, a dynamic class weight multiplier is calculated as <i>n_neg / n_pos</i>. This penalizes "
        "false negatives on specific failure modes severely during training."
    ]
    for bullet in imbalance_bullets:
        story.append(Paragraph(f"• {bullet}", styles['ReportBody']))
        story.append(Spacer(1, 1))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # SECTION 4.3 (Continuation) Calibration Explanation
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("4.3 Model Probability Calibration (Platt Scaling)", styles['ReportHeading2']))
    story.append(Paragraph(
        "Standard tree-based models, especially when trained with synthetic oversampling (SMOTE) or high scale_pos_weight, "
        "output distorted, non-calibrated probability estimates. While the ranking of risks remains correct, the absolute "
        "value of the output (e.g. 0.82) does not represent the true likelihood of failure. "
        "To correct this, the production XGBoost model uses <b>Platt Scaling</b> via Scikit-Learn's <i>CalibratedClassifierCV</i> "
        "with a 5-fold cross-validation loop.",
        styles['ReportBody']
    ))
    
    story.append(Paragraph(
        "Calibration maps raw classification scores into true empirical probability estimates. If the calibrated model predicts "
        "an 80% chance of failure, it means that empirically, 80 out of 100 machines with those exact physical indicators will fail. "
        "This calibration is crucial for operators when defining alarm thresholds.",
        styles['ReportBody']
    ))
    
    # ─────────────────────────────────────────────────────────────
    # SECTION 5: MODEL PERFORMANCE & CALIBRATION
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Model Performance & Calibration", styles['ReportHeading1']))
    story.append(Paragraph(
        "The models were evaluated on a hold-out test set (ai4i2020_test_real.csv, 2,000 rows) with zero data leakage. "
        "We compared XGBoost and Random Forest. Both models were calibrated and evaluated across two operating profiles:",
        styles['ReportBody']
    ))
    
    story.append(Paragraph(
        "1. <b>Balanced Mode</b>: Threshold calibrated to maximize F1-score. Suitable for routine operations.<br/>"
        "2. <b>Safety-First Mode</b>: Threshold calibrated to guarantee a minimum 90% Recall, catching critical anomalies "
        "at the expense of a higher false-alarm rate.",
        styles['ReportBody']
    ))
    
    # Evaluation Table
    eval_data = [
        [
            Paragraph("<b>Model & Profile</b>", styles['TableHeaderText']),
            Paragraph("<b>Thresh.</b>", styles['TableHeaderText']),
            Paragraph("<b>F1-Score</b>", styles['TableHeaderText']),
            Paragraph("<b>Recall (RUL)</b>", styles['TableHeaderText']),
            Paragraph("<b>Precision</b>", styles['TableHeaderText']),
            Paragraph("<b>Conf. Matrix (TN, FP, FN, TP)</b>", styles['TableHeaderText'])
        ],
        [
            Paragraph("XGBoost (Balanced)", styles['TableText']),
            Paragraph("0.82", styles['TableText']),
            Paragraph("0.8281", styles['TableText']),
            Paragraph("77.94% (53/68)", styles['TableText']),
            Paragraph("88.33%", styles['TableText']),
            Paragraph("[[1925, 7], [15, 53]]", styles['TableText'])
        ],
        [
            Paragraph("XGBoost (Safety-First)", styles['TableText']),
            Paragraph("0.08", styles['TableText']),
            Paragraph("0.5210", styles['TableText']),
            Paragraph("91.18% (62/68)", styles['TableText']),
            Paragraph("36.47%", styles['TableText']),
            Paragraph("[[1824, 108], [6, 62]]", styles['TableText'])
        ],
        [
            Paragraph("Random Forest (Balanced)", styles['TableText']),
            Paragraph("0.74", styles['TableText']),
            Paragraph("0.7705", styles['TableText']),
            Paragraph("69.12% (47/68)", styles['TableText']),
            Paragraph("87.04%", styles['TableText']),
            Paragraph("[[1925, 7], [21, 47]]", styles['TableText'])
        ],
        [
            Paragraph("Random Forest (Safety-First)", styles['TableText']),
            Paragraph("0.25", styles['TableText']),
            Paragraph("0.5536", styles['TableText']),
            Paragraph("91.18% (62/68)", styles['TableText']),
            Paragraph("39.74%", styles['TableText']),
            Paragraph("[[1838, 94], [6, 62]]", styles['TableText'])
        ],
    ]
    
    # 504 total width
    eval_table = Table(eval_data, colWidths=[130, 44, 55, 80, 55, 140])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
    ]))
    story.append(eval_table)
    
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("5.1 Model Selection & Rationale", styles['ReportHeading2']))
    story.append(Paragraph(
        "<b>XGBoost (ROC-AUC: 0.9776)</b> is selected as the primary online inference model over <b>Random Forest (ROC-AUC: 0.9712)</b> "
        "due to superior metrics in Balanced Mode (F1-score of 0.8281 vs 0.7705, and Recall of 77.94% vs 69.12%). "
        "XGBoost's gradient boosting algorithm captures complex non-linear combinations of rotational speed, torque, "
        "and tool wear more effectively. Furthermore, in Safety-First mode, setting the decision threshold to 0.08 allows "
        "the XGBoost model to achieve 91.18% recall while limiting false alarms (108 false alarms out of 2000 total samples).",
        styles['ReportBody']
    ))
    
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # SECTION 6: TIME-TO-FAILURE (TTF) FORECASTING
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("6. Time-to-Failure (TTF) Forecasting Engine", styles['ReportHeading1']))
    story.append(Paragraph(
        "The TTF Engine (implemented in ttf_engine.py) provides actionable maintenance schedules by projecting "
        "sensor parameters month-by-month using linear operational degradation drift rates.",
        styles['ReportBody']
    ))
    
    story.append(Paragraph("6.1 Default Degradation Drift Rates", styles['ReportHeading2']))
    story.append(Paragraph(
        "Based on fleet averages, the engine applies the following parameter changes per operating month:",
        styles['ReportBody']
    ))
    
    drift_data = [
        [Paragraph("<b>Parameter</b>", styles['TableHeaderText']), Paragraph("<b>Monthly Drift Rate</b>", styles['TableHeaderText']), Paragraph("<b>Physical Interpretation</b>", styles['TableHeaderText'])],
        [Paragraph("Tool Wear [min]", styles['TableText']), Paragraph("+12.0 min", styles['TableText']), Paragraph("Normal cutting tool material degradation.", styles['TableText'])],
        [Paragraph("Rotational Speed [rpm]", styles['TableText']), Paragraph("-25.0 rpm", styles['TableText']), Paragraph("Motor efficiency loss and friction increase.", styles['TableText'])],
        [Paragraph("Torque [Nm]", styles['TableText']), Paragraph("+1.2 Nm", styles['TableText']), Paragraph("Increased mechanical resistance due to wear.", styles['TableText'])],
        [Paragraph("Air Temperature [K]", styles['TableText']), Paragraph("+0.05 K", styles['TableText']), Paragraph("Ambient seasonal drift approximation.", styles['TableText'])],
        [Paragraph("Process Temperature [K]", styles['TableText']), Paragraph("+0.10 K", styles['TableText']), Paragraph("Internal operating friction heat accumulation.", styles['TableText'])],
    ]
    drift_table = Table(drift_data, colWidths=[130, 110, 264])
    drift_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
    ]))
    story.append(drift_table)
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("6.2 Forecasting Loop & Operational Interventions", styles['ReportHeading2']))
    story.append(Paragraph(
        "At each simulation step (Month <i>m</i>), the engine calculates physical engineered features (such as <i>power_W</i> and "
        "<i>torque_x_wear</i>) and evaluates the probability of failure using the calibrated XGBoost model. "
        "If the probability crosses the active threshold, the engine logs a failure, identifies the primary failure mode (e.g. HDF, OSF), "
        "and simulates recommended interventions. The engine tests three intervention actions:",
        styles['ReportBody']
    ))
    
    interventions_bullets = [
        "<b>Part Replacement:</b> Resets the tool wear parameter to 0 min, extending the TTF substantially.",
        "<b>Load Reduction:</b> Reduces operating torque by 15% and rotational speed by 10% to lower physical strain on the machine.",
        "<b>Ambient Cooling:</b> Simulates the installation of external cooling systems, lowering ambient air temperature by 5 K, "
        "which resolves Heat Dissipation (HDF) risks."
    ]
    for bullet in interventions_bullets:
        story.append(Paragraph(f"• {bullet}", styles['ReportBody']))
        story.append(Spacer(1, 1))

    # ─────────────────────────────────────────────────────────────
    # SECTION 7: HEAT EXCHANGER THERMAL ANALYTICS (FIRST-PRINCIPLES)
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("7. Heat Exchanger Thermal Analytics (First-Principles)", styles['ReportHeading1']))
    story.append(Paragraph(
        "Unlike rotary machinery, static heat exchangers do not require machine learning. Instead, they are monitored using "
        "first-principles thermodynamics based on the <b>Effectiveness-NTU (Number of Transfer Units)</b> method.",
        styles['ReportBody']
    ))
    
    story.append(Paragraph("7.1 Mathematical Model & Equations", styles['ReportHeading2']))
    story.append(Paragraph(
        "The system evaluates the thermal performance using the following formulation:",
        styles['ReportBody']
    ))
    
    equations = (
        "<b>1. Heat Capacity Rates:</b><br/>"
        "    C_shell = mass_flow_shell * Cp_shell  |  C_tube = mass_flow_tube * Cp_tube<br/>"
        "    C_min = min(C_shell, C_tube) | C_max = max(C_shell, C_tube)<br/>"
        "    C_ratio = C_min / C_max<br/><br/>"
        "<b>2. Actual Heat Duty (Q_actual):</b><br/>"
        "    Q_actual = mass_flow_shell * Cp_shell * (Temp_shell_in - Temp_shell_out)  [kW]<br/><br/>"
        "<b>3. Maximum Theoretical Heat Duty (Q_max):</b><br/>"
        "    Q_max = C_min * (Temp_hot_in - Temp_cold_in)  [kW]<br/><br/>"
        "<b>4. Thermal Effectiveness (epsilon):</b><br/>"
        "    epsilon = Q_actual / Q_max<br/><br/>"
        "<b>5. Number of Transfer Units (NTU) & Heat Transfer Coefficient (U):</b><br/>"
        "    Using the counter-flow relation (based on configuration):<br/>"
        "    NTU = (1 / (C_ratio - 1)) * ln((epsilon - 1) / (epsilon * C_ratio - 1))<br/>"
        "    U_actual = NTU * C_min / Area  [W/m^2 K]<br/><br/>"
        "<b>6. Thermal Fouling Resistance (R_f):</b><br/>"
        "    R_f = (1 / U_actual) - (1 / U_design)  [m^2 K/W]"
    )
    story.append(Paragraph(equations, styles['CodeSnippet']))
    
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────
    # SECTION 7.2 (Continuation) Fouling Limits
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("7.2 Industry Standard Fouling Limits (TEMA)", styles['ReportHeading2']))
    story.append(Paragraph(
        "Fouling resistance (R<sub>f</sub>) threshold limits are established using standard Tubular Exchanger Manufacturers "
        "Association (TEMA) and Heat Exchanger Design Handbook (HEDH) parameters. When a unit's calculated fouling resistance "
        "exceeds these limits, or when thermal effectiveness drops, maintenance alerts are generated:",
        styles['ReportBody']
    ))
    
    fouling_data = [
        [Paragraph("<b>Fluid Type</b>", styles['TableHeaderText']), Paragraph("<b>Fouling Limit (R_f)</b>", styles['TableHeaderText']), Paragraph("<b>Typical Industrial Application</b>", styles['TableHeaderText'])],
        [Paragraph("Cooling Water", styles['TableText']), Paragraph("0.000176 m^2 K/W", styles['TableText']), Paragraph("Utility cooling loops, cooling towers.", styles['TableText'])],
        [Paragraph("Process Liquid", styles['TableText']), Paragraph("0.000352 m^2 K/W", styles['TableText']), Paragraph("Organic chemical streams, hydrocarbon fractions.", styles['TableText'])],
        [Paragraph("Crude Oil", styles['TableText']), Paragraph("0.000528 m^2 K/W", styles['TableText']), Paragraph("Refinery preheater trains (heavy deposition).", styles['TableText'])],
        [Paragraph("Steam", styles['TableText']), Paragraph("0.000088 m^2 K/W", styles['TableText']), Paragraph("Reboilers and utility steam heaters (condensation).", styles['TableText'])],
        [Paragraph("Refrigerant", styles['TableText']), Paragraph("0.000176 m^2 K/W", styles['TableText']), Paragraph("Chiller evaporators and condensers.", styles['TableText'])],
    ]
    fouling_table = Table(fouling_data, colWidths=[110, 110, 284])
    fouling_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
    ]))
    story.append(fouling_table)
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("7.3 Degradation Rating & Days-to-Clean Estimation", styles['ReportHeading2']))
    story.append(Paragraph(
        "Each heat exchanger is assigned a health rating based on design effectiveness degradation:<br/>"
        "• <b>Normal:</b> Effectiveness &gt;= 80% of design value.<br/>"
        "• <b>Warning:</b> Effectiveness drops below 80% of design, or R<sub>f</sub> exceeds the TEMA limit.<br/>"
        "• <b>Critical:</b> Effectiveness drops below 65% of design.<br/><br/>"
        "The engine calculates <b>Days-to-Clean</b> by dividing the remaining allowable fouling increment by a default linear accumulation rate "
        "of <i>5e-7 m^2 K/W per day</i>. This allows operators to schedule pre-emptive tube cleaning campaigns during planned turnarounds.",
        styles['ReportBody']
    ))
    
    # ─────────────────────────────────────────────────────────────
    # SECTION 8: TECHNICAL RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("8. Technical Recommendations & Deployment Plan", styles['ReportHeading1']))
    story.append(Paragraph(
        "Based on the results of the Proof of Concept, the following steps are recommended for production deployment:",
        styles['ReportBody']
    ))
    
    recs = [
        "<b>Model Deployment:</b> Deploy the calibrated XGBoost model as a microservice using the FastAPI backend. "
        "Use the calibrated threshold of 0.82 for general operations to minimize false alarms, and 0.08 for critical, "
        "unredundant rotary machinery to prevent catastrophic mechanical failure.",
        
        "<b>Edge Data Integration:</b> Connect the FastAPI backend to the plant's SCADA or historian system (e.g. OSIsoft PI) "
        "to feed sensor streams (temperatures, speed, torque) into the /predict endpoint at 5-minute intervals.",
        
        "<b>Continuous Retraining:</b> Establish a closed-loop system where operator feedback (confirming true/false alarms) "
        "is recorded in the registry and used to retrain the classifiers quarterly, improving model precision over time.",
        
        "<b>Thermal Monitoring:</b> Implement the first-principles Heat Exchanger analytics using automatic thermal data "
        "collected from inlet/outlet thermocouples and orifice plate flowmeters."
    ]
    for rec in recs:
        story.append(Paragraph(f"• {rec}", styles['ReportBody']))
        story.append(Spacer(1, 2))
        
    story.append(Spacer(1, 30))
    
    # Signature Block
    sig_data = [
        [Paragraph("<b>Prepared by:</b>", styles['TableText']), Paragraph("<b>Approved by:</b>", styles['TableText'])],
        [Paragraph("Antigravity AI Coding Assistant", styles['TableText']), Paragraph("SYS-CONTROL Systems Engineering Group", styles['TableText'])],
        [Paragraph("_____________________________", styles['TableText']), Paragraph("_____________________________", styles['TableText'])]
    ]
    sig_table = Table(sig_data, colWidths=[252, 252])
    sig_table.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(sig_table)

    # ─────────────────────────────────────────────────────────────
    # BUILD DOCUMENT
    # ─────────────────────────────────────────────────────────────
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {pdf_filename}")

if __name__ == "__main__":
    build_report()
