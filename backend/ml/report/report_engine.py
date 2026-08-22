"""
BridgeGuardian AI — Report Engine
Compiles dynamic PDF reports using ReportLab and formats structured JSON packets.
Supports Inspection State Validation (FULL_ANALYSIS, PARTIAL_ANALYSIS, ALL_IMAGES_REJECTED).
No simulated demo-mode placeholder banners or fabricated numbers.
"""
from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class ReportEngine:
    def __init__(self, reports_dir: str = None) -> None:
        if reports_dir is None:
            from backend.core.config import get_settings
            reports_dir = get_settings().reports_dir
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(
        self,
        inspection_id: int,
        health_predictions: Dict[str, Any],
        aggregate_stats: Dict[str, Any],
        explainability: Dict[str, Any],
        maintenance: Dict[str, Any],
        image_results: List[Dict[str, Any]],
        model_metadata: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> str:
        """
        Generates an enterprise-grade PDF report.
        If accepted_images == 0, generates an Inspection Attempt Report (Status: Analysis Failed)
        with zero fabricated engineering metrics.
        """
        filename = f"inspection_report_{inspection_id}.pdf"
        filepath = self.reports_dir / filename
        
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        story = []
        styles = getSampleStyleSheet()
        
        c_primary = colors.HexColor("#1A365D")    # Deep navy
        c_secondary = colors.HexColor("#2B6CB0")  # Slate blue
        c_text = colors.HexColor("#2D3748")       # Charcoal
        c_light = colors.HexColor("#F7FAFC")      # Off-white
        c_border = colors.HexColor("#E2E8F0")     # Light gray
        c_danger = colors.HexColor("#C53030")     # Warning red
        c_warning = colors.HexColor("#DD6B20")    # Orange
        c_success = colors.HexColor("#2F855A")    # Green
        
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            textColor=c_primary,
            spaceAfter=6,
            alignment=0
        )
        
        h1_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading1"],
            fontSize=13,
            leading=16,
            textColor=c_primary,
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True
        )
        
        h2_style = ParagraphStyle(
            "SubSectionHeader",
            parent=styles["Heading2"],
            fontSize=10,
            leading=13,
            textColor=c_secondary,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=12,
            textColor=c_text
        )
        
        bold_body_style = ParagraphStyle(
            "ReportBodyBold",
            parent=body_style,
            fontName="Helvetica-Bold"
        )
        
        summary_style = ParagraphStyle(
            "ReportSummary",
            parent=styles["Italic"],
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1A202C")
        )
        
        alert_banner_style = ParagraphStyle(
            "AlertBanner",
            parent=body_style,
            fontName="Helvetica-Bold",
            textColor=c_danger,
            alignment=1
        )

        warning_banner_style = ParagraphStyle(
            "WarningBanner",
            parent=body_style,
            fontName="Helvetica-Bold",
            textColor=c_warning,
            alignment=1
        )

        valid_count = performance_metrics.get("accepted_images", 0)
        rejected_count = performance_metrics.get("rejected_images", 0)
        total_uploaded = valid_count + rejected_count
        pipeline_state = performance_metrics.get("pipeline_state", "FULL_ANALYSIS")

        # ---------------------------------------------------------------------
        # CASE 1: STATE 2 - ALL IMAGES REJECTED (INSPECTION FAILED)
        # ---------------------------------------------------------------------
        if valid_count == 0:
            # Header Title Block
            story.append(Paragraph("BridgeGuardian AI — Inspection Attempt Report", title_style))
            story.append(Paragraph(
                f"<b>Campaign Inspection ID:</b> CAMP-ID-{inspection_id} &nbsp;|&nbsp; "
                f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
                f"<b>Status:</b> <font color='#C53030'><b>Analysis Failed</b></font>",
                body_style
            ))
            story.append(Spacer(1, 10))

            # Red Failure Status Banner
            banner_data = [[Paragraph("❌ INSPECTION FAILED: No valid images passed quality validation. Predictions intentionally skipped to avoid misleading engineering decisions.", alert_banner_style)]]
            t_banner = Table(banner_data, colWidths=[540])
            t_banner.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFF5F5")),
                ('BORDER', (0,0), (-1,-1), 1, c_danger),
                ('PADDING', (0,0), (-1,-1), 8),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ]))
            story.append(t_banner)
            story.append(Spacer(1, 12))

            # Executive Summary
            story.append(Paragraph("1. Executive Summary", h1_style))
            exec_summary = (
                "Inspection could not be completed. All uploaded images failed quality validation. "
                "No structural conclusions can be drawn. Please upload clearer inspection photographs."
            )
            story.append(Paragraph(exec_summary, summary_style))
            story.append(Spacer(1, 12))

            # N/A Metrics Grid
            story.append(Paragraph("2. Inspection Summary & Evidence Status", h1_style))
            kpi_data = [
                [
                    Paragraph("<b>Health Score (SHI)</b>", bold_body_style),
                    Paragraph("<b>Failure Probability</b>", bold_body_style),
                    Paragraph("<b>Remaining Useful Life</b>", bold_body_style),
                ],
                [
                    Paragraph("<font size=12 color='#C53030'><b>N/A</b></font>", body_style),
                    Paragraph("<font size=12 color='#C53030'><b>N/A</b></font>", body_style),
                    Paragraph("<font size=12 color='#C53030'><b>N/A</b></font>", body_style),
                ],
                [
                    Paragraph("<b>Maintenance Action</b>", bold_body_style),
                    Paragraph("<b>Detection Confidence</b>", bold_body_style),
                    Paragraph("<b>Inspection Confidence</b>", bold_body_style),
                ],
                [
                    Paragraph("<b>Inspection Required</b>", body_style),
                    Paragraph("<b>0%</b>", body_style),
                    Paragraph("<b>0%</b>", body_style),
                ]
            ]
            t_kpi = Table(kpi_data, colWidths=[180, 180, 180])
            t_kpi.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
                ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#EDF2F7")),
                ('GRID', (0,0), (-1,-1), 1, c_border),
                ('PADDING', (0,0), (-1,-1), 6),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(t_kpi)
            story.append(Spacer(1, 14))

            # Image Quality & Rejection Report Table
            story.append(Paragraph("3. Image Quality Validation & Rejection Report", h1_style))
            rejection_data = [
                [
                    Paragraph("<b>Image Name</b>", bold_body_style),
                    Paragraph("<b>Status</b>", bold_body_style),
                    Paragraph("<b>Rejection Reason</b>", bold_body_style)
                ]
            ]

            for img in image_results:
                img_name = img.get("image_name", "unknown.jpg")
                reason = img.get("rejection_reason") or (img.get("warnings")[0] if img.get("warnings") else "Failed quality check")
                rejection_data.append([
                    Paragraph(img_name, body_style),
                    Paragraph("<font color='#C53030'><b>Rejected</b></font>", body_style),
                    Paragraph(reason, body_style)
                ])

            t_rejection = Table(rejection_data, colWidths=[160, 90, 290])
            t_rejection.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 1, c_border),
                ('PADDING', (0,0), (-1,-1), 5),
                ('BACKGROUND', (0,0), (-1,0), c_light),
            ]))
            story.append(t_rejection)
            story.append(Spacer(1, 14))

            # Next Steps & Recommendations
            story.append(Paragraph("4. Recommendations & Required Action", h1_style))
            recs_text = """
            1. <b>Re-inspection Campaign Required:</b> Re-fly drone or re-capture inspection photographs ensuring sharp focus, adequate ambient illumination, and clear bridge structural framing.
            2. <b>Image Quality Guidelines:</b>
               <br/>&nbsp;&nbsp;&bull;&nbsp; Ensure motion blur is minimized (shutter speed $\ge 1/500$s).
               <br/>&nbsp;&nbsp;&bull;&nbsp; Target minimum frame resolution of $1920 \times 1080$ pixels.
               <br/>&nbsp;&nbsp;&bull;&nbsp; Ensure bridge structural members occupy $\ge 30\%$ of the image frame.
               <br/>&nbsp;&nbsp;&bull;&nbsp; Avoid flying during dense fog, heavy rain, or glare/overexposure extremes.
            """
            story.append(Paragraph(recs_text, body_style))
            story.append(Spacer(1, 16))

            # Mandatory Engineering Disclaimer Footer
            story.append(Paragraph("<b>Engineering Disclaimer & Audit Trail:</b>", h2_style))
            disclaimer_text = (
                "<i>No engineering conclusions were generated because the inspection dataset failed validation. "
                "The bridge condition remains unknown. Re-inspection is required.</i>"
            )
            story.append(Paragraph(disclaimer_text, body_style))

            doc.build(story)
            return str(filepath)

        # ---------------------------------------------------------------------
        # CASE 2: STATE 3 (PARTIAL_ANALYSIS) or STATE 4 (FULL_ANALYSIS)
        # ---------------------------------------------------------------------
        story.append(Paragraph("BridgeGuardian AI — Structural Assessment Report", title_style))
        story.append(Paragraph(f"<b>Campaign Inspection ID:</b> CAMP-ID-{inspection_id} &nbsp;|&nbsp; <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Spacer(1, 10))

        # Partial Analysis Banner if any images were rejected
        if pipeline_state == "PARTIAL_ANALYSIS":
            p_banner_data = [[Paragraph(f"⚠️ PARTIAL ANALYSIS: Only {valid_count} of {total_uploaded} uploaded images passed quality validation. {rejected_count} images were rejected.", warning_banner_style)]]
            t_pbanner = Table(p_banner_data, colWidths=[540])
            t_pbanner.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFFAF0")),
                ('BORDER', (0,0), (-1,-1), 1, c_warning),
                ('PADDING', (0,0), (-1,-1), 6),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ]))
            story.append(t_pbanner)
            story.append(Spacer(1, 10))

        # Executive Summary
        story.append(Paragraph("1. Executive Summary", h1_style))
        summary_text = explainability.get("summary_report", "Campaign complete. Assessment logs compile visual defect features.")
        story.append(Paragraph(summary_text, summary_style))
        story.append(Spacer(1, 10))
        
        # KPI Metrics Table
        risk_cat = health_predictions.get("risk_category", "Unknown")
        priority = maintenance.get("maintenance_priority", "Low")
        action = maintenance.get("maintenance_action", "Monitor")
        
        shi_val = health_predictions.get('health_score')
        shi_str = f"{shi_val}%" if isinstance(shi_val, (int, float)) else "N/A"
        pof_val = health_predictions.get('failure_probability')
        pof_str = f"{pof_val}%" if isinstance(pof_val, (int, float)) else "N/A"
        rul_val = health_predictions.get('rul_days')
        rul_str = f"{rul_val} days" if isinstance(rul_val, (int, float)) else "N/A"

        kpi_data = [
            [
                Paragraph("<b>Health Score (SHI)</b>", bold_body_style),
                Paragraph("<b>Failure Probability</b>", bold_body_style),
                Paragraph("<b>Remaining Useful Life</b>", bold_body_style),
            ],
            [
                Paragraph(f"<font size=12 color='#C53030'><b>{shi_str}</b></font><br/>({risk_cat})", body_style),
                Paragraph(f"<font size=12 color='#DD6B20'><b>{pof_str}</b></font>", body_style),
                Paragraph(f"<font size=12><b>{rul_str}</b></font>", body_style),
            ],
            [
                Paragraph("<b>Maintenance Action</b>", bold_body_style),
                Paragraph("<b>Priority Rank</b>", bold_body_style),
                Paragraph("<b>Repair Window</b>", bold_body_style),
            ],
            [
                Paragraph(f"<b>{action}</b>", body_style),
                Paragraph(f"<b>{priority}</b>", body_style),
                Paragraph(f"<b>{maintenance.get('repair_window_days')} days</b>", body_style),
            ]
        ]
        t_kpi = Table(kpi_data, colWidths=[180, 180, 180])
        t_kpi.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
            ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#EDF2F7")),
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_kpi)
        story.append(Spacer(1, 12))
        
        # Detailed Inspection Statistics
        story.append(Paragraph("2. Detailed Inspection Statistics", h1_style))
        stats = aggregate_stats
        
        cw_val = stats.get('largest_crack_width', 0.0)
        cl_val = stats.get('largest_crack_length', 0.0)
        cw_str = f"{cw_val} mm" if isinstance(cw_val, (int, float)) and cw_val > 0.0 else "N/A"
        cl_str = f"{cl_val} mm" if isinstance(cl_val, (int, float)) and cl_val > 0.0 else "N/A"

        stats_data = [
            [Paragraph("<b>Extracted Feature</b>", bold_body_style), Paragraph("<b>Value (Measured / Estimated)</b>", bold_body_style)],
            [Paragraph("Images Uploaded (Total / Accepted / Rejected)", body_style), Paragraph(f"{total_uploaded} uploaded / {valid_count} accepted / {rejected_count} rejected", body_style)],
            [Paragraph("Largest Crack Width (Estimated)", body_style), Paragraph(cw_str, body_style)],
            [Paragraph("Largest Crack Length (Estimated)", body_style), Paragraph(cl_str, body_style)],
            [Paragraph("Total Crack Coverage Area (Measured)", body_style), Paragraph(f"{stats.get('total_crack_area_percent', 0.0)}%", body_style)],
            [Paragraph("Total Rust Area (Measured)", body_style), Paragraph(f"{stats.get('rust_coverage_percent', 0.0)}%", body_style)],
            [Paragraph("Total Corrosion Area (Measured)", body_style), Paragraph(f"{stats.get('corrosion_coverage_percent', 0.0)}%", body_style)],
            [Paragraph("Critical Defects Detected", body_style), Paragraph(f"{stats.get('critical_defect_count', 0)}", body_style)],
            [Paragraph("Damage Diversity Index (Entropy)", body_style), Paragraph(f"{stats.get('damage_diversity_index', 0.0)}", body_style)],
            [Paragraph("Most Damaged Structural Component", body_style), Paragraph(f"{stats.get('most_damaged_structural_component', 'N/A')}", body_style)],
        ]
        t_stats = Table(stats_data, colWidths=[270, 270])
        t_stats.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (1,0), c_light),
        ]))
        story.append(t_stats)
        story.append(Spacer(1, 12))

        # Rejection Reasons Table if any images were rejected
        if rejected_count > 0:
            story.append(Paragraph("<b>Rejected Images Quality Summary:</b>", h2_style))
            rej_summary_data = [
                [Paragraph("<b>Image Name</b>", bold_body_style), Paragraph("<b>Rejection Reason</b>", bold_body_style)]
            ]
            for img in image_results:
                if not img.get("is_valid"):
                    reason = img.get("rejection_reason") or (img.get("warnings")[0] if img.get("warnings") else "Failed quality check")
                    rej_summary_data.append([
                        Paragraph(img.get("image_name", "unknown.jpg"), body_style),
                        Paragraph(reason, body_style)
                    ])
            t_rej_sum = Table(rej_summary_data, colWidths=[200, 340])
            t_rej_sum.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 1, c_border),
                ('PADDING', (0,0), (-1,-1), 4),
                ('BACKGROUND', (0,0), (-1,0), c_light),
            ]))
            story.append(t_rej_sum)
            story.append(Spacer(1, 12))

        # Inspection Confidence, Coverage & Evidence Provenance Section
        story.append(Paragraph("3. Inspection Confidence & Metric Provenance Traceability", h1_style))
        conf_data = [
            [Paragraph("<b>Metric</b>", bold_body_style), Paragraph("<b>Score</b>", bold_body_style), Paragraph("<b>Traceability & Provenance</b>", bold_body_style)],
            [
                Paragraph("Average Image Quality", body_style), 
                Paragraph(f"{performance_metrics.get('avg_image_quality', 0.0)}%", body_style),
                Paragraph(f"Quality validation check across {valid_count} accepted image frames", body_style)
            ],
            [
                Paragraph("Overall Detection Confidence", body_style), 
                Paragraph(f"{int(stats.get('overall_detection_confidence', 0.95) * 100)}%", body_style),
                Paragraph("Average probability score generated by Vision AI Engine", body_style)
            ],
            [
                Paragraph("Structural Area Coverage", body_style), 
                Paragraph(f"{int(stats.get('coverage_score', 1.0) * 100)}%", body_style),
                Paragraph(stats.get("provenance", {}).get("coverage_provenance", {}).get("derivation", "Observed structural area ratio across accepted frames"), body_style)
            ],
            [
                Paragraph("Health Score (SHI) Derivation", body_style), 
                Paragraph(f"<b>{health_predictions.get('health_score')}%</b>", bold_body_style),
                Paragraph(stats.get("provenance", {}).get("shi_provenance", {}).get("derivation", "Derived strictly from verified defect penalties"), body_style)
            ],
            [
                Paragraph("Failure Probability Derivation", body_style), 
                Paragraph(f"<b>{health_predictions.get('failure_probability')}%</b>", bold_body_style),
                Paragraph(stats.get("provenance", {}).get("failure_probability_provenance", {}).get("derivation", "Derived from verified defects and confidence bounds"), body_style)
            ],
        ]
        t_conf = Table(conf_data, colWidths=[150, 80, 310])
        t_conf.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (-1,0), c_light),
            ('BACKGROUND', (0,-2), (-1,-1), colors.HexColor("#F0FFF4")),
        ]))
        story.append(t_conf)
        
        story.append(PageBreak())

        # Component Findings Matrix: 4 Engineering States
        story.append(Paragraph("4. Component-Wise Evidence & Assessment Matrix", h1_style))
        story.append(Paragraph("Engineering evaluation mapped across four explicit states (Verified Healthy, No Visible Defect Observed, Unknown, Not Inspected):", body_style))
        story.append(Spacer(1, 8))
        
        comp_matrix_data = [
            [
                Paragraph("<b>Component</b>", bold_body_style),
                Paragraph("<b>Cracks</b>", bold_body_style),
                Paragraph("<b>Rust/Corrosion</b>", bold_body_style),
                Paragraph("<b>Severity</b>", bold_body_style),
                Paragraph("<b>Engineering Status</b>", bold_body_style)
            ]
        ]
        
        comp_findings_list = stats.get("component_findings", [])
        if comp_findings_list:
            for item in comp_findings_list:
                status_str = item.get("status", "No Visible Defect Observed")
                status_color = "#2B6CB0" # Blue default for No Visible Defect Observed
                
                if status_str == "Verified Healthy":
                    status_color = "#2F855A"
                elif status_str == "No Visible Defect Observed":
                    status_color = "#2B6CB0"
                elif status_str == "Unknown":
                    status_color = "#D97706"
                elif status_str == "Not Inspected":
                    status_color = "#718096"
                elif status_str == "Replace":
                    status_color = "#C53030"
                elif status_str == "Repair":
                    status_color = "#DD6B20"
                elif status_str == "Inspect":
                    status_color = "#3182CE"
                elif status_str == "Monitor":
                    status_color = "#2B6CB0"
                    
                comp_matrix_data.append([
                    Paragraph(item["component"], body_style),
                    Paragraph(item["cracks"], body_style),
                    Paragraph(item["rust"], body_style),
                    Paragraph(item.get("severity", "None"), body_style),
                    Paragraph(f"<font color='{status_color}'><b>{status_str}</b></font>", body_style)
                ])
        else:
            comp_matrix_data.append([Paragraph("No detected components in imagery.", body_style), Paragraph("N/A", body_style), Paragraph("N/A", body_style), Paragraph("Not Inspected", body_style), Paragraph("Not Inspected", body_style)])
            
        t_matrix = Table(comp_matrix_data, colWidths=[130, 75, 105, 100, 130])
        t_matrix.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (-1,0), c_light),
        ]))
        story.append(t_matrix)
        story.append(Spacer(1, 14))

        # Defect Gallery Overview
        story.append(Paragraph("5. Defect Gallery Overview", h1_style))
        
        campaign_defects = stats.get("defects", [])
        if campaign_defects:
            gallery_data = [
                [
                    Paragraph("<b>ID</b>", bold_body_style),
                    Paragraph("<b>Type</b>", bold_body_style),
                    Paragraph("<b>Component</b>", bold_body_style),
                    Paragraph("<b>Severity</b>", bold_body_style),
                    Paragraph("<b>Confidence</b>", bold_body_style),
                    Paragraph("<b>Measurements</b>", bold_body_style)
                ]
            ]
            for d in campaign_defects[:12]:
                meas = d.get("measurements", {})
                meas_text = f"W: {meas.get('width_mm', 0.0)}mm, L: {meas.get('length_mm', 0.0)}mm" if d["type"] == "Crack" else f"Area: {meas.get('area_pct', 0.0)}%"
                    
                gallery_data.append([
                    Paragraph(d.get("defect_id", "N/A"), body_style),
                    Paragraph(d["type"], body_style),
                    Paragraph(d.get("component", "Deck"), body_style),
                    Paragraph(d["severity"], body_style),
                    Paragraph(f"{int(d['confidence'] * 100)}%", body_style),
                    Paragraph(meas_text, body_style)
                ])
            t_gallery = Table(gallery_data, colWidths=[80, 90, 100, 70, 70, 130])
            t_gallery.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 1, c_border),
                ('PADDING', (0,0), (-1,-1), 4),
                ('BACKGROUND', (0,0), (-1,0), c_light),
            ]))
            story.append(t_gallery)
        else:
            story.append(Paragraph("<b>No verified structural defects detected within inspected regions.</b>", summary_style))

        story.append(Spacer(1, 14))

        # Section 6: Inspection Limitations & Unassessed Regions
        story.append(Paragraph("6. Inspection Limitations & Unassessed Regions", h1_style))
        story.append(Paragraph("Summary of structural regions and components that could not be fully certified during this campaign:", body_style))
        story.append(Spacer(1, 6))

        limitations_info = stats.get("inspection_limitations", {})
        uninspected_str = ", ".join(limitations_info.get("uninspected_components", ["Under-Deck Substructure", "Bearings", "Expansion Joints"])) or "None"
        occluded_str = ", ".join(limitations_info.get("occluded_low_conf_regions", ["Shadowed connection joints"])) or "None"
        rejected_count_str = f"{rejected_count} rejected photo(s)" if rejected_count > 0 else "0 rejected photos"
        coverage_pct_str = f"{int(round(stats.get('coverage_score', 1.0) * 100, 0))}%"
        eng_conf_str = f"{int(round(stats.get('engineering_confidence', 0.5) * 100, 0))}%"

        limitations_data = [
            [Paragraph("<b>Category</b>", bold_body_style), Paragraph("<b>Inspection Limitation Details</b>", bold_body_style)],
            [Paragraph("Estimated Structural Coverage", body_style), Paragraph(f"<b>{coverage_pct_str}</b> of total bridge surface area observed", body_style)],
            [Paragraph("Engineering Confidence Index", body_style), Paragraph(f"<b>{eng_conf_str}</b> (Multiplicative coverage, quality, and viewpoint factor)", body_style)],
            [Paragraph("Uninspected Components", body_style), Paragraph(f"<font color='#DD6B20'><b>{uninspected_str}</b></font>", body_style)],
            [Paragraph("Occluded / Shadowed Regions", body_style), Paragraph(occluded_str, body_style)],
            [Paragraph("Rejected Imagery", body_style), Paragraph(rejected_count_str, body_style)]
        ]

        t_limitations = Table(limitations_data, colWidths=[180, 360])
        t_limitations.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (-1,0), c_light),
            ('BACKGROUND', (0,3), (-1,3), colors.HexColor("#FFFAF0")),
        ]))
        story.append(t_limitations)
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Engineering Disclaimer & Certification Limit:</b>", h2_style))
        story.append(Paragraph(
            "<i>This inspection report evaluates only the observed structural surface regions captured in the accepted imagery batch. "
            "Uninspected structural regions (such as bearings, expansion joints, or under-deck members) cannot be certified as defect-free without targeted follow-up inspection.</i>",
            body_style
        ))
        
        # Build document
        doc.build(story)
        return str(filepath)

    def compile_dashboard_packet(
        self,
        inspection_id: int,
        health_predictions: Dict[str, Any],
        aggregate_stats: Dict[str, Any],
        explainability: Dict[str, Any],
        maintenance: Dict[str, Any],
        image_results: List[Dict[str, Any]],
        model_metadata: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compiles the processed findings into a single unified JSON response block for the frontend.
        """
        return {
            "inspection_id": inspection_id,
            "timestamp": datetime.now().isoformat(),
            "health_predictions": health_predictions,
            "aggregate_stats": aggregate_stats,
            "explainability": explainability,
            "maintenance": maintenance,
            "image_results": [
                {
                    "image_name": img["image_name"],
                    "is_valid": img["is_valid"],
                    "warnings": img.get("warnings", []),
                    "rejection_reason": img.get("rejection_reason", ""),
                    "metrics": img.get("metrics", {}),
                    "features": img.get("features", {}),
                    "visualizations": img.get("visualizations", {})
                }
                for img in image_results
            ],
            "model_metadata": model_metadata,
            "performance_metrics": performance_metrics
        }
