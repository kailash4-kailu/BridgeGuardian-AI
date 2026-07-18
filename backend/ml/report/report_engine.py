"""
BridgeGuardian AI — Report Engine
Compiles dynamic PDF reports using ReportLab and formats structured JSON packets.
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
        Generates a comprehensive, enterprise-grade PDF assessment report containing
        all 14 analytical sections, visual overlays, and scientific disclosures.
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
        
        # Color definitions
        c_primary = colors.HexColor("#1A365D")    # Deep navy
        c_secondary = colors.HexColor("#2B6CB0")  # Slate blue
        c_text = colors.HexColor("#2D3748")       # Charcoal
        c_light = colors.HexColor("#F7FAFC")      # Off-white
        c_border = colors.HexColor("#E2E8F0")     # Light gray
        c_danger = colors.HexColor("#C53030")     # Warning red
        c_warning = colors.HexColor("#DD6B20")    # Orange
        c_success = colors.HexColor("#2F855A")    # Green
        
        # Typography styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
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
        
        demo_style = ParagraphStyle(
            "DemoBanner",
            parent=body_style,
            fontName="Helvetica-Bold",
            textColor=c_danger,
            alignment=1
        )

        # Header Title block
        story.append(Paragraph("BridgeGuardian AI — Structural Assessment Report", title_style))
        story.append(Paragraph(f"<b>Campaign Inspection ID:</b> CAMP-ID-{inspection_id} &nbsp;|&nbsp; <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Spacer(1, 10))
        
        # Demo Warning Banner
        demo_mode = os.environ.get("DEMO_MODE", "true").lower() == "true"
        if demo_mode:
            banner_data = [[Paragraph("⚠️ DEMO MODE ACTIVE: Visual defect detections and ML calibrations are simulated for UI demonstration.", demo_style)]]
            t_banner = Table(banner_data, colWidths=[540])
            t_banner.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFF5F5")),
                ('BORDER', (0,0), (-1,-1), 1, c_danger),
                ('PADDING', (0,0), (-1,-1), 6),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ]))
            story.append(t_banner)
            story.append(Spacer(1, 10))

        # 1. Executive Summary
        story.append(Paragraph("1. Executive Summary", h1_style))
        summary_text = explainability.get("summary_report", "Campaign complete. Assessment logs compile visual defect features.")
        story.append(Paragraph(summary_text, summary_style))
        story.append(Spacer(1, 10))
        
        # 1.1 KPI Metrics Grid Table
        risk_cat = health_predictions.get("risk_category", "Unknown")
        priority = maintenance.get("maintenance_priority", "Low")
        action = maintenance.get("maintenance_action", "Monitor")
        
        kpi_data = [
            [
                Paragraph("<b>Health Score (SHI)</b>", bold_body_style),
                Paragraph("<b>Failure Probability</b>", bold_body_style),
                Paragraph("<b>Remaining Useful Life</b>", bold_body_style),
            ],
            [
                Paragraph(f"<font size=12 color='#C53030'><b>{health_predictions.get('health_score')}%</b></font><br/>({risk_cat})", body_style),
                Paragraph(f"<font size=12 color='#DD6B20'><b>{health_predictions.get('failure_probability')}%</b></font>", body_style),
                Paragraph(f"<font size=12><b>{health_predictions.get('rul_days')} days</b></font>", body_style),
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
        
        # 2. Detailed Damage Statistics
        story.append(Paragraph("2. Detailed Inspection Statistics", h1_style))
        stats = aggregate_stats
        
        valid_count = performance_metrics.get("accepted_images", 0)
        rejected_count = performance_metrics.get("rejected_images", 0)
        total_uploaded = valid_count + rejected_count
        
        stats_data = [
            [Paragraph("<b>Extracted Feature</b>", bold_body_style), Paragraph("<b>Value (Measured / Estimated)</b>", bold_body_style)],
            [Paragraph("Images Uploaded (Total / Accepted / Rejected)", body_style), Paragraph(f"{total_uploaded} uploaded / {valid_count} accepted / {rejected_count} rejected", body_style)],
            [Paragraph("Largest Crack Width (Estimated)", body_style), Paragraph(f"{stats.get('largest_crack_width', 0.0)} mm", body_style)],
            [Paragraph("Largest Crack Length (Estimated)", body_style), Paragraph(f"{stats.get('largest_crack_length', 0.0)} mm", body_style)],
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

        # 3. Inspection Confidence & Coverage Section
        story.append(Paragraph("3. Inspection Confidence & Coverage", h1_style))
        conf_data = [
            [Paragraph("<b>Metric</b>", bold_body_style), Paragraph("<b>Score</b>", bold_body_style), Paragraph("<b>Methodology</b>", bold_body_style)],
            [
                Paragraph("Average Image Quality", body_style), 
                Paragraph(f"{performance_metrics.get('avg_image_quality', 0.0)}%", body_style),
                Paragraph("Blurriness and lighting constraints check across accepted images", body_style)
            ],
            [
                Paragraph("Overall Detection Confidence", body_style), 
                Paragraph(f"{int(stats.get('overall_detection_confidence', 0.95) * 100)}%", body_style),
                Paragraph("Average probability score generated by the vision detector", body_style)
            ],
            [
                Paragraph("Component Coverage Score", body_style), 
                Paragraph(f"{int(stats.get('coverage_score', 1.0) * 100)}%", body_style),
                Paragraph("Ratio of structural classes identified to total expected model classes", body_style)
            ],
            [
                Paragraph("<b>Integrated Inspection Confidence</b>", bold_body_style), 
                Paragraph(f"<b>{int(health_predictions.get('prediction_confidence', 0.95) * 100)}%</b>", bold_body_style),
                Paragraph("Multiplicative confidence mapping of visual quality and detection bounds", bold_body_style)
            ],
        ]
        t_conf = Table(conf_data, colWidths=[150, 80, 310])
        t_conf.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (-1,0), c_light),
            ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#F0FFF4")), # Light green highlight
        ]))
        story.append(t_conf)
        
        story.append(PageBreak())

        # 4. Component Findings Table
        story.append(Paragraph("4. Component-Wise Findings Matrix", h1_style))
        story.append(Paragraph("A comprehensive summary of defects, worst severities, and repair actions mapped directly to the bridge schematic hierarchy:", body_style))
        story.append(Spacer(1, 8))
        
        comp_matrix_data = [
            [
                Paragraph("<b>Component</b>", bold_body_style),
                Paragraph("<b>Cracks</b>", bold_body_style),
                Paragraph("<b>Rust/Corrosion</b>", bold_body_style),
                Paragraph("<b>Worst Severity</b>", bold_body_style),
                Paragraph("<b>Status / Action</b>", bold_body_style)
            ]
        ]
        
        comp_findings_list = stats.get("component_findings", [])
        if comp_findings_list:
            for item in comp_findings_list:
                status_color = "#2F855A"
                if item["status"] == "Replace":
                    status_color = "#C53030"
                elif item["status"] == "Repair":
                    status_color = "#DD6B20"
                elif item["status"] == "Inspect":
                    status_color = "#3182CE"
                    
                comp_matrix_data.append([
                    Paragraph(item["component"], body_style),
                    Paragraph(item["cracks"], body_style),
                    Paragraph(item["rust"], body_style),
                    Paragraph(item["severity"], body_style),
                    Paragraph(f"<font color='{status_color}'><b>{item['status']}</b></font>", body_style)
                ])
        else:
            comp_matrix_data.append([Paragraph("No component findings data compiled.", body_style)] + [Paragraph("", body_style)] * 4)
            
        t_matrix = Table(comp_matrix_data, colWidths=[130, 80, 110, 100, 120])
        t_matrix.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (-1,0), c_light),
        ]))
        story.append(t_matrix)
        story.append(Spacer(1, 14))

        # 5. Defect Gallery Overview
        story.append(Paragraph("5. Defect Gallery Overview", h1_style))
        story.append(Paragraph("Tabular index of unique structural defects identified during Campaign de-duplication:", body_style))
        story.append(Spacer(1, 8))
        
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
        
        campaign_defects = stats.get("defects", [])
        if campaign_defects:
            for d in campaign_defects[:12]: # Limit to first 12 defects to prevent overflow
                meas = d.get("measurements", {})
                meas_text = ""
                if d["type"] == "Crack":
                    meas_text = f"W: {meas.get('width_mm', 0.0)}mm, L: {meas.get('length_mm', 0.0)}mm"
                else:
                    meas_text = f"Area: {meas.get('area_pct', 0.0)}%"
                    
                gallery_data.append([
                    Paragraph(d.get("defect_id", "N/A"), body_style),
                    Paragraph(d["type"], body_style),
                    Paragraph(d.get("component", "Deck"), body_style),
                    Paragraph(d["severity"], body_style),
                    Paragraph(f"{int(d['confidence'] * 100)}%", body_style),
                    Paragraph(meas_text, body_style)
                ])
        else:
            gallery_data.append([Paragraph("No unique defects registered.", body_style)] + [Paragraph("", body_style)] * 5)
            
        t_gallery = Table(gallery_data, colWidths=[80, 90, 100, 70, 70, 130])
        t_gallery.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 4),
            ('BACKGROUND', (0,0), (-1,0), c_light),
        ]))
        story.append(t_gallery)
        
        story.append(PageBreak())

        # 6. Annotated Visual Overlays
        story.append(Paragraph("6. Annotated Inspection Images", h1_style))
        story.append(Paragraph("Visual defect overlays (Bounding Boxes) showing identified component boundaries and localized damage alerts:", body_style))
        story.append(Spacer(1, 10))
        
        img_count = 0
        for img in image_results:
            if not img.get("is_valid"):
                continue
            if img_count >= 2: # Show top 2 images with visual bounding boxes
                break
                
            img_name = img["image_name"]
            saved_paths = img.get("saved_paths", {})
            bbox_path = saved_paths.get("bboxes")
            
            if bbox_path and os.path.exists(bbox_path):
                story.append(Paragraph(f"<b>Image File:</b> {img_name} &mdash; Defect Bounding Boxes", h2_style))
                try:
                    r_img = Image(bbox_path, width=440, height=240)
                    story.append(r_img)
                    story.append(Spacer(1, 12))
                    img_count += 1
                except Exception as e:
                    logger.error(f"Failed to add Image {img_name} to PDF: {e}")
                    
        story.append(PageBreak())

        # 7. Heat Map Visualizations
        story.append(Paragraph("7. Defect Heat Map Visualizations", h1_style))
        story.append(Paragraph("Pixel density heatmaps indicating localized corrosion, cracking, or spalling concentration zones:", body_style))
        story.append(Spacer(1, 10))
        
        heatmap_count = 0
        for img in image_results:
            if not img.get("is_valid"):
                continue
            if heatmap_count >= 2:
                break
                
            img_name = img["image_name"]
            saved_paths = img.get("saved_paths", {})
            heatmap_path = saved_paths.get("heatmap")
            
            if heatmap_path and os.path.exists(heatmap_path):
                story.append(Paragraph(f"<b>Image File:</b> {img_name} &mdash; Damage Density Heat Map", h2_style))
                try:
                    h_img = Image(heatmap_path, width=440, height=240)
                    story.append(h_img)
                    story.append(Spacer(1, 12))
                    heatmap_count += 1
                except Exception as e:
                    logger.error(f"Failed to add heatmap {img_name} to PDF: {e}")
                    
        story.append(PageBreak())

        # 8. SHAP Explainability & ML input parameters
        story.append(Paragraph("8. SHAP Model Explanations & Input Features", h1_style))
        story.append(Paragraph("Point deductions indicating the direct visual defect penalty applied to baseline health scores:", body_style))
        story.append(Spacer(1, 8))
        
        shap_data = [
            [Paragraph("<b>Aggregated Feature</b>", bold_body_style), Paragraph("<b>Feature Value</b>", bold_body_style), Paragraph("<b>Health Score Deduction</b>", bold_body_style)]
        ]
        
        point_deductions = health_predictions.get("point_deductions", [])
        if point_deductions:
            for d in point_deductions:
                shap_data.append([
                    Paragraph(d["feature"], body_style),
                    Paragraph(d["value"], body_style),
                    Paragraph(f"<font color='#C53030'><b>-{d['deduction']} points</b></font>", body_style)
                ])
        else:
            shap_data.append([Paragraph("No penalty deductions applied (Bridge at healthy baseline).", body_style)] + [Paragraph("", body_style)] * 2)
            
        t_shap = Table(shap_data, colWidths=[200, 140, 200])
        t_shap.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (-1,0), c_light),
        ]))
        story.append(t_shap)
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("<b>Prediction Engine Input Features (Tabular baseline):</b>", h2_style))
        features_data = [
            [Paragraph("<b>Sensor Parameter</b>", bold_body_style), Paragraph("<b>Baseline Value</b>", bold_body_style), Paragraph("<b>Sensor Parameter</b>", bold_body_style), Paragraph("<b>Baseline Value</b>", bold_body_style)]
        ]
        
        baseline_features = health_predictions.get("baseline_features", {})
        feat_list = list(baseline_features.items())
        for idx in range(0, len(feat_list), 2):
            if idx + 1 < len(feat_list):
                k1, v1 = feat_list[idx]
                k2, v2 = feat_list[idx + 1]
                features_data.append([
                    Paragraph(k1, body_style), Paragraph(str(v1), body_style),
                    Paragraph(k2, body_style), Paragraph(str(v2), body_style)
                ])
                
        t_feat = Table(features_data, colWidths=[160, 110, 160, 110])
        t_feat.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 4),
            ('BACKGROUND', (0,0), (-1,0), c_light),
        ]))
        story.append(t_feat)
        story.append(Spacer(1, 14))

        # 9. Maintenance Recommendations
        story.append(Paragraph("9. Maintenance & Remediation Action Plan", h1_style))
        story.append(Paragraph(f"Based on the evaluated Structural Health Index (SHI) of <b>{health_predictions.get('health_score')}%</b>, the predictive maintenance planning window is compiled as follows:", body_style))
        story.append(Spacer(1, 8))
        
        rec_data = [
            [Paragraph("<b>Recommendation Detail</b>", bold_body_style), Paragraph("<b>Specification / Planning</b>", bold_body_style)],
            [Paragraph("Maintenance Category Action", body_style), Paragraph(f"<b>{action}</b> ({priority} Priority)", body_style)],
            [Paragraph("Recommended Repair Window", body_style), Paragraph(f"Within {maintenance.get('repair_window_days')} days", body_style)],
            [Paragraph("Next Recommended Routine Inspection", body_style), Paragraph(f"Every {maintenance.get('inspection_interval_days')} days", body_style)],
            [Paragraph("Reasoning for Remediation Action", body_style), Paragraph(f"Bridge score is downgraded by {round(health_predictions.get('health_baseline_score', 84.0) - health_predictions.get('health_score', 0.0), 1)} points due to the presence of {stats.get('critical_defect_count', 0)} critical defects, with worst severity rated as {stats.get('maximum_severity', 'Minor')}.", body_style)]
        ]
        t_rec = Table(rec_data, colWidths=[200, 340])
        t_rec.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (0,-1), c_light),
        ]))
        story.append(t_rec)
        
        story.append(PageBreak())

        # 10. Vision Model Information
        story.append(Paragraph("10. Computer Vision Model Architecture", h1_style))
        story.append(Paragraph(" Pluggable Deep Learning vision architectures registered under the Provider-Agnostic Engine schema:", body_style))
        story.append(Spacer(1, 8))
        
        model_info_data = [
            [Paragraph("<b>Subsystem</b>", bold_body_style), Paragraph("<b>Registered Provider</b>", bold_body_style), Paragraph("<b>Device Configuration</b>", bold_body_style)],
            [Paragraph("Base Detector Engine", body_style), Paragraph(str(model_metadata.get("model_name", "YOLOv11-BridgeGuardian")), body_style), Paragraph(str(model_metadata.get("device", "CPU")), body_style)],
            [Paragraph("Instance Segmenter", body_style), Paragraph("Segment Anything (SAM2-FastSegmenter)", body_style), Paragraph("CPU Multi-threaded Execution", body_style)],
            [Paragraph("Geometry Extractor", body_style), Paragraph("OpenCV Contours & Ellipse Fit", body_style), Paragraph("Single-thread CPU", body_style)],
            [Paragraph("Duplicate Merger", body_style), Paragraph("ORB Descriptors + Homography", body_style), Paragraph("CPU SIMD optimized", body_style)]
        ]
        t_m_info = Table(model_info_data, colWidths=[150, 210, 180])
        t_m_info.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (-1,0), c_light),
        ]))
        story.append(t_m_info)
        story.append(Spacer(1, 14))

        # 11. Performance Metrics
        story.append(Paragraph("11. Execution Performance Audit", h1_style))
        perf_data = [
            [Paragraph("<b>Parameter</b>", bold_body_style), Paragraph("<b>Specification</b>", bold_body_style)],
            [Paragraph("Total Computational Execution Time", body_style), Paragraph(f"{performance_metrics.get('total_processing_time_sec', 0.0)} seconds", body_style)],
            [Paragraph("Inference Rate (fps)", body_style), Paragraph(f"{performance_metrics.get('images_per_second', 0.0)} frames/sec", body_style)],
            [Paragraph("Peak Engine Memory Allocation", body_style), Paragraph(f"{performance_metrics.get('memory_usage_mb', 0.0)} MB", body_style)],
            [Paragraph("Inference Processing Host", body_style), Paragraph(f"{performance_metrics.get('device', 'CPU')}", body_style)],
        ]
        t_perf = Table(perf_data, colWidths=[250, 290])
        t_perf.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, c_border),
            ('PADDING', (0,0), (-1,-1), 5),
            ('BACKGROUND', (0,0), (0,-1), c_light),
        ]))
        story.append(t_perf)
        story.append(Spacer(1, 14))

        # 12. Scientific Transparency & Limitations
        story.append(Paragraph("12. Scientific Transparency & Model Limitations", h1_style))
        limitations_text = """
        <b>LIMITATION DISCLOSURE:</b> This inspection report is generated exclusively based on visual surface indicators captured through RGB drone imagery. Computer vision processing (YOLOv11/SAM2) cannot estimate internal materials parameters, fatigue limits, or subsurface corrosion.
        Specifically, the following cannot be evaluated through visible drone photographs:
        <br/>&nbsp;&nbsp;&bull;&nbsp; <b>Internal stress profiles and load capacity.</b>
        <br/>&nbsp;&nbsp;&bull;&nbsp; <b>Concrete core material strength and chemical carbonation.</b>
        <br/>&nbsp;&nbsp;&bull;&nbsp; <b>Fatigue cracking in steel components covered by paint.</b>
        <br/>&nbsp;&nbsp;&bull;&nbsp; <b>Hidden corrosion and subsurface rebar delamination.</b>
        <br/>&nbsp;&nbsp;&bull;&nbsp; <b>Overall structural deformation/deflection without geodetic baseline targets.</b>
        <br/><br/>
        Visual features serve as proxies to adjust numerical baseline models, which assume normal operations for unobserved internal mechanics.
        """
        story.append(Paragraph(limitations_text, body_style))
        story.append(Spacer(1, 14))

        # 13. Future Monitoring Plan
        story.append(Paragraph("13. Future Monitoring & Inspection Plan", h1_style))
        monitoring_plan_text = f"""
        Based on structural deterioration, the recommended monitoring plan consists of:
        <br/>1. <b>Routine UAV Drone Imagery Campaign:</b> Conduct scheduled photogrammetry every <b>{maintenance.get('inspection_interval_days')} days</b> to track defect propagation rates.
        <br/>2. <b>Localized Non-Destructive Testing (NDT):</b> Perform ultrasonic or radar scanning on Girders and bearing points within <b>{maintenance.get('repair_window_days')} days</b> to verify internal concrete delamination.
        <br/>3. <b>Sensor Baseline Audit:</b> Calibrate real-time strain gauges on Girders to verify load distribution and deflection values.
        """
        story.append(Paragraph(monitoring_plan_text, body_style))
        story.append(Spacer(1, 14))

        # 14. System Signature
        story.append(Paragraph("14. System Signature & Execution Record", h1_style))
        sig_data = [
            [Paragraph("<b>Campaign Identifier</b>", body_style), Paragraph(f"CAMP-ID-{inspection_id}", body_style)],
            [Paragraph("<b>Vision Engine Core</b>", body_style), Paragraph("YOLOv11-BridgeGuardian (v2.0.4-Demo)", body_style)],
            [Paragraph("<b>Database Stamp</b>", body_style), Paragraph(f"STAMP-{datetime.now().strftime('%Y%m%d%H%M%S')}-SQLITE", body_style)],
            [Paragraph("<b>Assigned Auditor</b>", body_style), Paragraph("BridgeGuardian AI Automated Pipeline", body_style)]
        ]
        t_sig = Table(sig_data, colWidths=[200, 340])
        t_sig.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 0.5, c_border),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_sig)

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
            "aggregate_stats": {
                "largest_crack_width": aggregate_stats.get("largest_crack_width", 0.0),
                "largest_crack_length": aggregate_stats.get("largest_crack_length", 0.0),
                "total_crack_area_percent": aggregate_stats.get("total_crack_area_percent", 0.0),
                "rust_coverage_percent": aggregate_stats.get("rust_coverage_percent", 0.0),
                "corrosion_coverage_percent": aggregate_stats.get("corrosion_coverage_percent", 0.0),
                "critical_defect_count": aggregate_stats.get("critical_defect_count", 0),
                "critical_defect_locations": aggregate_stats.get("critical_defect_locations", []),
                "most_damaged_structural_component": aggregate_stats.get("most_damaged_structural_component", "None"),
                "affected_structural_components": aggregate_stats.get("affected_structural_components", []),
                "damage_diversity_index": aggregate_stats.get("damage_diversity_index", 0.0),
                "images_containing_damage_percent": aggregate_stats.get("images_containing_damage_percent", 0.0),
                "maximum_severity": aggregate_stats.get("maximum_severity", "Minor"),
                "critical_zones": aggregate_stats.get("critical_zones", []),
                "hierarchy": aggregate_stats.get("hierarchy", {})
            },
            "explainability": explainability,
            "maintenance": maintenance,
            "image_results": [
                {
                    "image_name": img["image_name"],
                    "is_valid": img["is_valid"],
                    "warnings": img["warnings"],
                    "metrics": img["metrics"],
                    "features": img.get("features", {}),
                    "visualizations": img.get("visualizations", {})
                }
                for img in image_results
            ],
            "model_metadata": model_metadata,
            "performance_metrics": performance_metrics
        }
