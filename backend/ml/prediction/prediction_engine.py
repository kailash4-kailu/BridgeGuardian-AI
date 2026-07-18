"""
BridgeGuardian AI — Prediction AI Engine Wrapper
Integrates aggregated vision feature stats into the tabular ML models,
applying a calibrated visual defect penalty to adjust Health, Failure Probability, and RUL.
"""
from __future__ import annotations
import logging
from typing import Any, Dict
from backend.ml.inference import InferencePipeline

logger = logging.getLogger("bridgeguardian.prediction.prediction_engine")

class PredictionEngine:
    def __init__(self, inference_pipeline: InferencePipeline) -> None:
        self.pipeline = inference_pipeline

    def predict(self, aggregate_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes aggregated visual features, runs the tabular ML model on a healthy baseline,
        and scales outputs using visual damage penalties.
        """
        if not self.pipeline.is_ready:
            logger.info("Initializing baseline ML Inference Pipeline in Prediction Engine")
            self.pipeline.load()
            
        # 1. Establish a healthy baseline sensor profile
        # Since sensor data is not available on drone images, we default to healthy values
        baseline_input = {
            "Deflection_mm": 1.2,
            "Strain_microstrain": 85.0,
            "Strain_microstrain_roll_mean_30": 85.0,
            "Strain_microstrain_ema_15": 85.0,
            "Air_Temp_degC": 20.0,
            "Relative_Humidity_percent": 50.0,
            "Wind_Speed_m_s": 3.0,
            "Traffic_Count_minute": 5.0,
            "SHI_Predicted_7d_Ahead": 0.84,
            "SHI_Predicted_30d_Ahead": 0.84
        }
        
        # 2. Get baseline predictions from the trained Random Forest / XGBoost models
        pred_res = self.pipeline.predict(baseline_input)
        
        # Extract baseline values
        health_baseline = pred_res["health_score"] / 100.0 # 0.0 to 1.0
        fail_prob_baseline = pred_res["failure_probability"] / 100.0 # 0.0 to 1.0
        rul_days = pred_res["rul_days"]
        
        # 3. Calculate calibrated visual damage penalty based on multi-image aggregated stats
        # Defect stats extracted by structural analysis engine
        max_crack_width = aggregate_stats.get("largest_crack_width", 0.0)
        rust_pct = aggregate_stats.get("rust_coverage_percent", 0.0)
        corrosion_pct = aggregate_stats.get("corrosion_coverage_percent", 0.0)
        critical_count = aggregate_stats.get("critical_defect_count", 0)
        max_severity = aggregate_stats.get("maximum_severity", "Minor")
        
        # Multipliers (calibrated impact values)
        # Crack width penalty
        crack_penalty = 1.0
        if max_crack_width > 4.0:
            crack_penalty = 0.70 # -30% health
        elif max_crack_width > 2.0:
            crack_penalty = 0.82 # -18% health
        elif max_crack_width > 0.8:
            crack_penalty = 0.92 # -8% health
            
        # Corrosion & Rust coverage penalty
        rust_pct_combined = rust_pct + corrosion_pct
        corrosion_penalty = 1.0
        if rust_pct_combined > 8.0:
            corrosion_penalty = 0.75
        elif rust_pct_combined > 3.0:
            corrosion_penalty = 0.88
        elif rust_pct_combined > 1.0:
            corrosion_penalty = 0.95
            
        # Critical defect density penalty
        critical_penalty = 1.0
        if critical_count >= 5:
            critical_penalty = 0.70
        elif critical_count >= 2:
            critical_penalty = 0.85
        elif critical_count == 1:
            critical_penalty = 0.93
            
        # Overall worst severity modifier
        severity_modifiers = {
            "Critical": 0.80,
            "Severe": 0.90,
            "Moderate": 0.96,
            "Minor": 1.0
        }
        severity_penalty = severity_modifiers.get(max_severity, 1.0)
        
        # Combined multiplicative penalty
        total_penalty = crack_penalty * corrosion_penalty * critical_penalty * severity_penalty
        
        # Apply bounds to prevent health score going below 10%
        final_health = max(0.10, health_baseline * total_penalty)
        
        # Adjusted failure probability increases proportionally as health drops
        # Failure probability is scaled up towards a maximum of 95%
        health_reduction = max(0.0, health_baseline - final_health)
        final_fail_prob = min(0.95, fail_prob_baseline + (health_reduction * 1.25))
        
        # Adjust remaining useful life (RUL) linearly based on health reduction
        final_rul_days = max(30.0, float(round(rul_days * (final_health / health_baseline), 1)))
        
        # Re-classify risk category based on adjusted health score
        risk_category = self.pipeline._classify_risk(final_health)
        
        # Calculate prediction confidence
        agreement = 1.0 - abs(final_health - (1.0 - final_fail_prob))
        prediction_confidence = round(max(0.5, min(1.0, agreement)), 3)
        
        # Calculate point deductions
        base_h = health_baseline * 100
        point_deductions = []
        if crack_penalty < 1.0:
            point_deductions.append({
                "feature": "Crack Width",
                "value": f"{max_crack_width:.1f} mm",
                "deduction": float(round((1.0 - crack_penalty) * base_h, 1))
            })
        if corrosion_penalty < 1.0:
            point_deductions.append({
                "feature": "Rust/Corrosion Coverage",
                "value": f"{rust_pct_combined:.2f}%",
                "deduction": float(round((1.0 - corrosion_penalty) * base_h, 1))
            })
        if critical_penalty < 1.0:
            point_deductions.append({
                "feature": "Critical Defects",
                "value": f"{critical_count} detected",
                "deduction": float(round((1.0 - critical_penalty) * base_h, 1))
            })
        if severity_penalty < 1.0:
            point_deductions.append({
                "feature": "Worst Severity Modifier",
                "value": max_severity,
                "deduction": float(round((1.0 - severity_penalty) * base_h, 1))
            })

        return {
            "health_score": round(final_health * 100, 2),
            "failure_probability": round(final_fail_prob * 100, 2),
            "rul_days": final_rul_days,
            "risk_category": risk_category,
            "prediction_confidence": prediction_confidence,
            "health_baseline_score": round(base_h, 2),
            "baseline_features": baseline_input,
            "point_deductions": point_deductions,
            "penalties": {
                "crack_penalty": round(crack_penalty, 3),
                "corrosion_penalty": round(corrosion_penalty, 3),
                "critical_penalty": round(critical_penalty, 3),
                "severity_penalty": round(severity_penalty, 3),
                "total_penalty": round(total_penalty, 3)
            }
        }
