"""
BridgeGuardian AI — Inference Pipeline
Orchestrates the full prediction flow: feature engineering → preprocessing → model prediction → RUL → explanation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from backend.core.config import get_config
from backend.ml.explainer import Explainer
from backend.ml.feature_engineer import FeatureEngineer
from backend.ml.rul_estimator import RULEstimator

logger = logging.getLogger("bridgeguardian.inference")

# Risk category boundaries (SHI-based)
RISK_CATEGORIES = {
    "Critical": (0.0, 0.40),
    "Poor": (0.40, 0.60),
    "Fair": (0.60, 0.75),
    "Good": (0.75, 0.85),
    "Excellent": (0.85, 1.01),
}

MAINTENANCE_PRIORITIES = {
    "Critical": "Emergency",
    "Poor": "High",
    "Fair": "Medium",
    "Good": "Low",
    "Excellent": "Routine",
}

RECOMMENDATIONS = {
    "Emergency": (
        "🚨 IMMEDIATE ACTION REQUIRED: Deploy emergency inspection team. "
        "Consider load restrictions or bridge closure. Contact structural engineer urgently."
    ),
    "High": (
        "⚠️ HIGH PRIORITY: Schedule maintenance within 7 days. "
        "Perform detailed crack and corrosion assessment. Review load capacity."
    ),
    "Medium": (
        "📋 MODERATE: Plan maintenance within 30 days. "
        "Inspect key structural joints and bearing pads. Check drainage systems."
    ),
    "Low": (
        "✅ LOW PRIORITY: Routine maintenance recommended within 3 months. "
        "Continue regular sensor monitoring. Clean debris from expansion joints."
    ),
    "Routine": (
        "🟢 HEALTHY: Standard inspection schedule. "
        "Maintain monitoring frequency. Review sensors annually."
    ),
}


RECOMMENDATIONS.update({
    "Emergency": (
        "IMMEDIATE ACTION REQUIRED: Deploy emergency inspection team. "
        "Consider load restrictions or bridge closure. Contact structural engineer urgently."
    ),
    "High": (
        "HIGH PRIORITY: Schedule maintenance within 7 days. "
        "Perform detailed crack and corrosion assessment. Review load capacity."
    ),
    "Medium": (
        "MODERATE: Plan maintenance within 30 days. "
        "Inspect key structural joints and bearing pads. Check drainage systems."
    ),
    "Low": (
        "LOW PRIORITY: Routine maintenance recommended within 3 months. "
        "Continue regular sensor monitoring. Clean debris from expansion joints."
    ),
    "Routine": (
        "HEALTHY: Standard inspection schedule. "
        "Maintain monitoring frequency. Review sensors annually."
    ),
})


class InferencePipeline:
    """
    End-to-end inference pipeline for BridgeGuardian AI.
    Loads trained models and preprocessors, runs predictions,
    generates explanations, and returns structured output.
    """

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.config = get_config()
        self._models: Dict[str, Any] = {}
        self._preprocessor: Optional[Any] = None
        self._feature_engineer: Optional[FeatureEngineer] = None
        self._explainers: Dict[str, Explainer] = {}
        self._rul_estimator = RULEstimator(self.config)
        self._model_version: str = "unknown"
        self._feature_columns: List[str] = []
        self._is_loaded = False

    def load(self) -> None:
        """Load all model artifacts from disk."""
        try:
            self._preprocessor = joblib.load(self.models_dir / "preprocessor.joblib")
            self._feature_engineer = joblib.load(self.models_dir / "feature_engineer.joblib")

            model_targets = {
                "health_score": "model_health_score.joblib",
                "failure_prob": "model_failure_prob.joblib",
                "maintenance_alert": "model_maintenance_alert.joblib",
            }
            for key, filename in model_targets.items():
                path = self.models_dir / filename
                if path.exists():
                    self._models[key] = joblib.load(path)
                    logger.info(f"Loaded model: {key} from {filename}")

            # Load explainers
            for key in self._models:
                expl_path = self.models_dir / f"explainer_{key}.joblib"
                if expl_path.exists():
                    self._explainers[key] = joblib.load(expl_path)

            # Load feature columns
            cols_path = self.models_dir / "feature_columns.json"
            if cols_path.exists():
                with open(cols_path) as f:
                    self._feature_columns = json.load(f)

            # Load version
            version_path = self.models_dir / "model_version.txt"
            if version_path.exists():
                self._model_version = version_path.read_text().strip()

            self._is_loaded = True
            logger.info(f"Inference pipeline loaded. Version: {self._model_version}")

        except Exception as e:
            logger.error(f"Failed to load inference pipeline: {e}")
            self._is_loaded = False
            raise

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and bool(self._models)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full prediction pipeline on a single bridge sensor reading.

        Args:
            input_data: Dictionary of feature values.

        Returns:
            Comprehensive prediction result dict.
        """
        if not self.is_ready:
            raise RuntimeError("Inference pipeline not loaded. Call load() first or train a model.")

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Feature engineering
        if self._feature_engineer is not None:
            df = self._feature_engineer.transform(df)

        # Preprocessing
        if self._preprocessor is not None:
            df_proc = self._preprocessor.transform(df)
        else:
            df_proc = df

        # Align feature columns
        X = self._align_features(df_proc)

        # Predictions
        health_score = self._predict_target("health_score", X, clip=(0, 1))
        failure_prob = self._predict_target("failure_prob", X, clip=(0, 1))
        maintenance_alert = self._predict_classification("maintenance_alert", X)

        # Extract SHI predictions from input for RUL
        shi_7d = input_data.get("SHI_Predicted_7d_Ahead")
        shi_30d = input_data.get("SHI_Predicted_30d_Ahead")
        rul_result = self._rul_estimator.estimate(health_score, shi_7d, shi_30d)

        # Risk classification
        risk_category = self._classify_risk(health_score)
        maintenance_priority = MAINTENANCE_PRIORITIES.get(risk_category, "Routine")
        recommendation = RECOMMENDATIONS.get(maintenance_priority, "")

        # Confidence estimate (based on model type and distance from boundaries)
        confidence = self._estimate_confidence(health_score, failure_prob)

        return {
            "health_score": round(health_score * 100, 2),  # 0–100 scale
            "health_score_raw": round(health_score, 4),
            "failure_probability": round(failure_prob * 100, 2),  # %
            "failure_probability_raw": round(failure_prob, 4),
            "rul_days": rul_result["rul_days"],
            "rul_degradation_rate": rul_result["degradation_rate_per_day"],
            "rul_confidence": rul_result["confidence"],
            "rul_message": rul_result["message"],
            "risk_category": risk_category,
            "maintenance_priority": maintenance_priority,
            "maintenance_recommendation": recommendation,
            "maintenance_alert": bool(maintenance_alert),
            "prediction_confidence": confidence,
            "model_version": self._model_version,
        }

    def explain(self, input_data: Dict[str, Any], target: str = "health_score") -> Dict[str, Any]:
        """
        Generate SHAP explanation for a prediction.

        Args:
            input_data: Input feature dictionary.
            target: Which model to explain ("health_score", "failure_prob").

        Returns:
            SHAP explanation dictionary.
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline not loaded")

        df = pd.DataFrame([input_data])
        if self._feature_engineer is not None:
            df = self._feature_engineer.transform(df)
        if self._preprocessor is not None:
            df_proc = self._preprocessor.transform(df)
        else:
            df_proc = df

        X = self._align_features(df_proc)

        if target in self._explainers:
            return self._explainers[target].explain_instance(
                self._features_for_target(target, X, prefer_explainer=True)
            )

        return Explainer._empty_explanation()

    def get_global_importance(self, target: str = "health_score") -> List[Dict]:
        if target in self._explainers and hasattr(self._explainers[target], "_background_data"):
            bg = self._explainers[target]._background_data
            if bg is not None:
                return self._explainers[target].get_global_importance(bg)
        return []

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has exactly the trained feature columns."""
        if not self._feature_columns:
            return df
        for col in self._feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[self._feature_columns]

    def _features_for_target(
        self,
        key: str,
        X: pd.DataFrame,
        prefer_explainer: bool = False,
    ) -> pd.DataFrame:
        """Subset features to the names used when the target model was trained."""
        feature_names: List[str] = []
        explainer = self._explainers.get(key)
        model = self._models.get(key)

        if prefer_explainer and explainer is not None:
            names = getattr(explainer, "_feature_names", None)
            if names is not None and len(names) > 0:
                feature_names = list(names)

        if not feature_names and model is not None:
            names = getattr(model, "feature_names_in_", None)
            if names is not None and len(names) > 0:
                feature_names = list(names)

        if not feature_names and explainer is not None:
            names = getattr(explainer, "_feature_names", None)
            if names is not None and len(names) > 0:
                feature_names = list(names)

        if not feature_names:
            return X

        X_target = X.copy()
        for col in feature_names:
            if col not in X_target.columns:
                X_target[col] = 0.0
        return X_target[feature_names]

    def _predict_target(
        self,
        key: str,
        X: pd.DataFrame,
        clip: Optional[tuple] = None,
    ) -> float:
        if key not in self._models:
            return 0.5  # Default fallback
        try:
            model = self._models[key]
            pred = float(model.predict(self._features_for_target(key, X))[0])
            if clip:
                pred = max(clip[0], min(clip[1], pred))
            return pred
        except Exception as e:
            logger.error(f"Prediction failed for '{key}': {e}")
            return 0.5

    def _predict_classification(self, key: str, X: pd.DataFrame) -> int:
        if key not in self._models:
            return 0
        try:
            model = self._models[key]
            return int(model.predict(self._features_for_target(key, X))[0])
        except Exception as e:
            logger.error(f"Classification failed for '{key}': {e}")
            return 0

    @staticmethod
    def _classify_risk(health_score: float) -> str:
        for category, (low, high) in RISK_CATEGORIES.items():
            if low <= health_score < high:
                return category
        return "Unknown"

    @staticmethod
    def _estimate_confidence(health_score: float, failure_prob: float) -> float:
        """
        Estimate prediction confidence (0–1) based on agreement between
        health score and failure probability.
        High SHI + Low PoF = high confidence; contradiction = lower confidence.
        """
        agreement = 1.0 - abs(health_score - (1 - failure_prob))
        return round(max(0.5, min(1.0, agreement)), 3)
