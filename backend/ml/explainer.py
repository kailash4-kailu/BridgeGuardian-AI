"""
BridgeGuardian AI — SHAP Explainability Pipeline
Generates SHAP explanations for all predictions with multiple plot types.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("bridgeguardian.explainer")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available — explainability features disabled")


class Explainer:
    """
    SHAP-based model explainer.
    Supports TreeExplainer (fast, for tree-based models) with
    KernelExplainer fallback for non-tree models.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.ml_cfg = config.get("ml", {})
        self.max_samples = self.ml_cfg.get("shap_max_samples", 500)
        self._explainer: Optional[Any] = None
        self._feature_names: List[str] = []
        self._background_data: Optional[pd.DataFrame] = None

    def fit(self, model: Any, X_background: pd.DataFrame) -> None:
        """
        Fit the SHAP explainer on background training data.

        Args:
            model: Trained ML model.
            X_background: Training data for background distribution.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed — skipping explainer fit")
            return

        self._feature_names = list(X_background.columns)

        # Sample background for efficiency
        sample_size = min(self.max_samples, len(X_background))
        bg = X_background.sample(n=sample_size, random_state=42)
        self._background_data = bg

        try:
            self._explainer = shap.TreeExplainer(model)
            logger.info("SHAP TreeExplainer created ✓")
        except Exception as tree_err:
            logger.warning(f"TreeExplainer failed ({tree_err}) — using KernelExplainer")
            try:
                self._explainer = shap.KernelExplainer(model.predict, bg)
                logger.info("SHAP KernelExplainer created ✓")
            except Exception as e:
                logger.error(f"Failed to create SHAP explainer: {e}")
                self._explainer = None

    def explain_instance(
        self,
        X_instance: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction instance.

        Returns:
            Dict with shap_values, base_value, feature_importances, top_features
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            return self._empty_explanation()

        try:
            shap_values = self._explainer.shap_values(X_instance)

            # Handle multi-output (take first output for regression)
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            sv = np.array(sv).flatten()
            feature_names = self._feature_names

            # Build per-feature importance
            importances = [
                {
                    "feature": feat,
                    "shap_value": round(float(val), 6),
                    "direction": "positive" if val > 0 else "negative",
                }
                for feat, val in zip(feature_names, sv)
            ]
            importances.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            top_positive = [f for f in importances if f["direction"] == "positive"][:5]
            top_negative = [f for f in importances if f["direction"] == "negative"][:5]

            base_value = float(self._explainer.expected_value)
            if isinstance(self._explainer.expected_value, (list, np.ndarray)):
                base_value = float(self._explainer.expected_value[-1])

            return {
                "base_value": round(base_value, 6),
                "shap_values": [round(float(v), 6) for v in sv],
                "feature_names": feature_names,
                "feature_importances": importances[:20],  # top 20
                "top_positive_features": top_positive,
                "top_negative_features": top_negative,
                "prediction_contribution": round(float(sv.sum()), 6),
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._empty_explanation()

    def get_global_importance(
        self,
        X_sample: pd.DataFrame,
        max_samples: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Compute global feature importance from SHAP values across a sample.

        Returns:
            List of dicts with feature name and mean absolute SHAP value.
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            return []

        try:
            sample = X_sample.sample(n=min(max_samples, len(X_sample)), random_state=42)
            shap_vals = self._explainer.shap_values(sample)

            if isinstance(shap_vals, list):
                sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            else:
                sv = shap_vals

            sv = np.array(sv)
            mean_abs = np.mean(np.abs(sv), axis=0)

            result = [
                {"feature": feat, "importance": round(float(imp), 6)}
                for feat, imp in zip(self._feature_names, mean_abs)
            ]
            result.sort(key=lambda x: x["importance"], reverse=True)
            return result[:30]

        except Exception as e:
            logger.error(f"Global importance computation failed: {e}")
            return []

    @staticmethod
    def _empty_explanation() -> Dict[str, Any]:
        return {
            "base_value": 0.0,
            "shap_values": [],
            "feature_names": [],
            "feature_importances": [],
            "top_positive_features": [],
            "top_negative_features": [],
            "prediction_contribution": 0.0,
            "note": "Explainability not available",
        }
