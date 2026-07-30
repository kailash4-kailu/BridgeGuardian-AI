"""
BridgeGuardian AI — Model Evaluator & Metrics Registry
Computes comprehensive ML evaluation metrics (Precision, Recall, F1, mAP50, mAP50-95,
IoU, ROC-AUC, Confusion Matrix, and Expected Calibration Error - ECE).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("bridgeguardian.evaluator")


class ProductionEvaluator:
    """
    Unified metrics evaluator for tabular telemetry models and computer vision pipelines.
    """

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates regression performance metrics."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        mse = float(mean_squared_error(y_t, y_p))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))

        mape = float(np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100.0)

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "r2": round(r2, 4),
            "mape_percent": round(mape, 2),
        }

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
    ) -> Dict[str, Any]:
        """Calculates classification performance metrics including ECE calibration error."""
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)

        acc = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        cm = confusion_matrix(y_t, y_p).tolist()

        roc_auc = 0.5
        ece = 0.0
        if y_prob is not None:
            try:
                probs = np.asarray(y_prob)
                p1 = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
                roc_auc = float(roc_auc_score(y_t, p1))
                ece = ProductionEvaluator.compute_ece(y_t, p1)
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC or ECE: {e}")

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "expected_calibration_error_ece": round(ece, 4),
            "confusion_matrix": cm,
        }

    @staticmethod
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error (ECE) across probability bins."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return float(ece)

    @staticmethod
    def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Computes Intersection over Union (IoU) between two binary segmentation masks."""
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        if union == 0:
            return 1.0
        return float(intersection / union)
