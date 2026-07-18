"""
BridgeGuardian AI - Feature selection utilities.

Compares leakage-safe feature subsets on a chronological validation split.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

logger = logging.getLogger("bridgeguardian.feature_selector")


@dataclass
class FeatureSelectionResult:
    """Serializable feature-selection outcome for one target."""

    target_name: str
    selected_features: List[str]
    selected_method: str
    baseline_score: float
    selected_score: float
    method_scores: Dict[str, float]
    importances: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "selected_features": self.selected_features,
            "selected_method": self.selected_method,
            "baseline_score": round(float(self.baseline_score), 6),
            "selected_score": round(float(self.selected_score), 6),
            "method_scores": {
                name: round(float(score), 6) for name, score in self.method_scores.items()
            },
            "importances": self.importances,
        }


class FeatureSelector:
    """
    Selects feature subsets using multiple research-friendly techniques.

    The selector never looks at the held-out test set. It splits the provided
    training data chronologically and only keeps a subset when it improves the
    validation score over the all-feature baseline.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.ml_cfg = config.get("ml", {})
        training_cfg = config.get("training", {})
        self.enabled = training_cfg.get("feature_selection_enabled", True)
        self.random_state = self.ml_cfg.get("random_state", 42)
        self.n_jobs = self.ml_cfg.get("n_jobs", 1)
        self.max_rows = training_cfg.get("feature_selection_max_rows", 8000)
        self.min_features = training_cfg.get("feature_selection_min_features", 12)
        self.max_features = training_cfg.get("feature_selection_max_features", 60)
        self.validation_fraction = training_cfg.get("feature_selection_validation_fraction", 0.2)
        self.min_improvement = training_cfg.get("feature_selection_min_improvement", 0.001)

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str,
        target_name: str,
    ) -> FeatureSelectionResult:
        """Compare feature-selection methods and return the best safe subset."""
        X_numeric = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(X_numeric.median()).fillna(0.0)
        y = pd.Series(y).reset_index(drop=True)
        X_numeric = X_numeric.reset_index(drop=True)

        if not self.enabled or X_numeric.shape[1] <= self.min_features:
            features = list(X_numeric.columns)
            score = self._score_subset(X_numeric, y, features, task)
            return FeatureSelectionResult(
                target_name=target_name,
                selected_features=features,
                selected_method="all_features",
                baseline_score=score,
                selected_score=score,
                method_scores={"all_features": score},
                importances=[],
            )

        X_work, y_work = self._limit_rows(X_numeric, y)
        X_train, X_val, y_train, y_val = self._chronological_split(X_work, y_work)

        baseline_features = list(X_work.columns)
        baseline_score = self._score_train_val(
            X_train, X_val, y_train, y_val, baseline_features, task
        )
        candidates: Dict[str, Tuple[List[str], List[Dict[str, Any]]]] = {
            "mutual_information": self._mutual_information_features(X_train, y_train, task),
            "permutation_importance": self._permutation_features(
                X_train, X_val, y_train, y_val, task
            ),
            "recursive_feature_elimination": self._rfe_features(X_train, y_train, task),
        }

        method_scores: Dict[str, float] = {"all_features": baseline_score}
        for method, (features, _) in candidates.items():
            method_scores[method] = self._score_train_val(
                X_train, X_val, y_train, y_val, features, task
            )

        best_method = max(method_scores, key=method_scores.get)
        if best_method == "all_features":
            selected_features = baseline_features
            importances: List[Dict[str, Any]] = []
        elif method_scores[best_method] >= baseline_score + self.min_improvement:
            selected_features, importances = candidates[best_method]
        else:
            best_method = "all_features"
            selected_features = baseline_features
            importances = []

        logger.info(
            "Feature selection for %s: %s (%d/%d features, score %.4f vs baseline %.4f)",
            target_name,
            best_method,
            len(selected_features),
            len(baseline_features),
            method_scores[best_method],
            baseline_score,
        )

        return FeatureSelectionResult(
            target_name=target_name,
            selected_features=selected_features,
            selected_method=best_method,
            baseline_score=baseline_score,
            selected_score=method_scores[best_method],
            method_scores=method_scores,
            importances=importances[:50],
        )

    def _limit_rows(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if len(X) <= self.max_rows:
            return X, y
        return X.tail(self.max_rows).reset_index(drop=True), y.tail(self.max_rows).reset_index(drop=True)

    def _chronological_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        split_idx = max(1, int(len(X) * (1 - self.validation_fraction)))
        if split_idx >= len(X):
            split_idx = len(X) - 1
        return (
            X.iloc[:split_idx],
            X.iloc[split_idx:],
            y.iloc[:split_idx],
            y.iloc[split_idx:],
        )

    def _target_feature_count(self, n_features: int) -> int:
        max_features = min(self.max_features, n_features)
        min_features = min(self.min_features, max_features)
        return max(min_features, min(max_features, max(1, n_features // 2)))

    def _base_estimator(self, task: str) -> Any:
        if task == "classification":
            return RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        return RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=3,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def _mutual_information_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, task: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        try:
            if task == "classification":
                scores = mutual_info_classif(
                    X_train, y_train.astype(int), random_state=self.random_state
                )
            else:
                scores = mutual_info_regression(X_train, y_train, random_state=self.random_state)
        except Exception as exc:
            logger.warning("Mutual information failed: %s", exc)
            return list(X_train.columns), []

        return self._top_features_from_scores(X_train.columns, scores, "mutual_information")

    def _permutation_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        task: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        try:
            estimator = self._base_estimator(task)
            estimator.fit(X_train, y_train)
            if task == "classification":
                scoring = "roc_auc" if y_val.nunique() > 1 else "accuracy"
            else:
                scoring = "r2"
            result = permutation_importance(
                estimator,
                X_val,
                y_val,
                n_repeats=5,
                random_state=self.random_state,
                scoring=scoring,
                n_jobs=self.n_jobs,
            )
        except Exception as exc:
            logger.warning("Permutation importance failed: %s", exc)
            return list(X_train.columns), []

        return self._top_features_from_scores(
            X_train.columns, result.importances_mean, "permutation_importance"
        )

    def _rfe_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, task: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        try:
            estimator = self._base_estimator(task)
            selector = RFE(
                estimator=estimator,
                n_features_to_select=self._target_feature_count(X_train.shape[1]),
                step=0.35,
            )
            selector.fit(X_train, y_train)
        except Exception as exc:
            logger.warning("RFE failed: %s", exc)
            return list(X_train.columns), []

        features = [feature for feature, keep in zip(X_train.columns, selector.support_) if keep]
        rankings = [
            {
                "feature": feature,
                "importance": round(1.0 / float(rank), 6),
                "method": "recursive_feature_elimination",
            }
            for feature, rank in zip(X_train.columns, selector.ranking_)
        ]
        rankings.sort(key=lambda item: item["importance"], reverse=True)
        return features, rankings

    def _top_features_from_scores(
        self, columns: pd.Index, scores: np.ndarray, method: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        importances = [
            {"feature": feature, "importance": round(float(score), 8), "method": method}
            for feature, score in zip(columns, scores)
        ]
        importances.sort(key=lambda item: item["importance"], reverse=True)
        k = self._target_feature_count(len(importances))
        positive = [item for item in importances if item["importance"] > 0]
        chosen = positive[:k] if len(positive) >= self.min_features else importances[:k]
        return [item["feature"] for item in chosen], importances

    def _score_subset(
        self, X: pd.DataFrame, y: pd.Series, features: List[str], task: str
    ) -> float:
        X_work, y_work = self._limit_rows(X[features], y)
        X_train, X_val, y_train, y_val = self._chronological_split(X_work, y_work)
        return self._score_train_val(X_train, X_val, y_train, y_val, features, task)

    def _score_train_val(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        features: List[str],
        task: str,
    ) -> float:
        if not features:
            return float("-inf")
        try:
            estimator = clone(self._base_estimator(task))
            estimator.fit(X_train[features], y_train)
            if task == "classification":
                if y_val.nunique() < 2:
                    return float(accuracy_score(y_val, estimator.predict(X_val[features])))
                if hasattr(estimator, "predict_proba"):
                    scores = estimator.predict_proba(X_val[features])[:, 1]
                else:
                    scores = estimator.predict(X_val[features])
                return float(roc_auc_score(y_val, scores))
            predictions = estimator.predict(X_val[features])
            return float(r2_score(y_val, predictions))
        except Exception as exc:
            logger.warning("Feature subset scoring failed: %s", exc)
            return float("-inf")
