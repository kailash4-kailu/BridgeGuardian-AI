"""
BridgeGuardian AI — Model Trainer
Trains, compares, tunes, and persists multiple ML models for each target.
Uses TimeSeriesSplit for CV to respect temporal ordering.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

logger = logging.getLogger("bridgeguardian.model_trainer")


class ModelTrainer:
    """
    Trains multiple regression and classification models, compares them,
    selects the best, and persists all artifacts.

    Supports: RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost
    """

    REGRESSION_PARAM_GRIDS: Dict[str, Dict] = {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
        "ExtraTrees": {
            "n_estimators": [100, 200],
            "max_depth": [None, 15, 30],
            "min_samples_split": [2, 5],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [4, 6, 8],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
            "num_leaves": [31, 63, 127],
            "min_child_samples": [20, 50],
        },
        "CatBoost": {
            "iterations": [100, 200],
            "learning_rate": [0.05, 0.1],
            "depth": [4, 6, 8],
        },
    }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.ml_cfg = config.get("ml", {})
        self.models_dir = Path(self.ml_cfg.get("models_dir", "models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = self.ml_cfg.get("random_state", 42)
        self.cv_folds = self.ml_cfg.get("cv_folds", 5)
        self.n_iter = config.get("training", {}).get("n_iter_search", 20)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_name: str,
        task: str = "regression",
        X_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[pd.Series] = None,
    ) -> Tuple[Any, Dict]:
        """
        Train all applicable models, select the best, return it with metrics.

        Args:
            X_train: Feature matrix.
            y_train: Target series.
            target_name: Name of the target variable.
            task: "regression" or "classification".

        Returns:
            Tuple of (best_model, results_dict)
        """
        logger.info(f"Training models for target='{target_name}' task='{task}'")
        results: Dict[str, Dict] = {}

        models = self._build_model_candidates(task)
        tscv = self._build_time_series_split(len(X_train))

        for name, model in models.items():
            logger.info(f"  → Training {name} ...")
            t0 = time.time()

            param_grid = self.REGRESSION_PARAM_GRIDS.get(name, {})
            if param_grid:
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grid,
                    n_iter=min(self.n_iter, 10),
                    cv=tscv,
                    scoring="r2" if task == "regression" else "roc_auc",
                    n_jobs=self.ml_cfg.get("n_jobs", -1),
                    random_state=self.random_state,
                    verbose=0,
                    refit=True,
                )
                search.fit(X_train, y_train)
                tuned_model = search.best_estimator_
                best_params = search.best_params_
            else:
                tuned_model = clone(model)
                tuned_model.fit(X_train, y_train)
                best_params = {}

            score = self._cross_val_score(tuned_model, X_train, y_train, task, tscv)
            best = self._fit_final_model(tuned_model, name, X_train, y_train, task)
            elapsed = round(time.time() - t0, 2)
            evaluation = (
                self._evaluate_candidate(best, X_eval, y_eval, task)
                if X_eval is not None and y_eval is not None and len(X_eval) > 0
                else {}
            )

            results[name] = {
                "model": best,
                "params": best_params,
                "cv_score": score,
                "train_time_s": elapsed,
                "evaluation": evaluation,
            }
            logger.info(f"    {name}: cv_score={score:.4f} in {elapsed}s")

        best_name = max(results, key=lambda k: results[k]["cv_score"])
        best_model = results[best_name]["model"]
        logger.info(f"Best model for '{target_name}': {best_name} (cv={results[best_name]['cv_score']:.4f})")

        return best_model, {
            "best_model_name": best_name,
            "results": {k: {i: v for i, v in r.items() if i != "model"} for k, r in results.items()},
        }

    def save_model(self, model: Any, name: str) -> Path:
        """Persist a trained model to disk."""
        path = self.models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        logger.info(f"Model saved → {path}")
        return path

    def load_model(self, name: str) -> Any:
        """Load a persisted model from disk."""
        path = self.models_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return joblib.load(path)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_model_candidates(self, task: str) -> Dict[str, Any]:
        """Build dict of model candidates for the given task type."""
        n_jobs = self.ml_cfg.get("n_jobs", -1)
        rs = self.random_state

        if task == "regression":
            models: Dict[str, Any] = {
                "RandomForest": RandomForestRegressor(
                    n_estimators=100, random_state=rs, n_jobs=n_jobs
                ),
                "ExtraTrees": ExtraTreesRegressor(
                    n_estimators=100, random_state=rs, n_jobs=n_jobs
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=rs
                ),
            }
            if XGB_AVAILABLE:
                models["XGBoost"] = xgb.XGBRegressor(
                    n_estimators=100, random_state=rs, n_jobs=n_jobs,
                    verbosity=0, eval_metric="rmse",
                )
            if LGB_AVAILABLE:
                models["LightGBM"] = lgb.LGBMRegressor(
                    n_estimators=100, random_state=rs, n_jobs=n_jobs, verbose=-1
                )
            if CB_AVAILABLE:
                models["CatBoost"] = CatBoostRegressor(
                    iterations=100, random_seed=rs, verbose=0
                )
        else:  # classification
            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=100, class_weight="balanced",
                    random_state=rs, n_jobs=n_jobs
                ),
            }
            if XGB_AVAILABLE:
                scale_pos_weight = 50  # rough ratio for imbalanced alert
                models["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=100, random_state=rs, n_jobs=n_jobs,
                    scale_pos_weight=scale_pos_weight, verbosity=0,
                )
            if LGB_AVAILABLE:
                models["LightGBM"] = lgb.LGBMClassifier(
                    n_estimators=100, class_weight="balanced",
                    random_state=rs, n_jobs=n_jobs, verbose=-1
                )

        return models

    def _build_time_series_split(self, n_samples: int) -> TimeSeriesSplit:
        """Build a valid TimeSeriesSplit for the available sample count."""
        n_splits = min(self.cv_folds, max(2, n_samples // 10))
        n_splits = min(n_splits, max(2, n_samples - 2))
        return TimeSeriesSplit(n_splits=n_splits)

    def _fit_final_model(
        self,
        model: Any,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        task: str,
    ) -> Any:
        """Fit final model; use chronological early stopping where supported."""
        final_model = clone(model)
        early_stopping_rounds = self.ml_cfg.get("early_stopping_rounds", 25)
        supports_early_stopping = model_name in {"XGBoost", "LightGBM", "CatBoost"}

        if not supports_early_stopping or len(X) < 50:
            final_model.fit(X, y)
            return final_model

        split_idx = max(1, int(len(X) * 0.85))
        if split_idx >= len(X):
            final_model.fit(X, y)
            return final_model

        X_fit, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_fit, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        try:
            if model_name == "XGBoost":
                final_model.set_params(early_stopping_rounds=early_stopping_rounds)
                final_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
            elif model_name == "LightGBM":
                callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
                final_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], callbacks=callbacks)
            elif model_name == "CatBoost":
                final_model.fit(
                    X_fit,
                    y_fit,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=0,
                )
            logger.info("Early stopping enabled for %s", model_name)
            return final_model
        except Exception as exc:
            logger.warning(
                "Early stopping failed for %s (%s). Refitting on full training data.",
                model_name,
                exc,
            )
            final_model = clone(model)
            final_model.fit(X, y)
            return final_model

    def _cross_val_score(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task: str,
        cv: TimeSeriesSplit,
    ) -> float:
        """Compute mean CV score (R² for regression, ROC-AUC for classification)."""
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            fold_model = clone(model)
            try:
                fold_model.fit(X_tr, y_tr)
            except ValueError as exc:
                logger.warning("Skipping CV fold: %s", exc)
                continue
            if task == "regression":
                pred = fold_model.predict(X_val)
                scores.append(r2_score(y_val, pred))
            else:
                if y_val.nunique() < 2:
                    scores.append(0.5)
                    continue
                pred_proba = fold_model.predict_proba(X_val)[:, 1]
                try:
                    scores.append(roc_auc_score(y_val, pred_proba))
                except ValueError:
                    scores.append(0.5)
        return float(np.mean(scores)) if scores else float("-inf")

    def _evaluate_candidate(
        self,
        model: Any,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        task: str,
    ) -> Dict[str, float]:
        """Evaluate a candidate model on held-out chronological data."""
        if task == "regression":
            predictions = model.predict(X_eval)
            with np.errstate(divide="ignore", invalid="ignore"):
                denominator = np.where(y_eval == 0, 1, y_eval)
                mape_vals = np.abs((y_eval - predictions) / denominator)
                mape = float(np.nanmean(mape_vals) * 100)
            return {
                "r2": round(float(r2_score(y_eval, predictions)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_eval, predictions))), 6),
                "mae": round(float(mean_absolute_error(y_eval, predictions)), 6),
                "mape": round(mape, 2),
            }

        predictions = model.predict(X_eval)
        pred_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else predictions
        try:
            auc = roc_auc_score(y_eval, pred_proba) if y_eval.nunique() > 1 else 0.5
        except ValueError:
            auc = 0.5
        return {
            "accuracy": round(float(accuracy_score(y_eval, predictions)), 4),
            "f1_weighted": round(float(f1_score(y_eval, predictions, average="weighted", zero_division=0)), 4),
            "roc_auc": round(float(auc), 4),
        }


class ModelEvaluator:
    """Evaluates trained models on held-out test data."""

    def evaluate_regression(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_name: str,
    ) -> Dict[str, float]:
        """
        Compute regression metrics: R², RMSE, MAE, MAPE.

        Returns:
            Dictionary of metric names to values.
        """
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))

        # MAPE with guard for zero targets
        with np.errstate(divide="ignore", invalid="ignore"):
            mape_vals = np.abs((y_test - predictions) / np.where(y_test == 0, 1, y_test))
            mape = float(np.nanmean(mape_vals) * 100)

        metrics = {"r2": round(r2, 4), "rmse": round(rmse, 6), "mae": round(mae, 6), "mape": round(mape, 2)}
        logger.info(f"Evaluation [{target_name}]: {metrics}")
        return metrics

    def evaluate_classification(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_name: str,
    ) -> Dict[str, float]:
        """Compute classification metrics: accuracy, F1, ROC-AUC."""
        predictions = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, pred_proba)), 4),
        }
        logger.info(f"Evaluation [{target_name}]: {metrics}")
        return metrics
