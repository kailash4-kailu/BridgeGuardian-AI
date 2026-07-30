"""
BridgeGuardian AI — Stacked Model Ensemble Engine
Combines predictions from multiple distinct ML algorithms (XGBoost, LightGBM, CatBoost,
Random Forest, Extra Trees) using Out-Of-Fold (OOF) K-Fold cross-validation stacking.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger("bridgeguardian.ml.stacking_ensemble")

# Attempt imports of gradient boosting frameworks with graceful fallbacks
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — skipping XGB in stacking ensemble.")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("LightGBM not installed — skipping LightGBM in stacking ensemble.")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False
    logger.warning("CatBoost not installed — skipping CatBoost in stacking ensemble.")


class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Stacked Ensemble Regressor combining XGBoost, LightGBM, CatBoost,
    Random Forest, and Extra Trees base models via K-Fold Out-of-Fold cross validation
    and a meta-regressor (Ridge / ElasticNet).
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        meta_alpha: float = 1.0,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state
        self.meta_alpha = meta_alpha
        self.base_models_: Dict[str, Any] = {}
        self.meta_model_: Optional[Ridge] = None
        self.model_weights_: Dict[str, float] = {}

    def _init_base_models(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1
            ),
        }
        if XGB_AVAILABLE:
            models["xgboost"] = XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05, random_state=self.random_state, n_jobs=-1
            )
        if LGB_AVAILABLE:
            models["lightgbm"] = LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05, random_state=self.random_state, verbose=-1, n_jobs=-1
            )
        if CAT_AVAILABLE:
            models["catboost"] = CatBoostRegressor(
                iterations=100, depth=6, learning_rate=0.05, random_seed=self.random_state, verbose=0
            )
        return models

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> StackedEnsembleRegressor:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        self.base_models_ = self._init_base_models()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_predictions = np.zeros((X_arr.shape[0], len(self.base_models_)))

        model_keys = list(self.base_models_.keys())

        # Generate Out-Of-Fold predictions for meta-learner training
        for col_idx, key in enumerate(model_keys):
            model = self.base_models_[key]
            oof = np.zeros(X_arr.shape[0])
            for train_idx, val_idx in kf.split(X_arr, y_arr):
                X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
                X_val = X_arr[val_idx]
                
                try:
                    model.fit(X_tr, y_tr)
                    oof[val_idx] = model.predict(X_val)
                except Exception as e:
                    logger.warning(f"Error fitting {key} on fold: {e}")
                    oof[val_idx] = np.mean(y_tr)
            
            oof_predictions[:, col_idx] = oof
            
            # Re-fit base model on entire dataset
            try:
                model.fit(X_arr, y_arr)
            except Exception as e:
                logger.error(f"Error re-fitting {key} on full dataset: {e}")

        # Fit Ridge meta-model on OOF prediction matrix
        self.meta_model_ = Ridge(alpha=self.meta_alpha, random_state=self.random_state)
        self.meta_model_.fit(oof_predictions, y_arr)

        # Store model weights derived from positive meta-coefficients
        raw_coefs = np.maximum(0.0, self.meta_model_.coef_)
        denom = np.sum(raw_coefs) + 1e-8
        for idx, key in enumerate(model_keys):
            self.model_weights_[key] = float(round(raw_coefs[idx] / denom, 4))

        logger.info(f"StackedEnsembleRegressor fitted. Model weights: {self.model_weights_}")
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X)
        model_keys = list(self.base_models_.keys())
        meta_features = np.zeros((X_arr.shape[0], len(model_keys)))

        for idx, key in enumerate(model_keys):
            meta_features[:, idx] = self.base_models_[key].predict(X_arr)

        if self.meta_model_ is not None:
            return self.meta_model_.predict(meta_features)
        
        # Weighted average fallback
        preds = np.zeros(X_arr.shape[0])
        for idx, key in enumerate(model_keys):
            preds += meta_features[:, idx] * self.model_weights_.get(key, 1.0 / len(model_keys))
        return preds


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Stacked Ensemble Classifier combining XGBoost, LightGBM, CatBoost,
    Random Forest, and Extra Trees base models via Stratified K-Fold Out-of-Fold cross validation
    and a Logistic Regression meta-classifier.
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state
        self.base_models_: Dict[str, Any] = {}
        self.meta_model_: Optional[LogisticRegression] = None
        self.classes_: np.ndarray = np.array([0, 1])

    def _init_base_models(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1
            ),
        }
        if XGB_AVAILABLE:
            models["xgboost"] = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.05, random_state=self.random_state, n_jobs=-1, eval_metric="logloss"
            )
        if LGB_AVAILABLE:
            models["lightgbm"] = LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.05, random_state=self.random_state, verbose=-1, n_jobs=-1
            )
        if CAT_AVAILABLE:
            models["catboost"] = CatBoostClassifier(
                iterations=100, depth=6, learning_rate=0.05, random_seed=self.random_state, verbose=0
            )
        return models

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> StackedEnsembleClassifier:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        self.base_models_ = self._init_base_models()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_probabilities = np.zeros((X_arr.shape[0], len(self.base_models_)))

        model_keys = list(self.base_models_.keys())

        for col_idx, key in enumerate(model_keys):
            model = self.base_models_[key]
            oof = np.zeros(X_arr.shape[0])
            for train_idx, val_idx in skf.split(X_arr, y_arr):
                X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
                X_val = X_arr[val_idx]
                
                try:
                    model.fit(X_tr, y_tr)
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_val)
                        oof[val_idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                    else:
                        oof[val_idx] = model.predict(X_val)
                except Exception as e:
                    logger.warning(f"Error fitting {key} classifier on fold: {e}")
                    oof[val_idx] = np.mean(y_tr)
            
            oof_probabilities[:, col_idx] = oof
            
            # Re-fit base model on full dataset
            try:
                model.fit(X_arr, y_arr)
            except Exception as e:
                logger.error(f"Error re-fitting {key} classifier on full dataset: {e}")

        # Fit LogisticRegression meta-model on OOF prediction probability matrix
        self.meta_model_ = LogisticRegression(random_state=self.random_state)
        self.meta_model_.fit(oof_probabilities, y_arr)

        logger.info(f"StackedEnsembleClassifier fitted across {len(self.base_models_)} base models.")
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_arr = np.asarray(X)
        model_keys = list(self.base_models_.keys())
        meta_features = np.zeros((X_arr.shape[0], len(model_keys)))

        for idx, key in enumerate(model_keys):
            model = self.base_models_[key]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_arr)
                meta_features[:, idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            else:
                meta_features[:, idx] = model.predict(X_arr)

        if self.meta_model_ is not None:
            return self.meta_model_.predict_proba(meta_features)

        # Average probability fallback
        avg_p = np.mean(meta_features, axis=1)
        return np.vstack([1.0 - avg_p, avg_p]).T

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
