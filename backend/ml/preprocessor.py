"""
BridgeGuardian AI — Data Preprocessor
Handles cleaning, encoding, imputation, and scaling of bridge sensor data.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger("bridgeguardian.preprocessor")

# High-missing-rate columns to drop
HIGH_MISSING_THRESHOLD = 0.75


class Preprocessor:
    """
    Full preprocessing pipeline:
      1. Drop high-missing and irrelevant columns
      2. Remove duplicates
      3. Handle missing values (imputation)
      4. Encode categorical features
      5. Scale numerical features
      6. Clip target values to valid ranges
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.feature_cfg = config.get("features", {})
        self.target_cfg = config.get("targets", {})
        self.thresholds = config.get("thresholds", {})

        # State (learned during fit)
        self._numeric_imputer: Optional[SimpleImputer] = None
        self._cat_imputer: Optional[SimpleImputer] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scaler: Optional[StandardScaler] = None
        self._feature_columns: List[str] = []
        self._numeric_columns: List[str] = []
        self._cat_columns: List[str] = []
        self._dropped_columns: List[str] = []

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor on training data and return transformed data."""
        df = self._clean(df, fit=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        df = self._clean(df, fit=False)
        return df

    @property
    def feature_names(self) -> List[str]:
        return self._feature_columns.copy()

    # ------------------------------------------------------------------ #
    #  Internal pipeline                                                   #
    # ------------------------------------------------------------------ #

    def _clean(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        logger.info(f"Preprocessing start. Shape: {df.shape}, fit={fit}")

        df = self._drop_irrelevant(df, fit)
        df = self._remove_duplicates(df)
        df = self._impute_missing(df, fit)
        df = self._encode_categoricals(df, fit)
        df = self._clip_targets(df)
        df = self._scale_features(df, fit)

        logger.info(f"Preprocessing complete. Output shape: {df.shape}")
        return df

    def _drop_irrelevant(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Drop Timestamp and high-missing columns."""
        cols_to_drop = ["Timestamp"]

        if fit:
            for col in df.columns:
                missing_rate = df[col].isnull().mean()
                if missing_rate > HIGH_MISSING_THRESHOLD:
                    cols_to_drop.append(col)
                    logger.info(
                        f"Dropping '{col}': {missing_rate:.1%} missing > threshold"
                    )
            self._dropped_columns = cols_to_drop

        existing_drops = [c for c in self._dropped_columns if c in df.columns]
        return df.drop(columns=existing_drops, errors="ignore")

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df

    def _impute_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Impute numeric missing with median, categorical with mode."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Identify string-like categorical columns (exclude already-encoded)
        cat_cols = [
            c for c in df.columns
            if df[c].dtype == object or str(df[c].dtype) == "string"
        ]

        if fit:
            self._numeric_columns = num_cols

        if self._numeric_columns:
            if fit:
                self._numeric_imputer = SimpleImputer(strategy="median")
                df[self._numeric_columns] = self._numeric_imputer.fit_transform(
                    df[self._numeric_columns]
                )
            else:
                for col in self._numeric_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                if self._numeric_imputer is not None:
                    df[self._numeric_columns] = self._numeric_imputer.transform(
                        df[self._numeric_columns]
                    )
                else:
                    df[self._numeric_columns] = df[self._numeric_columns].fillna(0.0)

                extra_num_cols = [c for c in num_cols if c not in self._numeric_columns]
                if extra_num_cols:
                    df[extra_num_cols] = df[extra_num_cols].fillna(0.0)

        if cat_cols:
            for col in cat_cols:
                mode_val = df[col].mode()
                fill = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                df[col] = df[col].fillna(fill)

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Ordinal-encode categorical features with LabelEncoder."""
        cat_cols = [
            c for c in df.columns
            if df[c].dtype == object or str(df[c].dtype) == "string"
        ]
        self._cat_columns = cat_cols if fit else self._cat_columns

        if not fit:
            for col in self._cat_columns:
                if col not in df.columns:
                    df[col] = "Unknown"
            cat_cols = list(dict.fromkeys(cat_cols + self._cat_columns))

        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                # Add an Unknown class for unseen values at inference
                unique_vals = list(df[col].astype(str).unique()) + ["Unknown"]
                le.fit(unique_vals)
                self._label_encoders[col] = le
                df[col] = le.transform(df[col].astype(str))
            else:
                if col in self._label_encoders:
                    le = self._label_encoders[col]
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else "Unknown"
                    )
                    df[col] = le.transform(df[col])
                else:
                    df[col] = 0  # Default for unseen col at inference

        return df

    def _clip_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip probability and index targets to valid domain [0,1]."""
        clip_cols = [
            self.target_cfg.get("health_score", "Structural_Health_Index_SHI"),
            self.target_cfg.get("failure_prob", "Probability_of_Failure_PoF"),
            self.target_cfg.get("shi_24h", "SHI_Predicted_24h_Ahead"),
            self.target_cfg.get("shi_7d", "SHI_Predicted_7d_Ahead"),
            self.target_cfg.get("shi_30d", "SHI_Predicted_30d_Ahead"),
        ]
        for col in clip_cols:
            if col in df.columns:
                df[col] = df[col].clip(0.0, 1.0)
        return df

    def _scale_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """StandardScaler on numerical feature columns (NOT target columns)."""
        all_target_cols = list(self.target_cfg.values())
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in all_target_cols
        ]

        if not num_cols:
            return df

        if fit:
            self._scaler = StandardScaler()
            df[num_cols] = self._scaler.fit_transform(df[num_cols])
            self._feature_columns = num_cols
        else:
            if self._scaler is not None:
                available = [c for c in self._feature_columns if c in df.columns]
                missing_cols = [c for c in self._feature_columns if c not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing feature columns at inference: {missing_cols}")
                    for c in missing_cols:
                        df[c] = 0.0
                df[self._feature_columns] = self._scaler.transform(df[self._feature_columns])

        return df
