"""
BridgeGuardian AI — Feature Engineer
Creates time-series features: rolling statistics, lag features, and temporal encodings.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("bridgeguardian.feature_engineer")


class FeatureEngineer:
    """
    Generates enriched features from raw bridge sensor data.

    Adds:
      - Rolling statistics (mean, median, std, min, max) over configurable windows
      - Exponential moving averages and expanding-window statistics
      - Lag features for time-series awareness
      - Temporal features (hour, day-of-week, month)
      - Rate-of-change and acceleration features
      - Composite health indicators
      - Domain interaction features
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        training_cfg = config.get("training", {})
        self.rolling_windows: List[int] = training_cfg.get("rolling_windows", [5, 15, 30, 60])
        self.lag_steps: List[int] = training_cfg.get("lag_features", [1, 5, 15, 30])
        self.generated_features: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and apply transformations."""
        return self._transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to new data (same as fit for stateless ops)."""
        return self._transform(df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering steps."""
        df = df.copy()
        logger.info(f"Feature engineering starting. Input shape: {df.shape}")

        new_cols: Dict[str, pd.Series] = {}

        self._add_temporal_features(df, new_cols)
        self._add_rolling_features(df, new_cols)
        self._add_expanding_features(df, new_cols)
        self._add_lag_features(df, new_cols)
        self._add_rate_of_change(df, new_cols)
        self._add_acceleration_features(df, new_cols)
        self._add_composite_features(df, new_cols)
        self._add_interaction_features(df, new_cols)

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)

        logger.info(f"Feature engineering complete. Output shape: {df.shape}")
        return df

    def _add_temporal_features(self, df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> None:
        """Extract time-based features from Timestamp column."""
        ts_col = self.config.get("dataset", {}).get("timestamp_col", "Timestamp")
        if ts_col not in df.columns:
            return

        try:
            ts = pd.to_datetime(df[ts_col])
            new_cols["hour_of_day"] = ts.dt.hour
            new_cols["day_of_week"] = ts.dt.dayofweek
            new_cols["day_of_month"] = ts.dt.day
            new_cols["month"] = ts.dt.month
            new_cols["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

            # Cyclic encoding for hour and day
            new_cols["hour_sin"] = np.sin(2 * np.pi * new_cols["hour_of_day"] / 24)
            new_cols["hour_cos"] = np.cos(2 * np.pi * new_cols["hour_of_day"] / 24)
            new_cols["dow_sin"] = np.sin(2 * np.pi * new_cols["day_of_week"] / 7)
            new_cols["dow_cos"] = np.cos(2 * np.pi * new_cols["day_of_week"] / 7)

            self.generated_features.extend([
                "hour_of_day", "day_of_week", "day_of_month", "month", "is_weekend",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"
            ])
            logger.debug("Added temporal features")
        except Exception as e:
            logger.warning(f"Could not parse timestamps: {e}")

    def _add_rolling_features(self, df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> None:
        """Add rolling mean, std, min, max for key structural and sensor columns."""
        key_cols = [
            "Structural_Health_Index_SHI",
            "Strain_microstrain",
            "Deflection_mm",
            "Vibration_ms2",
            "Fatigue_Accumulation_au",
            "Corrosion_Level_percent",
            "Crack_Propagation_mm",
        ]
        available = [c for c in key_cols if c in df.columns]

        for col in available:
            s = df[col]
            for w in self.rolling_windows:
                roll = s.rolling(window=w, min_periods=1)
                new_cols[f"{col}_roll_mean_{w}"] = roll.mean()
                new_cols[f"{col}_roll_median_{w}"] = roll.median()
                new_cols[f"{col}_roll_std_{w}"] = roll.std().fillna(0)
                new_cols[f"{col}_ema_{w}"] = s.ewm(span=w, adjust=False, min_periods=1).mean()
                if w >= 30:
                    new_cols[f"{col}_roll_min_{w}"] = roll.min()
                    new_cols[f"{col}_roll_max_{w}"] = roll.max()

        logger.debug(f"Added rolling features with windows {self.rolling_windows}")

    def _add_expanding_features(self, df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> None:
        """Add past-aware expanding statistics for slowly changing sensor baselines."""
        expanding_cols = [
            "Strain_microstrain",
            "Deflection_mm",
            "Vibration_ms2",
            "Fatigue_Accumulation_au",
            "Corrosion_Level_percent",
        ]
        available = [c for c in expanding_cols if c in df.columns]

        for col in available:
            s = df[col]
            expanding = s.expanding(min_periods=2)
            new_cols[f"{col}_expanding_mean"] = expanding.mean().fillna(s)
            new_cols[f"{col}_expanding_std"] = expanding.std().fillna(0)

        logger.debug("Added expanding-window features")

    def _add_lag_features(self, df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> None:
        """Add lagged values for temporal context."""
        lag_cols = [
            "Structural_Health_Index_SHI",
            "Probability_of_Failure_PoF",
            "Strain_microstrain",
            "Vibration_ms2",
        ]
        available = [c for c in lag_cols if c in df.columns]

        for col in available:
            s = df[col]
            for lag in self.lag_steps:
                feat_name = f"{col}_lag_{lag}"
                new_cols[feat_name] = s.shift(lag).ffill().fillna(s.iloc[0])
                self.generated_features.append(feat_name)

        logger.debug(f"Added lag features with steps {self.lag_steps}")

    def _add_rate_of_change(self, df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> None:
        """Add first-order rate of change for key sensors."""
        roc_cols = [
            "Structural_Health_Index_SHI",
            "Strain_microstrain",
            "Deflection_mm",
            "Fatigue_Accumulation_au",
        ]
        available = [c for c in roc_cols if c in df.columns]

        for col in available:
            s = df[col]
            new_cols[f"{col}_roc_1"] = s.diff(1).fillna(0)
            new_cols[f"{col}_roc_5"] = s.diff(5).fillna(0)

        logger.debug("Added rate-of-change features")

    def _add_acceleration_features(self, df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> None:
        """Add second derivative features to capture acceleration in degradation."""
        accel_cols = [
            "Strain_microstrain",
            "Deflection_mm",
            "Vibration_ms2",
            "Fatigue_Accumulation_au",
            "Crack_Propagation_mm",
        ]
        available = [c for c in accel_cols if c in df.columns]

        for col in available:
            s = df[col]
            first_derivative = s.diff(1).fillna(0)
            new_cols[f"{col}_acceleration_1"] = first_derivative.diff(1).fillna(0)
            new_cols[f"{col}_acceleration_5"] = s.diff(5).diff(5).fillna(0)

        logger.debug("Added acceleration features")

    def _add_composite_features(self, df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> None:
        """Create domain-specific composite indicators."""

        # Structural stress index
        if all(c in df.columns for c in ["Strain_microstrain", "Deflection_mm", "Tilt_deg"]):
            strain_norm = df["Strain_microstrain"] / (df["Strain_microstrain"].max() + 1e-9)
            defl_norm = df["Deflection_mm"] / (df["Deflection_mm"].max() + 1e-9)
            tilt_norm = df["Tilt_deg"] / (df["Tilt_deg"].max() + 1e-9)
            new_cols["composite_stress_index"] = (strain_norm + defl_norm + tilt_norm) / 3.0

        # Environmental stress index
        if all(c in df.columns for c in ["Wind_Speed_ms", "Temperature_C", "Humidity_percent"]):
            wind_n = df["Wind_Speed_ms"] / (df["Wind_Speed_ms"].max() + 1e-9)
            temp_n = df["Temperature_C"].abs() / (df["Temperature_C"].abs().max() + 1e-9)
            hum_n = df["Humidity_percent"] / 100.0
            new_cols["composite_env_index"] = (wind_n * 0.5 + temp_n * 0.3 + hum_n * 0.2)

        # Load intensity
        if all(c in df.columns for c in ["Vehicle_Load_tons", "Traffic_Volume_vph"]):
            new_cols["load_intensity"] = (
                df["Vehicle_Load_tons"] * df["Traffic_Volume_vph"] / 1000.0
            )

        # Degradation velocity (SHI trend) - vectorized
        if "Structural_Health_Index_SHI" in df.columns:
            s = df["Structural_Health_Index_SHI"]
            new_cols["shi_degradation_velocity"] = (s.diff(60) / 60.0).fillna(0)

        logger.debug("Added composite features")

    def _add_interaction_features(self, df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> None:
        """Create physics-informed interaction terms for structural risk drivers."""
        interactions = {
            "humidity_x_corrosion": ("Humidity_percent", "Corrosion_Level_percent"),
            "traffic_x_vehicle_load": ("Traffic_Volume_vph", "Vehicle_Load_tons"),
            "temperature_x_strain": ("Temperature_C", "Strain_microstrain"),
            "wind_x_deflection": ("Wind_Speed_ms", "Deflection_mm"),
            "vibration_x_fatigue": ("Vibration_ms2", "Fatigue_Accumulation_au"),
            "crack_x_corrosion": ("Crack_Propagation_mm", "Corrosion_Level_percent"),
            "strain_x_deflection": ("Strain_microstrain", "Deflection_mm"),
            "env_severity_index": ("Temperature_C", "Humidity_percent"),
            "fatigue_x_corrosion": ("Fatigue_Accumulation_au", "Corrosion_Level_percent"),
        }

        for feature_name, (left, right) in interactions.items():
            if left in df.columns and right in df.columns:
                new_cols[feature_name] = df[left] * df[right]
                self.generated_features.append(feature_name)

        if "Vibration_ms2" in df.columns and "Modal_Frequency_Hz" in df.columns:
            new_cols["vibration_modal_ratio"] = df["Vibration_ms2"] / (df["Modal_Frequency_Hz"].abs() + 1e-4)
            self.generated_features.append("vibration_modal_ratio")

        if "Cable_Member_Tension_kN" in df.columns and "Bearing_Joint_Forces_kN" in df.columns:
            new_cols["cable_bearing_ratio"] = df["Cable_Member_Tension_kN"] / (df["Bearing_Joint_Forces_kN"].abs() + 1e-4)
            self.generated_features.append("cable_bearing_ratio")

        logger.debug("Added interaction features")

