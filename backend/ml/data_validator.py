"""
BridgeGuardian AI — Data Validator
Validates raw dataset integrity before processing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("bridgeguardian.data_validator")


@dataclass
class ValidationReport:
    """Container for data validation results."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False
        logger.error(f"[Validation ERROR] {msg}")

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(f"[Validation WARNING] {msg}")


class DataValidator:
    """
    Validates bridge sensor dataset for completeness, schema correctness,
    and domain-level constraints before ML processing.
    """

    REQUIRED_COLUMNS = [
        "Timestamp",
        "Strain_microstrain",
        "Deflection_mm",
        "Structural_Health_Index_SHI",
        "Probability_of_Failure_PoF",
    ]

    def __init__(self, config: dict) -> None:
        self.config = config
        self.targets = config.get("targets", {})

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation checks on the dataframe.

        Args:
            df: Raw input dataframe.

        Returns:
            ValidationReport with errors, warnings, and stats.
        """
        report = ValidationReport()
        logger.info(f"Starting data validation on DataFrame of shape {df.shape}")

        self._check_required_columns(df, report)
        self._check_row_count(df, report)
        self._check_missing_rates(df, report)
        self._check_duplicates(df, report)
        self._check_domain_constraints(df, report)
        self._check_target_distributions(df, report)
        self._populate_stats(df, report)

        if report.is_valid:
            logger.info("Data validation passed ✓")
        else:
            logger.error(f"Data validation failed with {len(report.errors)} error(s)")

        return report

    def _check_required_columns(self, df: pd.DataFrame, report: ValidationReport) -> None:
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            report.add_error(f"Missing required columns: {missing_cols}")

    def _check_row_count(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if len(df) < 100:
            report.add_error(f"Dataset too small: {len(df)} rows (minimum 100 required)")
        elif len(df) < 1000:
            report.add_warning(f"Dataset has only {len(df)} rows — model quality may be limited")

    def _check_missing_rates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        for col in df.columns:
            missing_pct = df[col].isnull().mean() * 100
            if missing_pct > 80:
                report.add_warning(
                    f"Column '{col}' has {missing_pct:.1f}% missing values — will be dropped"
                )
            elif missing_pct > 30:
                report.add_warning(
                    f"Column '{col}' has {missing_pct:.1f}% missing values — imputation will be applied"
                )

    def _check_duplicates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            report.add_warning(f"Found {n_dupes} duplicate rows — will be removed")

    def _check_domain_constraints(self, df: pd.DataFrame, report: ValidationReport) -> None:
        shi_col = self.targets.get("health_score", "Structural_Health_Index_SHI")
        pof_col = self.targets.get("failure_prob", "Probability_of_Failure_PoF")

        if shi_col in df.columns:
            shi = df[shi_col].dropna()
            if (shi < 0).any() or (shi > 1).any():
                report.add_warning(
                    f"'{shi_col}' contains values outside [0,1] — will be clipped"
                )

        if pof_col in df.columns:
            pof = df[pof_col].dropna()
            if (pof < 0).any() or (pof > 1).any():
                report.add_warning(
                    f"'{pof_col}' contains values outside [0,1] — will be clipped"
                )

    def _check_target_distributions(self, df: pd.DataFrame, report: ValidationReport) -> None:
        alert_col = self.targets.get("maintenance_alert", "Maintenance_Alert")
        if alert_col in df.columns:
            alert = df[alert_col].dropna()
            positive_rate = alert.mean() * 100
            if positive_rate < 0.5:
                report.add_warning(
                    f"'{alert_col}' is severely imbalanced: {positive_rate:.2f}% positive — "
                    "class weighting will be applied"
                )

    def _populate_stats(self, df: pd.DataFrame, report: ValidationReport) -> None:
        report.stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "missing_total": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }
