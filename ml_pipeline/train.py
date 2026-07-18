"""
BridgeGuardian AI — Standalone ML Training Orchestrator
Runs the complete ML pipeline: validate → clean → engineer → train → evaluate → save.

Usage:
    python ml_pipeline/train.py
    python ml_pipeline/train.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logging_config import setup_logging
from backend.ml.data_validator import DataValidator
from backend.ml.explainer import Explainer
from backend.ml.feature_engineer import FeatureEngineer
from backend.ml.feature_selector import FeatureSelector
from backend.ml.model_trainer import ModelEvaluator, ModelTrainer
from backend.ml.preprocessor import Preprocessor

logger = setup_logging(level="INFO", name="bridgeguardian.train")

TRAIN_TARGET_PRIORITY: Tuple[Tuple[str, str], ...] = (
    ("shi_24h", "regression"),
    ("shi_7d", "regression"),
    ("shi_30d", "regression"),
    ("failure_prob", "regression"),
    ("maintenance_alert", "classification"),
    ("health_score", "regression"),
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(config: dict) -> pd.DataFrame:
    dataset_cfg = config.get("dataset", {})
    path = dataset_cfg.get("path", "dataset/bridge dataset.csv")
    logger.info(f"Loading dataset from '{path}' ...")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def dataset_version(config: dict) -> str:
    """Create a stable dataset fingerprint for metadata and auditability."""
    path = Path(config.get("dataset", {}).get("path", "dataset/bridge dataset.csv"))
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def build_target_series(df: pd.DataFrame, config: dict) -> Dict[str, pd.Series]:
    """Extract all target columns as a dict of Series."""
    target_cfg = config.get("targets", {})
    targets = {}
    for key, col in target_cfg.items():
        if col in df.columns:
            targets[key] = df[col].copy()
    return targets


def build_training_targets(targets: Dict[str, pd.Series], config: dict) -> "OrderedDict[str, Tuple[str, str]]":
    """Return forecast-first target definitions available in the dataset."""
    target_cfg = config.get("targets", {})
    train_targets: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()
    for key, task in TRAIN_TARGET_PRIORITY:
        if key in targets and key in target_cfg:
            train_targets[key] = (target_cfg[key], task)
    return train_targets


def remove_target_leakage(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop target columns and direct look-ahead columns from feature matrix
    to prevent data leakage.
    """
    target_cfg = config.get("targets", {})
    target_cols = [col for col in target_cfg.values() if col]
    explicit_leakage_cols = {
        "Bridge_Mood_Meter",
        "Carbon_Footprint_tCO2e_incremental",
    }

    def is_leakage_column(column: str) -> bool:
        if column in explicit_leakage_cols or column.startswith("Simulated_"):
            return True
        return any(column == target or column.startswith(f"{target}_") for target in target_cols)

    cols_to_drop = [column for column in df.columns if is_leakage_column(column)]
    if cols_to_drop:
        logger.info("Dropping %d leakage columns before training", len(cols_to_drop))
    return df.drop(columns=cols_to_drop, errors="ignore")

    target_cfg = config.get("targets", {})
    leakage_cols = list(target_cfg.values()) + [
        "SHI_Predicted_24h_Ahead",
        "SHI_Predicted_7d_Ahead",
        "SHI_Predicted_30d_Ahead",
        "Bridge_Mood_Meter",  # derived label — not available at inference
        "Estimated_Repair_Cost_USD_incremental",
        "Carbon_Footprint_tCO2e_incremental",
        "Simulated_Localized_Stress_Index",
        "Simulated_Slope_Displacement_mm",
        "Simulated_Water_Flow_m3s",
        "Simulated_Wind_Load_Pressure_kPa",
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def write_model_comparison(
    all_results: Dict[str, Dict[str, Any]],
    docs_dir: Path,
    training_date: str,
) -> None:
    """Write a Markdown comparison table for all trained candidates."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for target_key, result in all_results.items():
        task = result["task"]
        feature_count = result.get("feature_count", 0)
        for model_name, info in result["training_info"]["results"].items():
            metrics = info.get("evaluation", {})
            ranking_score = (
                metrics.get("roc_auc", 0.0)
                if task == "classification"
                else -metrics.get("rmse", float("inf"))
            )
            rows.append({
                "target_key": target_key,
                "model": model_name,
                "training_time": info.get("train_time_s", 0.0),
                "rmse": metrics.get("rmse", ""),
                "mae": metrics.get("mae", ""),
                "mape": metrics.get("mape", ""),
                "r2": metrics.get("r2", ""),
                "roc_auc": metrics.get("roc_auc", ""),
                "best_hyperparameters": info.get("params", {}),
                "feature_count": feature_count,
                "ranking_score": ranking_score,
            })

    rows.sort(key=lambda row: row["ranking_score"], reverse=True)
    lines = [
        "# Model Comparison",
        "",
        f"Training date: {training_date}",
        "",
        "| Rank | Target | Model | Training Time (s) | RMSE | MAE | MAPE | R2 | ROC AUC | Best Hyperparameters | Feature Count |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        params = json.dumps(row["best_hyperparameters"], sort_keys=True)
        lines.append(
            "| {rank} | {target} | {model} | {time} | {rmse} | {mae} | {mape} | "
            "{r2} | {roc_auc} | `{params}` | {feature_count} |".format(
                rank=rank,
                target=row["target_key"],
                model=row["model"],
                time=row["training_time"],
                rmse=row["rmse"],
                mae=row["mae"],
                mape=row["mape"],
                r2=row["r2"],
                roc_auc=row["roc_auc"],
                params=params,
                feature_count=row["feature_count"],
            )
        )

    (docs_dir / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_importance_report(
    feature_selection_reports: Dict[str, Dict[str, Any]],
    docs_dir: Path,
) -> None:
    """Write feature-selection and importance results."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature Importance Report",
        "",
        "Feature subsets were compared using mutual information, permutation importance, and recursive feature elimination.",
        "A subset is used only when it improves chronological validation performance over the all-feature baseline.",
        "",
    ]
    for target_key, report in feature_selection_reports.items():
        lines.extend([
            f"## {target_key}",
            "",
            f"- Selected method: `{report.get('selected_method')}`",
            f"- Feature count: {len(report.get('selected_features', []))}",
            f"- Baseline score: {report.get('baseline_score')}",
            f"- Selected score: {report.get('selected_score')}",
            "",
            "| Feature | Importance | Method |",
            "| --- | ---: | --- |",
        ])
        for item in report.get("importances", [])[:25]:
            lines.append(
                f"| {item.get('feature')} | {item.get('importance')} | {item.get('method')} |"
            )
        lines.append("")

    (docs_dir / "feature_importance_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_research_notes(all_results: Dict[str, Dict[str, Any]], docs_dir: Path) -> None:
    """Generate concise research notes for interview/demo use."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    best_forecast = all_results.get("shi_30d") or all_results.get("shi_7d") or all_results.get("shi_24h")
    best_model = best_forecast.get("best_algorithm", "the selected ensemble") if best_forecast else "the selected ensemble"
    notes = f"""# Research Notes

## Why Forecasting Is Better Than Simple Prediction

Predicting the current Structural Health Index mostly describes the bridge state that sensors already imply. Forecasting SHI at 24 hours, 7 days, and 30 days turns the system into a decision-support tool: maintenance teams can prioritize interventions before a risk threshold is crossed.

## Why TimeSeriesSplit Was Used

Structural health data is ordered in time. Random splits can train on future observations and validate on past observations, which inflates performance. TimeSeriesSplit preserves chronological order so every validation fold is evaluated on data later than its training fold.

## Why SHAP Was Selected

SHAP gives local, feature-level contribution scores for tree models and supports deterministic natural language explanations. That makes each prediction auditable for engineers who need to know which sensor patterns drove a risk estimate.

## Model Findings

The current best forecast family is `{best_model}` for the prioritized forecast target. Tree ensembles are appropriate here because they handle nonlinear interactions such as humidity-corrosion, traffic-load, and wind-deflection effects without requiring a rigid linear specification.

## Why XGBoost Can Outperform Random Forest

When XGBoost wins, it is usually because boosting corrects residual errors stage by stage and captures subtle degradation patterns more efficiently than independent bagged trees. Random Forest remains a strong baseline because it is stable and less sensitive to hyperparameters.

## Limitations

- Forecast quality depends on whether the provided future SHI labels were generated from real inspections or simulation.
- The API handles single readings; richer confidence estimates would improve with recent sensor history.
- Rule-based recommendations are deterministic and auditable, but should be reviewed by a qualified bridge engineer before operational use.

## Future Work

- Add sequence models or probabilistic forecasting once longer bridge-specific histories are available.
- Calibrate failure probabilities with reliability diagrams.
- Store per-bridge model drift metrics and trigger retraining when feature distributions shift.
"""
    (docs_dir / "research_notes.md").write_text(notes, encoding="utf-8")


def write_model_metadata(
    models_dir: Path,
    all_results: Dict[str, Dict[str, Any]],
    feature_sets: Dict[str, List[str]],
    dataset_fingerprint: str,
    training_timestamp: str,
    elapsed_seconds: float,
) -> None:
    """Persist model metadata required for inference confidence and auditability."""
    metadata = {
        "training_timestamp": training_timestamp,
        "dataset_version": dataset_fingerprint,
        "training_time_seconds": elapsed_seconds,
        "models": {},
    }
    for model_key, result in all_results.items():
        selected_features = feature_sets.get(model_key, [])
        best_algorithm = result.get("best_algorithm")
        best_info = result.get("training_info", {}).get("results", {}).get(best_algorithm, {})
        metadata["models"][model_key] = {
            "target": result.get("target"),
            "task": result.get("task"),
            "algorithm": best_algorithm,
            "feature_count": len(selected_features),
            "feature_list": selected_features,
            "training_time": best_info.get("train_time_s"),
            "best_parameters": best_info.get("params", {}),
            "cross_validation_score": best_info.get("cv_score"),
            "evaluation_metrics": result.get("metrics", {}),
            "feature_selection": result.get("feature_selection", {}),
        }
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Full ML training pipeline.

    Steps:
      1. Load config & dataset
      2. Validate data
      3. Feature engineering
      4. Extract targets & build feature matrix
      5. Preprocessing (fit on train, transform on test)
      6. Train multiple models per target
      7. Evaluate on test set
      8. Fit SHAP explainers
      9. Save all artifacts

    Returns:
        Summary dict of results.
    """
    start_time = time.time()
    config = load_config(config_path)
    ml_cfg = config.get("ml", {})
    models_dir = Path(ml_cfg.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path("docs")

    # ── Step 1: Load dataset ──────────────────────────────────────────── #
    df = load_dataset(config)

    # ── Step 2: Validate ─────────────────────────────────────────────── #
    validator = DataValidator(config)
    report = validator.validate(df)
    if not report.is_valid:
        logger.error(f"Data validation failed: {report.errors}")
        raise ValueError(f"Data validation failed: {report.errors}")
    logger.info(f"Validation stats: {report.stats}")

    # ── Step 3: Feature Engineering ───────────────────────────────────── #
    feature_engineer = FeatureEngineer(config)
    df_engineered = feature_engineer.fit_transform(df)
    logger.info(f"After feature engineering: {df_engineered.shape}")

    # ── Step 4: Extract targets ───────────────────────────────────────── #
    targets = build_target_series(df_engineered, config)
    logger.info(f"Targets available: {list(targets.keys())}")

    # ── Step 5: Build feature matrix (remove leakage) ─────────────────── #
    X = remove_target_leakage(df_engineered, config)

    # Time-aware train/test split (last 20% = test, respects temporal order)
    test_size = ml_cfg.get("test_size", 0.20)
    split_idx = int(len(X) * (1 - test_size))
    X_train_raw = X.iloc[:split_idx]
    X_test_raw = X.iloc[split_idx:]

    # ── Step 6: Preprocessing ─────────────────────────────────────────── #
    preprocessor = Preprocessor(config)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Get feature columns after preprocessing
    num_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns]
    feature_columns = [
        c for c in num_cols
        if c not in list(config.get("targets", {}).values())
    ]
    X_train_feat = X_train[feature_columns].reset_index(drop=True)
    X_test_feat = X_test[feature_columns] if all(c in X_test.columns for c in feature_columns) else X_test

    # Ensure feature columns align between train/test
    for col in feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test_feat = X_test[feature_columns].reset_index(drop=True)

    # Save feature columns list
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    logger.info(f"Feature matrix: {X_train_feat.shape} train / {X_test_feat.shape} test")

    # ── Step 7: Train models ───────────────────────────────────────────── #
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator()
    feature_selector = FeatureSelector(config)
    all_results: Dict[str, Dict] = {}
    target_feature_columns: Dict[str, List[str]] = {}
    feature_selection_reports: Dict[str, Dict[str, Any]] = {}

    train_targets = build_training_targets(targets, config)
    logger.info(f"Training target priority: {list(train_targets.keys())}")

    for model_key, (target_col, task) in train_targets.items():
        if model_key not in targets:
            logger.warning(f"Target '{model_key}' not found — skipping")
            continue

        y = targets[model_key]
        y_train = y.iloc[:split_idx].reset_index(drop=True)
        y_test = y.iloc[split_idx:].reset_index(drop=True)

        # Drop NaN rows
        train_mask = y_train.notna()
        test_mask = y_test.notna()

        y_tr = y_train[train_mask]
        X_tr = X_train_feat[train_mask]
        y_te = y_test[test_mask]
        X_te = X_test_feat[test_mask]

        if task == "classification" and y_tr.nunique() < 2:
            logger.warning("Skipping '%s': classification target has one class", model_key)
            continue

        selection = feature_selector.select(X_tr, y_tr, task, target_col)
        selected_features = selection.selected_features
        target_feature_columns[model_key] = selected_features
        feature_selection_reports[model_key] = selection.to_dict()
        X_tr_selected = X_tr[selected_features].reset_index(drop=True)
        X_te_selected = X_te[selected_features].reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        y_te = y_te.reset_index(drop=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"Training → {model_key} ({task}): {len(X_tr):,} samples")

        # Train
        best_model, train_info = trainer.train_all(
            X_tr_selected,
            y_tr,
            target_col,
            task,
            X_eval=X_te_selected,
            y_eval=y_te,
        )

        # Evaluate
        if task == "regression":
            metrics = evaluator.evaluate_regression(best_model, X_te_selected, y_te, target_col)
        else:
            metrics = evaluator.evaluate_classification(best_model, X_te_selected, y_te, target_col)

        all_results[model_key] = {
            "target": target_col,
            "task": task,
            "best_algorithm": train_info["best_model_name"],
            "metrics": metrics,
            "training_info": train_info,
            "feature_count": len(selected_features),
            "feature_selection": selection.to_dict(),
        }

        # Save model
        trainer.save_model(best_model, f"model_{model_key}")

        # ── Step 8: SHAP Explainer ─────────────────────────────────────── #
        explainer = Explainer(config)
        explainer.fit(best_model, X_tr_selected)
        joblib.dump(explainer, models_dir / f"explainer_{model_key}.joblib")
        logger.info(f"SHAP explainer saved for '{model_key}'")

    # ── Step 9: Save preprocessor, feature engineer, metadata ─────────── #
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(feature_engineer, models_dir / "feature_engineer.joblib")

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    (models_dir / "model_version.txt").write_text(version)

    with open(models_dir / "training_results.json", "w") as f:
        # Convert non-serializable objects to strings for JSON
        def make_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return str(obj)
        json.dump(all_results, f, indent=2, default=make_serializable)

    elapsed = round(time.time() - start_time, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Training complete in {elapsed}s. Version: {version}")
    logger.info(f"Results saved → {models_dir}/training_results.json")

    return {
        "version": version,
        "elapsed_seconds": elapsed,
        "models_trained": list(all_results.keys()),
        "results": all_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BridgeGuardian AI Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    results = train_pipeline(args.config)
    print(json.dumps({k: v for k, v in results.items() if k != "results"}, indent=2))
