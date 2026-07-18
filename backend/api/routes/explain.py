"""BridgeGuardian AI — /explain endpoint"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, HTTPException
from backend.schemas.request import ExplainRequest
from backend.schemas.response import ExplainResponse, GlobalImportanceItem, ShapExplanation

logger = logging.getLogger("bridgeguardian.api.explain")
router = APIRouter()


def get_pipeline():
    from backend.main import inference_pipeline
    return inference_pipeline


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain a prediction with SHAP",
    tags=["Explainability"],
)
async def explain(
    request: ExplainRequest,
    pipeline=Depends(get_pipeline),
) -> ExplainResponse:
    """
    Returns SHAP-based explanation for a prediction:
    - Feature importances (positive & negative contributions)
    - Base value and prediction contribution
    - Top influencing features
    """
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready. Train first.")

    input_dict = request.input_data.model_dump(exclude_none=False)
    input_dict = {k: (v if v is not None else 0.0) for k, v in input_dict.items()}

    try:
        explanation_raw = pipeline.explain(input_dict, target=request.target)
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    explanation = ShapExplanation(
        base_value=explanation_raw.get("base_value", 0.0),
        shap_values=explanation_raw.get("shap_values", []),
        feature_names=explanation_raw.get("feature_names", []),
        feature_importances=explanation_raw.get("feature_importances", []),
        top_positive_features=explanation_raw.get("top_positive_features", []),
        top_negative_features=explanation_raw.get("top_negative_features", []),
        prediction_contribution=explanation_raw.get("prediction_contribution", 0.0),
        note=explanation_raw.get("note"),
    )

    return ExplainResponse(target=request.target, explanation=explanation)


@router.get(
    "/explain/global/{target}",
    summary="Get global feature importance",
    tags=["Explainability"],
)
async def global_importance(
    target: str = "health_score",
    pipeline=Depends(get_pipeline),
):
    """Returns mean absolute SHAP values across training samples."""
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready.")
    importances = pipeline.get_global_importance(target)
    return {"target": target, "importances": importances}
