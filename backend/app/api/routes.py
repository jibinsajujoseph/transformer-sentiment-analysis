"""
API routes for the Sentiment Model Arena.
"""

from fastapi import APIRouter, HTTPException

from app.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a movie review using both sentiment models.

    Runs inference on the custom scratch Transformer and the
    fine-tuned DistilRoBERTa model, returning predictions,
    confidence scores, latency, and attention weights.
    """
    # Import here to avoid circular imports — services are initialized at startup
    from app.main import distilroberta_service, scratch_service

    if scratch_service is None or distilroberta_service is None:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please try again shortly.",
        )

    try:
        scratch_result = scratch_service.predict(request.text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scratch Transformer inference failed: {e}",
        )

    try:
        distilroberta_result = distilroberta_service.predict(request.text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DistilRoBERTa inference failed: {e}",
        )

    return AnalyzeResponse(
        scratch=scratch_result,
        distilroberta=distilroberta_result,
    )
