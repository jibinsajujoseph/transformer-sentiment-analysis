"""
Pydantic request/response schemas for the Sentiment Model Arena API.
"""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Movie review text to analyze",
        examples=["This movie was absolutely fantastic! The acting was superb."],
    )


class ScratchResult(BaseModel):
    """Inference result from the custom scratch Transformer model."""

    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence (0-1)"
    )
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in ms")
    tokens: list[str] = Field(
        ..., description="Tokenized input (non-PAD tokens only)"
    )
    attention: list[float] = Field(
        ...,
        description="Per-token attention weights (head-averaged, normalized)",
    )


class DistilRoBERTaResult(BaseModel):
    """Inference result from the fine-tuned DistilRoBERTa model."""

    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence (0-1)"
    )
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in ms")


class AnalyzeResponse(BaseModel):
    """Combined response from both models."""

    scratch: ScratchResult
    distilroberta: DistilRoBERTaResult
