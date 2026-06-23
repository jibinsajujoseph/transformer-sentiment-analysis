"""
Tests for the DistilRoBERTa HuggingFace wrapper.

Validates:
    1. Model and tokenizer download and load successfully from Hugging Face Hub
    2. Inference on positive/negative reviews returns valid outputs
"""

import pytest

from app.config import settings
from app.services.distilroberta_service import DistilRoBERTaService


class TestModelLoading:
    """Test that the model loads correctly."""

    def test_service_initializes(self) -> None:
        """Service should initialize without errors."""
        service = DistilRoBERTaService(settings.DISTILROBERTA_REPO_ID)
        assert service.model is not None
        assert service.tokenizer is not None


class TestLabelMapping:
    """Test that the label mapping is correct."""

    @pytest.fixture(scope="class")
    def service(self) -> DistilRoBERTaService:
        return DistilRoBERTaService(settings.DISTILROBERTA_REPO_ID)

    def test_label_count(self, service: DistilRoBERTaService) -> None:
        assert len(service.model.config.id2label) == 2

    def test_label_values(self, service: DistilRoBERTaService) -> None:
        # Depending on how it was trained, it might be 0/1 or LABEL_0/LABEL_1
        assert "positive" in (service.id2label[0].lower(), service.id2label[1].lower())


class TestInference:
    """Test end-to-end inference via the service."""

    @pytest.fixture(scope="class")
    def service(self) -> DistilRoBERTaService:
        return DistilRoBERTaService(settings.DISTILROBERTA_REPO_ID)

    def test_positive_review(self, service: DistilRoBERTaService) -> None:
        result = service.predict("This movie was absolutely wonderful!")
        assert result.label in ("positive", "negative")
        assert 0.0 <= result.confidence <= 1.0

    def test_negative_review(self, service: DistilRoBERTaService) -> None:
        result = service.predict("Terrible film, completely boring and predictable.")
        assert result.label in ("positive", "negative")
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_range(self, service: DistilRoBERTaService) -> None:
        result = service.predict("It was an okay movie.")
        assert 0.0 <= result.confidence <= 1.0

    def test_latency_measured(self, service: DistilRoBERTaService) -> None:
        result = service.predict("Test")
        assert result.latency_ms > 0

    def test_short_input(self, service: DistilRoBERTaService) -> None:
        result = service.predict("Bad")
        assert result.label in ("positive", "negative")

    def test_long_input(self, service: DistilRoBERTaService) -> None:
        # Just ensure it doesn't crash on long inputs (should be truncated to 512 max length)
        long_text = "movie " * 1000
        result = service.predict(long_text)
        assert result.label in ("positive", "negative")
