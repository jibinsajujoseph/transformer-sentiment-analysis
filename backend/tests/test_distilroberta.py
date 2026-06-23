"""
Tests for the fine-tuned DistilRoBERTa model.

Validates:
    1. Model and tokenizer load successfully
    2. Inference on a positive review returns valid output
    3. Inference on a negative review returns valid output
    4. Label mapping matches the expected values
"""

from pathlib import Path

import pytest

from app.services.distilroberta_service import DistilRoBERTaService

# Resolve model directory relative to this test file
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "distilroberta-imdb"


class TestModelLoading:
    """Test that model and tokenizer artifacts load correctly."""

    def test_model_dir_exists(self) -> None:
        """Model directory should exist with required files."""
        assert MODEL_DIR.exists(), f"Model dir not found at {MODEL_DIR}"
        assert (MODEL_DIR / "config.json").exists(), "config.json not found"
        assert (MODEL_DIR / "model.safetensors").exists(), "model.safetensors not found"
        assert (MODEL_DIR / "tokenizer").is_dir(), "tokenizer/ directory not found"

    def test_service_initializes(self) -> None:
        """Service should initialize without errors."""
        service = DistilRoBERTaService(MODEL_DIR)
        assert service.model is not None
        assert service.tokenizer is not None


class TestLabelMapping:
    """Test that label mapping matches expected values."""

    @pytest.fixture(scope="class")
    def service(self) -> DistilRoBERTaService:
        return DistilRoBERTaService(MODEL_DIR)

    def test_label_count(self, service: DistilRoBERTaService) -> None:
        """Should have exactly 2 labels."""
        assert len(service.id2label) == 2

    def test_label_values(self, service: DistilRoBERTaService) -> None:
        """Labels should be 'Negative' and 'Positive'."""
        labels = set(service.id2label.values())
        assert "Negative" in labels
        assert "Positive" in labels


class TestInference:
    """Test end-to-end inference via the service."""

    @pytest.fixture(scope="class")
    def service(self) -> DistilRoBERTaService:
        """Load the service once for all tests in this class."""
        return DistilRoBERTaService(MODEL_DIR)

    def test_positive_review(self, service: DistilRoBERTaService) -> None:
        """A clearly positive review should return a valid prediction."""
        result = service.predict(
            "This movie was absolutely fantastic! The acting was superb "
            "and the plot kept me engaged throughout."
        )
        assert result.label in ("positive", "negative")
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0

    def test_negative_review(self, service: DistilRoBERTaService) -> None:
        """A clearly negative review should return a valid prediction."""
        result = service.predict(
            "Terrible film. Boring, predictable, and a complete waste of time. "
            "I want my money back."
        )
        assert result.label in ("positive", "negative")
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0

    def test_confidence_range(self, service: DistilRoBERTaService) -> None:
        """Confidence should be between 0 and 1 for any input."""
        result = service.predict("An average film, nothing special.")
        assert 0.0 <= result.confidence <= 1.0

    def test_latency_measured(self, service: DistilRoBERTaService) -> None:
        """Latency should be a positive number."""
        result = service.predict("Good movie")
        assert result.latency_ms > 0

    def test_short_input(self, service: DistilRoBERTaService) -> None:
        """Very short input should still work."""
        result = service.predict("Bad")
        assert result.label in ("positive", "negative")

    def test_long_input(self, service: DistilRoBERTaService) -> None:
        """Long input should be truncated gracefully (max 512 tokens)."""
        long_review = "This is a great movie. " * 200
        result = service.predict(long_review)
        assert result.label in ("positive", "negative")
        assert 0.0 <= result.confidence <= 1.0
