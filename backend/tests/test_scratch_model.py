"""
Tests for the custom scratch Transformer model.

Validates:
    1. Config, vocabulary, and weights load successfully via Hugging Face Hub
    2. Model architecture matches the config
    3. Inference on a positive review returns valid output
    4. Inference on a negative review returns valid output
    5. Attention weights have the correct shape
"""

import json
import pickle
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download

from app.config import settings
from app.models.scratch_transformer import SentimentClassifier
from app.services.scratch_service import ScratchTransformerService, preprocess
from app.services.model_manager import ModelManager


@pytest.fixture(scope="module")
def downloaded_paths() -> dict[str, Path]:
    """Download artifacts once for the test module."""
    config_path = hf_hub_download(repo_id=settings.SCRATCH_REPO_ID, filename="model_config.json")
    vocab_path = hf_hub_download(repo_id=settings.SCRATCH_REPO_ID, filename="vocab.pkl")
    weights_path = hf_hub_download(repo_id=settings.SCRATCH_REPO_ID, filename="sentiment_transformer.pt")
    return {
        "config": Path(config_path),
        "vocab": Path(vocab_path),
        "weights": Path(weights_path),
    }


@pytest.fixture(scope="module")
def service(downloaded_paths: dict[str, Path]) -> ScratchTransformerService:
    """Initialize the service once for inference tests."""
    return ScratchTransformerService(
        config_path=downloaded_paths["config"],
        vocab_path=downloaded_paths["vocab"],
        weights_path=downloaded_paths["weights"],
    )


class TestModelLoading:
    """Test that all model artifacts load correctly from the Hub cache."""

    def test_config_loads(self, downloaded_paths: dict[str, Path]) -> None:
        """model_config.json should load and contain required keys."""
        with open(downloaded_paths["config"]) as f:
            config = json.load(f)

        required_keys = [
            "vocab_size", "embed_dim", "num_heads", "ff_dim",
            "num_blocks", "max_len", "num_classes", "dropout",
        ]
        for key in required_keys:
            assert key in config, f"Missing key '{key}' in model_config.json"

    def test_vocab_loads(self, downloaded_paths: dict[str, Path]) -> None:
        """vocab.pkl should load and contain word2idx, idx2word, and size."""
        with open(downloaded_paths["vocab"], "rb") as f:
            vocab = pickle.load(f)

        assert hasattr(vocab, "word2idx"), "Vocab missing word2idx"
        assert hasattr(vocab, "idx2word"), "Vocab missing idx2word"
        assert hasattr(vocab, "size"), "Vocab missing size"
        assert vocab.size > 0, "Vocab size should be > 0"

    def test_weights_load(self, downloaded_paths: dict[str, Path]) -> None:
        """sentiment_transformer.pt should load as a valid state dict."""
        state_dict = torch.load(downloaded_paths["weights"], map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict), "State dict should be a dict"
        assert len(state_dict) > 0, "State dict should not be empty"


class TestArchitecture:
    """Test that the reconstructed model matches the config."""

    @pytest.fixture()
    def config(self, downloaded_paths: dict[str, Path]) -> dict:
        with open(downloaded_paths["config"]) as f:
            return json.load(f)

    def test_model_construction(self, config: dict) -> None:
        """Model should instantiate without errors."""
        model = SentimentClassifier(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            num_blocks=config["num_blocks"],
            max_len=config["max_len"] + 10,
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )
        assert model is not None

    def test_block_count(self, config: dict) -> None:
        """Number of Transformer blocks should match config."""
        model = SentimentClassifier(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            num_blocks=config["num_blocks"],
            max_len=config["max_len"] + 10,
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )
        assert len(model.encoder.blocks) == config["num_blocks"]

    def test_weight_loading_matches(self, config: dict, downloaded_paths: dict[str, Path]) -> None:
        """Saved weights should load into the reconstructed model."""
        model = SentimentClassifier(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            num_blocks=config["num_blocks"],
            max_len=config["max_len"] + 10,
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )
        state_dict = torch.load(
            downloaded_paths["weights"],
            map_location="cpu",
            weights_only=True,
        )
        # This will raise if architecture doesn't match
        model.load_state_dict(state_dict)


class TestPreprocessing:
    """Test the text preprocessing function."""

    def test_basic_preprocessing(self) -> None:
        tokens = preprocess("This movie was great!")
        assert tokens == ["this", "movie", "was", "great"]

    def test_html_stripping(self) -> None:
        tokens = preprocess("<b>Amazing</b> film <br/>")
        assert "amazing" in tokens
        assert "<b>" not in " ".join(tokens)

    def test_special_characters(self) -> None:
        tokens = preprocess("Wow!!! 10/10 --- a must-see")
        # Only alphanumeric kept
        for token in tokens:
            assert token.isalnum(), f"Token '{token}' contains non-alphanumeric chars"


class TestInference:
    """Test end-to-end inference via the service."""

    def test_positive_review(self, service: ScratchTransformerService) -> None:
        """A clearly positive review should return a valid prediction."""
        result = service.predict(
            "This movie was absolutely fantastic! The acting was superb "
            "and the plot kept me engaged throughout."
        )
        assert result.label in ("positive", "negative")
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0
        assert len(result.tokens) > 0
        assert len(result.attention) == len(result.tokens)

    def test_negative_review(self, service: ScratchTransformerService) -> None:
        """A clearly negative review should return a valid prediction."""
        result = service.predict(
            "Terrible film. Boring, predictable, and a complete waste of time. "
            "I want my money back."
        )
        assert result.label in ("positive", "negative")
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0
        assert len(result.tokens) > 0

    def test_attention_shape(self, service: ScratchTransformerService) -> None:
        """Attention weights should match the number of non-PAD tokens."""
        result = service.predict("Great movie with wonderful acting")
        # tokens and attention should have the same length
        assert len(result.attention) == len(result.tokens)
        # All attention values should be in [0, 1] after normalization
        for w in result.attention:
            assert 0.0 <= w <= 1.0, f"Attention weight {w} out of [0, 1] range"

    def test_short_input(self, service: ScratchTransformerService) -> None:
        """Very short input should still work."""
        result = service.predict("Bad")
        assert result.label in ("positive", "negative")
        assert len(result.tokens) >= 1
