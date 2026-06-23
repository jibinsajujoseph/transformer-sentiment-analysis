"""
Model manager for downloading, caching, and loading models from Hugging Face Hub.
"""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.config import settings
from app.services.distilroberta_service import DistilRoBERTaService
from app.services.scratch_service import ScratchTransformerService

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages downloading missing model artifacts from Hugging Face Hub,
    loading both models into memory, and exposing them for inference.
    """

    def __init__(self) -> None:
        self.scratch_service: ScratchTransformerService | None = None
        self.distilroberta_service: DistilRoBERTaService | None = None

    def initialize_models(self) -> None:
        """
        Download required artifacts and initialize the model services.
        This blocks until all files are available and loaded.
        """
        try:
            logger.info("Downloading missing artifacts...")
            
            # 1. Download/Cache Scratch Transformer files
            scratch_config_path = hf_hub_download(
                repo_id=settings.SCRATCH_REPO_ID,
                filename="model_config.json"
            )
            scratch_vocab_path = hf_hub_download(
                repo_id=settings.SCRATCH_REPO_ID,
                filename="vocab.pkl"
            )
            scratch_weights_path = hf_hub_download(
                repo_id=settings.SCRATCH_REPO_ID,
                filename="sentiment_transformer.pt"
            )

            # 2. Load Scratch Transformer
            logger.info("Loading Scratch Transformer...")
            self.scratch_service = ScratchTransformerService(
                config_path=Path(scratch_config_path),
                vocab_path=Path(scratch_vocab_path),
                weights_path=Path(scratch_weights_path)
            )

            # 3. Load DistilRoBERTa (downloads happen inside the service via HF transformers)
            logger.info("Loading DistilRoBERTa...")
            self.distilroberta_service = DistilRoBERTaService(
                repo_id=settings.DISTILROBERTA_REPO_ID
            )

            logger.info("Models loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def get_scratch_service(self) -> ScratchTransformerService:
        """Return the initialized scratch service."""
        if not self.scratch_service:
            raise RuntimeError("Scratch service not initialized.")
        return self.scratch_service

    def get_distilroberta_service(self) -> DistilRoBERTaService:
        """Return the initialized distilroberta service."""
        if not self.distilroberta_service:
            raise RuntimeError("DistilRoBERTa service not initialized.")
        return self.distilroberta_service
