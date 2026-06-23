"""
Inference service for the fine-tuned DistilRoBERTa model.

Handles:
    - Model + tokenizer loading via HuggingFace
    - Text tokenization with RoBERTa BPE tokenizer
    - Forward inference with latency measurement
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from app.models.distilroberta import load_distilroberta
from app.schemas import DistilRoBERTaResult


class DistilRoBERTaService:
    """
    Service for loading and running inference with DistilRoBERTa.

    Initialized once at app startup — holds model and tokenizer in memory.
    """

    def __init__(self, repo_id: str) -> None:
        """
        Load the fine-tuned DistilRoBERTa model and tokenizer.

        Args:
            repo_id: The Hugging Face repository ID.
        """
        self.device = torch.device("cpu")
        self.model, self.tokenizer = load_distilroberta(repo_id)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping from model config
        self.id2label: dict[int, str] = self.model.config.id2label

    def predict(self, text: str) -> DistilRoBERTaResult:
        """
        Run inference on a single review text.

        Args:
            text: Movie review text to classify.

        Returns:
            DistilRoBERTaResult with label, confidence, and latency.
        """
        # Tokenize with the BPE tokenizer
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference with timing
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, 2)
            probs = F.softmax(logits, dim=-1)[0]  # (2,)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract prediction
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
        label = self.id2label[pred_class].lower()

        return DistilRoBERTaResult(
            label=label,
            confidence=round(confidence, 4),
            latency_ms=round(elapsed_ms, 2),
        )
