"""
DistilRoBERTa model wrapper.

Uses HuggingFace AutoModelForSequenceClassification to load the
fine-tuned DistilRoBERTa model from local artifacts.

No custom architecture needed — the model is a standard
RobertaForSequenceClassification with 2 labels (Negative, Positive).
"""

from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)


def load_distilroberta(
    repo_id: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Load the fine-tuned DistilRoBERTa model and tokenizer from Hugging Face Hub.

    Args:
        repo_id: The Hugging Face repository ID (e.g., "jibinsajujoseph/distilroberta-imdb")

    Returns:
        Tuple of (model, tokenizer), both ready for inference.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
    )

    return model, tokenizer
