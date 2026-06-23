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
    model_dir: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Load the fine-tuned DistilRoBERTa model and tokenizer from disk.

    Args:
        model_dir: Path to the model directory containing config.json,
                   model.safetensors, and tokenizer/ subdirectory.

    Returns:
        Tuple of (model, tokenizer), both ready for inference.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir / "tokenizer"),
        local_files_only=True,
    )

    return model, tokenizer
