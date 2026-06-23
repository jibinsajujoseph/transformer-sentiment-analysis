"""
Inference service for the custom scratch Transformer model.

Handles:
    - Model + vocabulary loading from disk
    - Text preprocessing and tokenization
    - Forward inference with latency measurement
    - Attention weight extraction via forward hooks
"""

import json
import math
import pickle
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from app.models.scratch_transformer import (
    MultiHeadAttention,
    SentimentClassifier,
    scaled_dot_product_attention,
)
from app.schemas import ScratchResult


# Label mapping matches the notebook
LABELS: dict[int, str] = {0: "negative", 1: "positive"}


class Vocabulary:
    """
    Minimal vocabulary class for deserialization.

    The pickled vocab.pkl contains an instance of the Vocabulary class
    from the training notebook. We need compatible attributes:
        - word2idx: dict[str, int]
        - idx2word: dict[int, str]
        - size: int
    """

    word2idx: dict[str, int]
    idx2word: dict[int, str]
    size: int

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        """Convert token list → padded integer list of length max_len."""
        ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]  # 1 = UNK
        ids += [0] * (max_len - len(ids))  # 0 = PAD
        return ids


# Hack to allow pickle to load the Vocabulary class which was saved
# in a Jupyter notebook (where the module is __main__)
import __main__
__main__.Vocabulary = Vocabulary


def preprocess(text: str) -> list[str]:
    """
    Clean and tokenize a review into word tokens.

    Matches the exact preprocessing from the training notebook:
        1. Lowercase
        2. Strip HTML tags
        3. Keep only alphanumeric characters and spaces
        4. Collapse whitespace
        5. Split on whitespace
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text.split()


class ScratchTransformerService:
    """
    Service for loading and running inference with the custom Transformer.

    Initialized once at app startup — holds model and vocab in memory.
    """

    def __init__(self, model_dir: Path) -> None:
        """
        Load the model config, vocabulary, and weights from disk.

        Args:
            model_dir: Path to models/scratch-transformer/ containing
                       model_config.json, vocab.pkl, sentiment_transformer.pt
        """
        self.device = torch.device("cpu")

        # 1. Load config
        config_path = model_dir / "model_config.json"
        with open(config_path) as f:
            self.config: dict = json.load(f)

        self.max_len: int = self.config["max_len"]

        # 2. Load vocabulary
        vocab_path = model_dir / "vocab.pkl"
        with open(vocab_path, "rb") as f:
            self.vocab: Vocabulary = pickle.load(f)

        # 3. Rebuild model and load weights
        self.model = SentimentClassifier(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            num_heads=self.config["num_heads"],
            ff_dim=self.config["ff_dim"],
            num_blocks=self.config["num_blocks"],
            max_len=self.config["max_len"] + 10,  # matches notebook's max_len + 10
            num_classes=self.config["num_classes"],
            dropout=self.config["dropout"],
        )

        weights_path = model_dir / "sentiment_transformer.pt"
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> ScratchResult:
        """
        Run inference on a single review text.

        Returns a ScratchResult with label, confidence, latency,
        tokens (non-PAD only), and per-token attention weights.
        """
        # Tokenize
        tokens = preprocess(text)
        token_ids = self.vocab.encode(tokens, self.max_len)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Count non-PAD tokens for output
        num_real_tokens = min(len(tokens), self.max_len)
        display_tokens = tokens[:num_real_tokens]

        # Set up attention extraction hook on last block
        attention_maps: list[torch.Tensor] = []

        def hook_fn(
            module: MultiHeadAttention,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            """Recompute attention weights inside the hook."""
            x = input[0]
            Q = module.split_heads(module.W_q(x))
            K = module.split_heads(module.W_k(x))
            V = module.split_heads(module.W_v(x))
            _, attn = scaled_dot_product_attention(Q, K, V)
            attention_maps.append(attn.detach().cpu())

        last_block_idx = self.config["num_blocks"] - 1
        handle = self.model.encoder.blocks[last_block_idx].attention.register_forward_hook(
            hook_fn
        )

        # Inference with timing
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_tensor)  # (1, 2)
            probs = F.softmax(logits, dim=-1)[0]  # (2,)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Clean up hook
        handle.remove()

        # Extract prediction
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

        # Extract attention weights
        # attention_maps[0]: (1, num_heads, seq_len, seq_len)
        # Average across heads → (seq_len, seq_len)
        # Then average across query dimension → per-token attention (seq_len,)
        attn_weights = attention_maps[0][0]  # (num_heads, S, S)
        attn_avg = attn_weights.mean(dim=0)  # (S, S) — average across heads
        per_token_attention = attn_avg.mean(dim=0)  # (S,) — average across queries

        # Keep only non-PAD tokens
        per_token_attention = per_token_attention[:num_real_tokens].tolist()

        # Normalize to [0, 1] range
        if per_token_attention:
            min_val = min(per_token_attention)
            max_val = max(per_token_attention)
            range_val = max_val - min_val
            if range_val > 0:
                per_token_attention = [
                    (v - min_val) / range_val for v in per_token_attention
                ]

        return ScratchResult(
            label=LABELS[pred_class],
            confidence=round(confidence, 4),
            latency_ms=round(elapsed_ms, 2),
            tokens=display_tokens,
            attention=[round(w, 4) for w in per_token_attention],
        )
