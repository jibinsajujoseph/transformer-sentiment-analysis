"""
Custom Transformer architecture for sentiment classification.

Reconstructed verbatim from the training notebook to guarantee
weight compatibility with sentiment_transformer.pt.

Architecture:
    TokenEmbedding → PositionalEncoding → N × TransformerBlock → MeanPool → Classifier

Each TransformerBlock:
    MultiHeadAttention (Add & Norm) → FeedForward (Add & Norm)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor (batch, heads, seq_len, d_k)
        K: Key tensor (batch, heads, seq_len, d_k)
        V: Value tensor (batch, heads, seq_len, d_v)
        mask: Boolean mask (batch, 1, 1, seq_len) — True where PAD

    Returns:
        output: Weighted values (batch, heads, seq_len, d_v)
        attention_weights: Softmax weights (batch, heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # Compute scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, S, S)

    # Scale
    scores = scores / math.sqrt(d_k)

    # Apply mask (set PAD positions to -inf so softmax ignores them)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax over key dimension
    attention_weights = F.softmax(scores, dim=-1)  # (B, H, S, S)

    # Weighted sum of values
    output = torch.matmul(attention_weights, V)  # (B, H, S, d_v)

    return output, attention_weights


class TokenEmbedding(nn.Module):
    """
    Token embedding with √d_model scaling.

    Scaling trick from 'Attention Is All You Need' to prevent
    embeddings from being drowned out by positional encodings.
    """

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) → (batch, seq_len, embed_dim)"""
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — fixed, not learned.
    Adds position information to token embeddings.
    """

    def __init__(
        self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build PE matrix: shape (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # Compute the division term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)  # even indices
        pe[:, 1::2] = torch.cos(pos * div_term)  # odd indices

        # Register as buffer (not a parameter — won't be updated during training)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim) → (batch, seq_len, embed_dim)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with separate Q/K/V projections."""

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # dimension per head

        # Separate projection matrices for Q, K, V (no bias on Q/K/V)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, seq_len, embed_dim)
            → (batch, num_heads, seq_len, d_k)
        """
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (B, H, S, d_k)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Padding mask (batch, 1, 1, seq_len) — True where PAD

        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        B, S, _ = x.shape

        # Project
        Q = self.split_heads(self.W_q(x))  # (B, H, S, d_k)
        K = self.split_heads(self.W_k(x))  # (B, H, S, d_k)
        V = self.split_heads(self.W_v(x))  # (B, H, S, d_k)

        # Attend
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        # attn_out: (B, H, S, d_k)

        # Merge heads: (B, H, S, d_k) → (B, S, embed_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(B, S, self.embed_dim)

        # Output projection
        output = self.W_o(attn_out)
        output = self.dropout(output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear → ReLU → Dropout → Linear."""

    def __init__(
        self, embed_dim: int, ff_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim) → (batch, seq_len, embed_dim)"""
        x = F.relu(self.linear1(x))  # → (batch, seq_len, ff_dim)
        x = self.dropout(x)
        x = self.linear2(x)  # → (batch, seq_len, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.
    Attention + Add&Norm → FeedForward + Add&Norm
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

        # LayerNorm normalizes across the embedding dimension
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, embed_dim)
            mask: Padding mask (batch, 1, 1, seq_len)

        Returns:
            Output (batch, seq_len, embed_dim)
        """
        # Sub-layer 1: Multi-Head Self-Attention + Residual + LayerNorm
        attn_out = self.attention(x, mask=mask)
        x = self.norm1(x + attn_out)  # Add & Norm

        # Sub-layer 2: Feed-Forward + Residual + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)  # Add & Norm

        return x


class TransformerEncoder(nn.Module):
    """
    Stacked Transformer encoder.
    Embedding → PositionalEncoding → N × TransformerBlock
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_blocks: int,
        max_len: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, embed_dim, padding_idx)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.padding_idx = padding_idx

    def make_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a boolean mask where True = PAD (should be ignored).
        Shape: (batch, 1, 1, seq_len) — broadcasts over heads and query positions.
        """
        return (x == self.padding_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Integer token IDs (batch, seq_len)

        Returns:
            Token representations (batch, seq_len, embed_dim)
        """
        mask = self.make_padding_mask(x)  # (B, 1, 1, S)

        x = self.embedding(x)  # (B, S, E)
        x = self.pos_enc(x)  # (B, S, E)

        for block in self.blocks:
            x = block(x, mask=mask)  # (B, S, E)

        return x  # (B, S, E)


class SentimentClassifier(nn.Module):
    """
    Full sentiment classification model.
    TransformerEncoder → Masked Mean Pooling → Dropout → Linear Classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_blocks: int,
        max_len: int,
        num_classes: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, embed_dim, num_heads, ff_dim,
            num_blocks, max_len, dropout, padding_idx,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

    def mean_pool(
        self, token_embeddings: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Average token embeddings, ignoring PAD positions.

        Args:
            token_embeddings: (batch, seq_len, embed_dim)
            input_ids: (batch, seq_len)

        Returns:
            Pooled representation (batch, embed_dim)
        """
        # Non-PAD mask: (B, S, 1)
        mask = (input_ids != self.padding_idx).unsqueeze(-1).float()
        # Zero out PAD positions
        masked = token_embeddings * mask
        # Sum / count
        summed = masked.sum(dim=1)  # (B, E)
        counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        return summed / counts  # (B, E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Integer token IDs (batch, seq_len)

        Returns:
            Logits (batch, num_classes)
        """
        token_out = self.encoder(x)  # (B, S, E)
        pooled = self.mean_pool(token_out, x)  # (B, E)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits
