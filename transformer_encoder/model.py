"""Transformer encoder implemented from elementary PyTorch operations."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention with inspectable attention probabilities."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        batch, length, _ = x.shape
        def split_heads(projection):
            return projection(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(self.q_proj), split_heads(self.k_proj), split_heads(self.v_proj)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            scores = scores.masked_fill(~padding_mask[:, None, None, :], -1e9)
        attention = F.softmax(scores, dim=-1)
        self.last_attention = attention.detach()
        context = self.dropout(attention) @ v
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(context)


class EncoderBlock(nn.Module):
    """Pre-normalized attention and feed-forward residual sublayers."""
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        x = x + self.dropout(self.attention(self.norm1(x), padding_mask))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class TransformerClassifier(nn.Module):
    """Learned token/position embeddings, encoder stack, masked pooling, classifier."""
    def __init__(self, vocab_size, max_length=14, d_model=64, num_heads=4, num_layers=2, ff_dim=128, num_classes=2, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids):
        padding_mask = token_ids.ne(0)
        positions = torch.arange(token_ids.size(1), device=token_ids.device)[None, :]
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.final_norm(x)
        pooled = (x * padding_mask.unsqueeze(-1)).sum(1) / padding_mask.sum(1, keepdim=True).clamp_min(1)
        return self.classifier(pooled)

    def attention_maps(self):
        return [block.attention.last_attention for block in self.blocks]
