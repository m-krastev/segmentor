from dataclasses import dataclass
from math import sqrt
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .rope import RotaryEmbedding


@dataclass
class ModelArgs:
    input_dim: int = 3
    dim: int = 64  # increase later
    n_layers: int = 12
    n_heads: int = 8

    n_kv_heads: Optional[int] = None

    # Optionally, we can also consider a "vocabulary" of the segments of the small bowel instead of relying on a fixed input_dim transformation
    vocab_size: int = -1

    norm_eps: float = 1e-5
    rope_theta: float = 10000
    p_dropout: float = 0.1
    ff_dropout: float = 0.1
    attn_dropout: float = 0.1
    emb_dropout: float = 0.1

    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash_attn: bool = False


class AttentionCell(nn.Module):
    """
    Attention cell for the point-cloud autoregressive model, complete with RoPE embeddings.
    """

    def __init__(self, config: ModelArgs):
        super(AttentionCell, self).__init__()

        self.config = config
        self.Wqkv = nn.Linear(config.dim, config.dim * 3)
        self.Wo = nn.Linear(config.dim, config.dim)
        self.layer_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.rope = RotaryEmbedding(
            config.dim,
            theta=config.rope_theta,
        )

        # Mask for the attention
        self.register_buffer(
            "attn_mask",
            torch.triu(
                torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool).unsqueeze(0).unsqueeze(0),
                diagonal=1,
            ),
        )

        # Additional dropout for attention because data is limited
        self.attn_dropout = nn.Dropout(config.attn_dropout)

        # Dropout for the output
        self.dropout = nn.Dropout(config.p_dropout)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.size()

        qkv = self.Wqkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q, k = self.rope.rotate_queries_and_keys(q, k, seq_len)

        q = q.reshape(
            batch_size,
            seq_len,
            self.config.n_heads,
            self.config.dim // self.config.n_heads,
        )
        k = k.reshape(
            batch_size,
            seq_len,
            self.config.n_heads,
            self.config.dim // self.config.n_heads,
        )
        v = v.reshape(
            batch_size,
            seq_len,
            self.config.n_heads,
            self.config.dim // self.config.n_heads,
        )

        attn_mask = self.attn_mask[:,:, :seq_len, :seq_len]
        if self.config.flash_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                mask=attn_mask,
                dropout=self.attn_dropout if self.training else 0,
            )
        else:
            out = q @ k.transpose(-2, -1) / sqrt(k.size(-1))
            out = out.masked_fill(attn_mask, -torch.inf)
            out = F.softmax(out, dim=-1)
            out = F.dropout(out, p=self.attn_dropout, training=self.training)
            out = out @ v

        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.config.dim)
        out = self.Wo(out)
        return self.layer_norm(self.dropout(out))


class AttentionBlock(nn.Module):
    """
    Attention module for the point-cloud autoregressive model.
    """

    def __init__(self, config: ModelArgs):
        super(AttentionBlock, self).__init__()

        self.attn = AttentionCell(config)
        self.layer_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim),
            nn.GELU(),
            nn.Linear(4 * config.dim, config.dim),
        )

    def forward(self, x):
        x = self.attn(self.layer_norm(x)) + x
        x = self.mlp(self.layer_norm(x)) + x
        return x


class Model(nn.Module):
    """
    Attention-based point-cloud autoregressive model for modelling small bowel structure. This is the basic input model.
    """

    def __init__(self, config: ModelArgs):
        super(Model, self).__init__()
        self.input = nn.Linear(config.input_dim, config.dim)
        self.attention = nn.Sequential(
            *[AttentionBlock(config) for _ in range(config.n_layers)]
        )
        self.out = nn.Linear(config.dim, config.input_dim)

    def forward(self, x):
        x = self.input(x)
        x = self.attention(x)
        x = self.out(x)
        # The output is then a prediction for the coordinates of the next point in the sequence (we could do L2 loss here?)
        return x
