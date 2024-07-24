# Date: 2024-07-21

import torch
from torch import nn
from torch.nn import functional as F

import math

"""
 Attention Module
  - Input: B, T, C
  - Config:
    - H: heads count
      - HS: head size
    - Emb: Embedding size
  - Output: B, T, C
  - Processes:
    - embeding((B, T)) -> (B, T, Emb)
    - Key: (B, T, Emb) -> Linear(Emb, HS) -> (B, T, HS)
    - Query: (B, T, Emb) -> Linear(Emb, HS) -> (B, T, HS)
    - Val: (B, T, Emb) -> Linear(Emb, HS) -> (B, T, HS)
"""


def apply_causal_mask(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, T)
     - each t-th only depending on the itself and before position
     - softmax => normalize to prob
    """
    B, T, T = x.size()
    tri = torch.tril(torch.ones((T, T)))
    tri = tri.to(device=x.device)
    wei = x.masked_fill(tri == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    return wei


def apply_causal_mask_v2(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, HC, T, T)
     - each t-th only depending on the itself and before position
     - softmax => normalize to prob
    """
    B, HC, T, T = x.size()
    tri = torch.tril(torch.ones((T, T)))
    tri = tri.to(device=x.device)
    wei = x.masked_fill(tri == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    return wei


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, emb: int, head_size: int, dropout: float):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(emb, head_size)
        self.query = nn.Linear(emb, head_size)
        self.val = nn.Linear(emb, head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, Emb)
        key = self.key(x)  # (B, T, HS)
        query = self.query(x)  # (B, T, HS)
        val = self.val(x)  # (B, T, HS)

        attention = (
            key @ query.transpose(-2, -1) * (1.0 / math.sqrt(key.size(-1)))
        )  # (B, T, HS) @ (B, HS, T) -> (B, T, T)
        wei = apply_causal_mask(attention)  # (B, T, T)
        wei = self.dropout(wei)
        return wei @ val  # (B, T, HS)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb: int, heads: int, dropout: float):
        """
        - emb: embendding length
        - heads: head count
        """
        super().__init__()
        assert emb % heads == 0, f"invalid config: emb: {emb}, heads: {heads}"

        self.emb = emb
        self.heads = heads
        self.head_size = emb // heads
        self.attention_heads = nn.ModuleList(
            [
                SingleHeadSelfAttention(emb, self.head_size, dropout)
                for _ in range(self.heads)
            ]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: (B, T, Emb)
        """
        B, T, Emb = x.size()
        x = torch.concat(
            [attention_head(x) for attention_head in self.attention_heads], dim=-1
        )  # (B, T, Emb)
        return x


class MultiHeadSelfAttentionV2(nn.Module):
    def __init__(self, emb: int, heads: int, dropout: float):
        super().__init__()
        assert emb > 0
        assert heads > 0
        assert emb % heads == 0
        assert dropout >= 0.0

        self.emb = emb
        self.heads = heads
        self.key = nn.Linear(emb, emb)
        self.query = nn.Linear(emb, emb)
        self.val = nn.Linear(emb, emb)
        self.norm = nn.LayerNorm(emb, eps=1e-6)
        self.atten_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(emb, emb)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: (B, T, Emb)
        """
        B, T, Emb = x.size()
        assert Emb % self.heads == 0

        residual = x

        head_size = Emb // self.heads

        query = self.query(x)  # (B, T, Emb)
        key = self.key(x)  # (B, T, Emb)
        val = self.val(x)  # (B, T, Emb)

        query = query.view(B, T, self.heads, head_size).transpose(
            1, 2
        )  # (B, T, HC, HS) -> (B, HC, T, HS)
        key = key.view(B, T, self.heads, head_size).transpose(
            1, 2
        )  # (B, T, HC, HS) -> (B, HC, T, HS)
        val = val.view(B, T, self.heads, head_size).transpose(
            1, 2
        )  # (B, T, HC, HS) -> (B, HC, T, HS)

        atten = (
            query @ key.transpose(-2, -1) * (1.0 / torch.tensor(T).sqrt())
        )  # (B, HC, T, HS) @ (B, HC, HS, T) -> (B, HC, T, T)
        atten = apply_causal_mask_v2(atten)  # (B, HC, T, T)
        atten = self.atten_dropout(atten)

        val = atten @ val  # (B, HC, T, T) @ (B, HC, T, HS) -> (B, HC, T, HS)
        val = val.transpose(1, 2).contiguous().view(B, T, Emb)
        val = self.c_proj(self.residual_dropout(val + residual))
        val = self.norm(val)
        return val


class SelfAttentionBlck(nn.Module):
    def __init__(self, emb: int, heads: int, dropout: float):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.normal1 = nn.LayerNorm((emb))
        self.normal2 = nn.LayerNorm((emb))

        self.mult_head_att = MultiHeadSelfAttentionV2(
            emb=emb, heads=heads, dropout=dropout
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(emb, 4 * emb, bias=True),
            nn.GELU(),
            nn.Linear(4 * emb, emb, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: (B, T, Emb)
        """
        x = x + self.mult_head_att(self.normal1(x))
        x = x + self.mlp(self.normal2(x))
        return x
