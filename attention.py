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


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, emb: int, hs: int, dropout: float):
        super().__init__()
        self.hs = hs
        self.key = nn.Linear(emb, hs)
        self.query = nn.Linear(emb, hs)
        self.val = nn.Linear(emb, hs)
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
    def __init__(self, emb: int, hc: int, dropout: float):
        """
        - emb: embendding length
        - hc: head count
        """
        super().__init__()
        assert emb % hc == 0, f"invalid config: emb: {emb}, hc: {hc}"

        self.emb = emb
        self.hc = hc
        self.hs = emb // hc
        self.attention_heads = nn.ModuleList(
            [SingleHeadSelfAttention(emb, self.hs, dropout) for _ in range(self.hc)]
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


class SelfAttentionBlck(nn.Module):
    def __init__(self, emb: int, hc: int, dropout: float):
        super().__init__()
        self.emb = emb
        self.hc = hc
        self.normal1 = nn.LayerNorm((emb))
        self.normal2 = nn.LayerNorm((emb))

        self.mult_head_att = MultiHeadSelfAttention(emb=emb, hc=hc, dropout=dropout)

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
