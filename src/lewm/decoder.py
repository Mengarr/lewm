"""Lightweight transformer decoder for CLS token visualization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, q, kv):
        """
        q:  (B, P, D) — learnable query tokens
        kv: (B, 1, D) — projected CLS token
        """
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        Q = rearrange(self.to_q(q), "b p (h d) -> b h p d", h=self.heads)
        K, V = self.to_kv(kv).chunk(2, dim=-1)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.heads)
        V = rearrange(V, "b n (h d) -> b h n d", h=self.heads)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=drop)
        out = rearrange(out, "b h p d -> b p (h d)")
        return self.to_out(out)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.cross_attn = CrossAttention(dim, heads, dim_head, dropout)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv)
        q = q + self.mlp(self.norm(q))
        return q


class CLSDecoder(nn.Module):
    """Decodes the CLS token into a reconstructed image for visualization.

    Architecture:
    - Project CLS embedding to hidden_dim (used as K and V)
    - P learnable query tokens (one per patch) attend to the CLS via cross-attention
    - Linear head projects each query to patch_size^2 * 3 pixels
    """

    def __init__(
        self,
        cls_dim: int = 192,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 512,
        dropout: float = 0.0,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        num_patches = (img_size // patch_size) ** 2  # 196 for 224/16

        self.cls_proj = nn.Linear(cls_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.patch_head = nn.Linear(hidden_dim, patch_size * patch_size * 3)

        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """
        cls_emb: (B, cls_dim)
        returns: (B, 3, H, W) in [-1, 1]
        """
        B = cls_emb.size(0)
        kv = self.cls_proj(cls_emb).unsqueeze(1)  # (B, 1, hidden_dim)
        q = self.query_tokens.expand(B, -1, -1)   # (B, P, hidden_dim)

        for block in self.blocks:
            q = block(q, kv)
        q = self.norm(q)

        patches = self.patch_head(q)  # (B, P, patch_size^2 * 3)
        h = w = self.img_size // self.patch_size
        img = rearrange(
            patches,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=3,
        )
        return torch.tanh(img)
