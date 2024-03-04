import os
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        if os.getenv("ACC_FLASH_ATTN", "0") == "1":
            self.fused_attn = False
            self.use_acc_flash_attn = True
        else:
            self.fused_attn = True
            self.use_acc_flash_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        elif self.use_acc_flash_attn:
            from torchacc.ops import flash_attn_varlen_xla
            q = q * self.scale
            q, k, v = [x.transpose(1,2) for x in [q, k, v]]
            q, k, v = [einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
            cu_q_lens = torch.arange(
                0, (B + 1) * N, step=N, dtype=torch.int32, device=q.device)
            output = flash_attn_varlen_xla(q, k, v, cu_q_lens, cu_q_lens, N, N, 0.0, softmax_scale=None, causal=True)
            output = einops.rearrange(output, "(b s) ... -> b s ...", b=B)
            output = self.proj(einops.rearrange(output, "b s h d -> b s (h d)"))
            output = self.proj_drop(output)
            return output
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x