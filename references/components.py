from typing import Optional

import einops
import torch
from einops.layers.torch import EinMix

from utils.config import NetworkConfig, WorldConfig


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_sincos_positions(dim: int, shape: tuple):
    wavelengths = torch.as_tensor(shape)
    coordinates = torch.stack(torch.unravel_index(indices=torch.arange(wavelengths.prod()), shape=shape), dim=-1)
    valid = wavelengths > 1
    log_wavelengths = wavelengths[valid].log()
    coordinates = coordinates[:, valid]
    negative_spacing = torch.linspace(0, -1, dim // (coordinates.size(-1) * 2))
    frequencies = torch.exp(negative_spacing * log_wavelengths[..., None])
    angles = torch.einsum("n i, i d -> n i d", coordinates, frequencies)
    positions = einops.rearrange([angles.sin(), angles.cos()], "two n i d -> n (two i d)")
    return torch.nn.functional.pad(positions, (0, dim - positions.size(-1)))


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor.div(keep_prob)


class GatedFFN(torch.nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 2, bias: bool = False):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.to_hidden = torch.nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.to_out = torch.nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.to_hidden(x).chunk(2, dim=-1)
        return self.to_out(torch.nn.functional.silu(x1) * x2)


class EinAttention(torch.nn.Module):
    def __init__(self, dim_q: int, dim_kv: Optional[int] = None, num_heads: Optional[int] = None, dim_heads: int = 64):
        super().__init__()
        dim_kv = default(dim_kv, dim_q)
        num_heads = default(num_heads, max(dim_q // dim_heads, dim_kv // dim_heads, 1))

        self.norm_q = torch.nn.RMSNorm(dim_heads)
        self.norm_k = torch.nn.RMSNorm(dim_heads)
        self.to_q = EinMix("... nq dq -> ... nh nq dh", weight_shape="dq nh dh",
                           nh=num_heads, dh=dim_heads, dq=dim_q)
        self.to_kv = EinMix("... nk dk -> kv ... nh nk dh", weight_shape="dk kv nh dh",
                            kv=2, nh=num_heads, dh=dim_heads, dk=dim_kv)
        self.to_out = EinMix("... nh nq dh -> ... nq dq", weight_shape="nh dh dq",
                             nh=num_heads, dh=dim_heads, dq=dim_q)

    def forward(self, q: torch.FloatTensor, kv: Optional[torch.FloatTensor] = None):
        kv = default(kv, q)
        KV = self.to_kv(kv)
        Q = self.to_q(q)
        A = torch.nn.functional.scaled_dot_product_attention(self.norm_q(Q), self.norm_k(KV[0]), KV[1])
        return self.to_out(A)


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, dim_kv: Optional[int] = None, num_heads: Optional[int] = None,
                 dim_heads: int = 64, expansion_factor: int = 2, drop_path: float = 0.):
        super().__init__()
        self.attn_norm = torch.nn.RMSNorm(dim)
        self.ffn_norm = torch.nn.RMSNorm(dim)
        self.att = EinAttention(dim, dim_kv=dim_kv, num_heads=num_heads, dim_heads=dim_heads)
        self.ffn = GatedFFN(dim=dim, expansion_factor=expansion_factor)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.FloatTensor, kv: Optional[torch.FloatTensor] = None):
        x = x + self.drop_path(self.att(self.attn_norm(x), kv=kv))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x


class ViT(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.world = world
        axes = {**world.token_sizes, **world.patch_sizes}

        self.to_tokens = torch.nn.Sequential(
            EinMix(f"b {world.field_pattern} -> b ({world.token_pattern}) d",
                   weight_shape=f"v {world.patch_pattern} d", d=network.dim, **axes),
            torch.nn.RMSNorm(network.dim),
        )
        self.to_fields = EinMix(f"b ({world.token_pattern}) d -> b {world.field_pattern}",
                                weight_shape=f"v d {world.patch_pattern}", d=network.dim, **axes)

        self.positions = torch.nn.Parameter(init_sincos_positions(network.dim, shape=world.token_shape))

        self.blocks = torch.nn.ModuleList([
            TransformerBlock(network.dim, num_heads=network.num_heads, dim_heads=network.dim_heads,
                             expansion_factor=network.expansion_factor, drop_path=network.drop_path)
            for _ in range(network.num_layers)
        ])

        self.apply(self.base_init)

    @staticmethod
    def base_init(m: torch.nn.Module):
        if isinstance(m, EinMix) or isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=m.weight.size(-1) ** -0.5)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        tokens = self.to_tokens(x) + self.positions
        for block in self.blocks:
            tokens = block(tokens)
        return self.to_fields(tokens)
