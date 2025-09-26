from typing import Union, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
import math

'''
A very condensed definition for the predictive coder model.
'''

# --------------------------------------
# Attention definitions from https://github.com/jgornet/predictive-coding-recovers-maps/blob/main/predictive_coding/models/attention.py
# --------------------------------------

from xformers.ops import memory_efficient_attention as sdpa_attention
from xformers.ops import LowerTriangularMask


class MultiHeadAttention(nn.Module):
    """
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)
    Args:
        d_model (int): The dimension of keys / values / queries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)
        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **mask** (-): tensor containing indices to be masked
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 2):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.query_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )
        self.key_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )
        self.value_proj = nn.Conv2d(
            d_model, self.d_head * num_heads, kernel_size=1, bias=False
        )

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size = value.size(0)
        channels = value.size(2)
        height = value.size(3)
        width = value.size(4)

        query = query.reshape(-1, channels, height, width)
        query = self.query_proj(query)
        query = query.reshape(
            batch_size, -1, self.num_heads, self.d_head, height, width
        )

        key = key.reshape(-1, channels, height, width)
        key = self.key_proj(key)
        key = key.reshape(batch_size, -1, self.num_heads, self.d_head, height, width)

        value = value.reshape(-1, channels, height, width)
        value = self.value_proj(value)
        value = value.reshape(
            batch_size, -1, self.num_heads, self.d_head, height, width
        )

        query = query.contiguous().view(
            batch_size, -1, self.num_heads, self.d_head * height * width
        )
        key = key.contiguous().view(
            batch_size, -1, self.num_heads, self.d_head * height * width
        )
        value = value.contiguous().view(
            batch_size, -1, self.num_heads, self.d_head * height * width
        )

        context = sdpa_attention(query, key, value, LowerTriangularMask())
        attn = None
        context = context.view(
            batch_size, -1, self.num_heads * self.d_head, height, width
        )

        return context, attn


class RotaryEncoding(nn.Module):
    def __init__(self, in_channels, max_len=2048, base=10000, device="cuda:0"):
        super().__init__()
        self.in_channels = in_channels
        self.max_len = max_len
        self.base = base
        self.inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.in_channels, 2, dtype=torch.int64)
                .float()
                .to(device)
                / self.in_channels
            )
        )
        self.cache_len = 0

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.cache_len = seq_len
        t = torch.arange(self.cache_len, device=device, dtype=torch.int64).type_as(
            self.inv_freq
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        # x: [batch_dim, time_dim, num_heads, head_dim]
        if seq_len > self.cache_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos[:seq_len].to(dtype=x.dtype),
            self.sin[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_position_embedding(q, k, cos, sin, position_ids, unsqueeze_dim=2):
    # q: [batch_dim, time_dim, num_heads, head_dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

# --------------------------------------
# Predictive Coder definitions
# from https://github.com/jgornet/predictive-coding-recovers-maps/blob/main/predictive_coding/models/residual.py
# and https://github.com/jgornet/predictive-coding-recovers-maps/blob/main/predictive_coding/models/encoder_decoder.py
# --------------------------------------

class Residual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1)
            self.skip = nn.Conv2d(in_channels, channels, kernel_size=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class ResidualTranspose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        upsample: bool = False
    ) -> None:
        super().__init__()
        self.upsample = upsample

        self.skip = None
        if in_channels < channels:
            self.skip = nn.ConvTranspose2d(in_channels, channels, kernel_size=1, stride=1)
        elif in_channels > channels:
            delta_channels = in_channels - channels
            self.skip = lambda x: torch.cat([x, torch.zeros(x.shape[0], delta_channels, x.shape[2], x.shape[3], device=x.device)], dim=1)

        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.skip = nn.ConvTranspose2d(in_channels, channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class DownBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        downsample: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Residual(in_channels, out_channels, downsample=downsample)])
        for _ in range(layers - 1):
            self.layers.append(Residual(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        upsample: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ResidualTranspose(in_channels, out_channels, upsample=upsample)])
        for _ in range(layers - 1):
            self.layers.append(ResidualTranspose(out_channels, out_channels))

    def forward(self, x: torch.Tensor, skip: Union[torch.Tensor, None]) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class ConvBottleneck(nn.Module):
    def __init__(self, actions: int, in_channels: int, out_channels: int, features: int=4):
        super().__init__()
        self.features = features
        # self.linear = nn.Linear(actions, self.features * 8 * 8, bias=False)
        self.linear = nn.Linear(actions, self.features * 2 * 2, bias=False)
        self.conv = nn.Conv2d(self.features + in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, features, actions):
        B, C, H, W = features.shape
        conv_actions = self.linear(actions).reshape(B, self.features, H, W)

        output = torch.cat([features, conv_actions], dim=1)
        output = self.conv(output)
        output = self.bn(output)
        output = F.relu(output)

        return output

class Encoder(nn.Module):
    def __init__(self, in_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down1 = DownBlock2d(64, 64, layers[0])
        self.down2 = DownBlock2d(64, 128, layers[1], downsample=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)  # 32
        x = self.bn1(x)
        x = F.relu(x)

        x = self.maxpool(x)  # 16

        x = self.down1(x)  # 16
        x = self.down2(x)  # 8

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(64, out_channels, kernel_size=1, stride=1)
        self.up1 = UpBlock2d(128, 64, layers[0], upsample=True)
        self.up2 = UpBlock2d(64, 64, layers[1], upsample=True)
        self.up3 = UpBlock2d(64, 64, 1, upsample=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x, None)
        x = self.up2(x, None)
        x = self.up3(x, None)

        x = self.conv1(x)

        return x


class FFN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.reshape(B, L, C, H, W)

        return x

class PredictiveCoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, layers: List[int], seq_len: int
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, layers)
        self.decoder = Decoder(in_channels, out_channels, layers[::-1])
        self.attention_1 = MultiHeadAttention(128, num_heads=8)
        self.ffn_1 = FFN(128, 128)
        self.attention_2 = MultiHeadAttention(128, num_heads=8)
        self.ffn_2 = FFN(128, 128)
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        #self.position = PositionalEncoding(512, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)

        f = self.encoder(x)

        f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])

        residual = f
        f = self.attention_1(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_1(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.attention_2(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_2(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        f = f.reshape(B * L, f.shape[2], f.shape[3], f.shape[4])

        x = self.decoder(f)
        x = x.reshape(B, L, C, H, W)

        return x

    def get_latents(self, x: torch.Tensor, actions: torch.Tensor = None, which=-1) -> torch.Tensor:
        with torch.no_grad():
            B, L, C, H, W = x.shape
            x = x.reshape(B * L, C, H, W)
    
            f = self.encoder(x)
            
            f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])
    
            if which == 0:
                return f
    
            residual = f
            f = self.attention_1(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_1(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            if which == 1:
                return f
    
            residual = f
            f = self.attention_2(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_2(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            return f

class VestibularCoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, layers: List[int], seq_len: int, num_actions:int = None) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, layers)
        self.decoder = BiggerDecoder(in_channels * 2, out_channels, layers[::-1])
        self.attention_1 = MultiHeadAttention(128 * 2, num_heads=8)
        self.ffn_1 = FFN(128 * 2, 128 * 2)
        self.attention_2 = MultiHeadAttention(128 * 2, num_heads=8)
        self.ffn_2 = FFN(128 * 2, 128 * 2)
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.bottleneck = ConvBottleneck(actions=num_actions, in_channels=128, out_channels=128, features=128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        actions = actions.reshape(B*L, -1)

        f = self.encoder(x)
        f = self.bottleneck(f, actions)

        f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])

        residual = f
        f = self.attention_1(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[2:])

        residual = f
        f = self.ffn_1(f) + residual
        f = F.layer_norm(f, f.shape[2:])

        residual = f
        f = self.attention_2(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[2:])

        residual = f
        f = self.ffn_2(f) + residual
        f = F.layer_norm(f, f.shape[2:])

        f = f.reshape(B * L, f.shape[2], f.shape[3], f.shape[4])

        x = self.decoder(f)
        x = x.reshape(B, L, C, H, W)

        return x

    def get_latents(self, x: torch.Tensor, actions: torch.Tensor = None, which=-1) -> torch.Tensor:
        with torch.no_grad():
            B, L, C, H, W = x.shape
            x = x.reshape(B * L, C, H, W)
    
            f = self.encoder(x)
            f = self.bottleneck(f, actions)
            
            f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])
    
            if which == 0:
                return f
    
            residual = f
            f = self.attention_1(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_1(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            if which == 1:
                return f
    
            residual = f
            f = self.attention_2(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_2(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            return f

class ConvBottleneck(nn.Module):
    def __init__(self, actions: int, in_channels: int, out_channels: int, features: int=128):
        super().__init__()
        self.features = features
        self.linear = nn.Linear(2, self.features * 8 * 8, bias=True) #512
        self.conv = nn.Conv2d(self.features + in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, features, actions):
        B, C, H, W = features.shape
        theta = torch.deg2rad(actions)
        phi = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)   # (B, 2)
        
        conv_actions = self.linear(phi).view(B, self.features, H, W)
        conv_actions = self.linear(actions).reshape(B, self.features, H, W)

        output = torch.cat([features, conv_actions], dim=1)
        
        #output = self.conv(output)
        #output = self.bn(output)
        #output = F.relu(output)

        return output

    def get_latents(self, features, actions):
        B, C, H, W = features.shape
        return self.linear(actions).reshape(B, self.features, H, W)