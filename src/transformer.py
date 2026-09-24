import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -math.log(10000.0)
            / d_model
        )
        angles = position * frequency
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(
            angles[:, : d_model // 2]
        )  # compatible with d_model is odd

        self.register_buffer(
            "pe", pe.unsqueeze(0)
        )  # [max_len, d_model] -> [1, max_len, d_model]

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):  # type: ignore
            raise ValueError("seq_len of x exceeds max_len of pe")
        return x + self.pe[:, :seq_len].to(dtype=x.dtype)  # type: ignore


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num) -> None:
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = d_model // head_num
        # d_q = d_k = d_v = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # get information from each head

    def forward(self, x, ctx=None):  # [batch_size, seq_len, d_model]
        """
        x: [batch_size, seq_len, d_model]
        ctx: [batch_size, src_len, d_model]
        """
        if ctx is None:  # encode stage
            ctx = x
        batch_size, q_len, _ = x.shape
        kv_len = ctx.shape(1)
        q = self.q_proj(x)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        q = torch.reshape(
            q, (batch_size, q_len, self.head_num, self.head_dim)
        ).transpose(1, 2)  # [batch_size, head_num, q_len, head_dim]
        k = torch.reshape(
            k, (batch_size, kv_len, self.head_num, self.head_dim)
        ).transpose(1, 2)  # [batch_size, head_num, kv_len, head_dim]
        v = torch.reshape(
            v, (batch_size, kv_len, self.head_num, self.head_dim)
        ).transpose(1, 2)
        scores = q @ k.transpose(-1, -2)  # [batch_size, head_num, q_len, kv_len]
        scores = scores / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, -1)
        out = (
            (weights @ v).transpose(1, 2).reshape(batch_size, q_len, self.d_model)
        )  # [batch_size, q_len, d_model]
        out = self.out_proj(out)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, input, x):  # add & norm
        assert x.shape == input.shape, "residual block shape mismatch"
        out = x + input  # add
        out = self.layernorm(out)  # norm
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, head_num, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True, dtype=torch.float32)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True, dtype=torch.float32)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.w_1(x)
        out = self.relu(out)
        out = self.w_2(out)
        return out


class Encoder(nn.Module):
    def __init__(self, d_model, head_num) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, head_num)
        self.conv1 = ResidualLayer(d_model)
        self.ffn = FeedForwardLayer(d_model, head_num, 4 * d_model)
        self.conv2 = ResidualLayer(d_model)

    def forward(self, x):  # [batch_size, seq_len, d_model]
        out = self.attention(x)  # [batch_size, seq_len, d_model]
        out = self.conv1(out, x)
        out = self.ffn(out)
        out = self.conv2(out, x)
        return out


class Decoder(nn.Module):
    def __init__(self, d_model, head_num) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, head_num)
        self.conv1 = ResidualLayer(d_model)
        self.encode_decode_attention = MultiHeadAttention(d_model, head_num)
        self.conv2 = ResidualLayer(d_model)
        self.ffn = FeedForwardLayer(d_model, head_num, 4 * d_model)
        self.conv3 = ResidualLayer(d_model)

    def forward(self, x, memory):
        out = self.attention(x)  # get information from generated output
        out = self.conv1(out, x)
        out = self.encode_decode_attention(x, memory)
        out = self.conv2(out, x)
        out = self.ffn(out)
        out = self.conv3(out)
        return out


class Transformer(nn.Module):
    """
    x: [batch_size, seq_len]
    padding token = 0, not accumulate gradient
    """

    def __init__(self, vocab_size, d_model, head_num=8, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.head_num = head_num

        assert self.d_model % self.head_num == 0, "head_num should divided by d_model"

        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.positional_embedding = PositionalEmbedding(d_model, max_len)

    def forward(self, x: torch.Tensor):
        out = self.embedding(x)  # [batch_size, seq_len, d_model]
        out = self.positional_embedding(out)  # [batch_size, seq_len, d_model]
