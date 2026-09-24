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
        self.out_proj = nn.Linear(d_model, d_model)  # gather information from each head

    def forward(self, x):  # [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = torch.reshape(
            q, (batch_size, seq_len, self.head_num, self.head_dim)
        ).transpose(1, 2)  # [batch_size, head_num, seq_len, head_dim]
        k = torch.reshape(
            k, (batch_size, seq_len, self.head_num, self.head_dim)
        ).transpose(1, 2)
        v = torch.reshape(
            v, (batch_size, seq_len, self.head_num, self.head_dim)
        ).transpose(1, 2)
        scores = q @ k.transpose(-1, -2)  # [batch_size, head_num, seq_len, seq_len]
        scores = scores / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, -1)
        out = (
            (weights @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        )  # [batch_size, seq_len, d_model]
        out = self.out_proj(out)
        return out


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class Encoder(nn.Module):
    def __init__(self, d_model, head_num) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, head_num)

    def forward(self, x):  # [batch_size, seq_len, d_model]
        x = self.attention(x)  # [batch_size, seq_len, d_model]


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
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_embedding(x)  # [batch_size, seq_len, d_model]
