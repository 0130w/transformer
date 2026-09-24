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

    def forward(self, x, ctx=None, mask=None):  # [batch_size, seq_len, d_model]
        """
        x: [batch_size, q_len, d_model]
        ctx: [batch_size, kv_len, d_model]
        """
        if ctx is None:
            ctx = x
        batch_size, q_len, _ = x.shape
        kv_len = ctx.size(1)
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
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
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

    def forward(self, x, mask=None):  # [batch_size, seq_len, d_model]
        x = self.conv1(
            self.attention(x, mask=mask), x
        )  # [batch_size, seq_len, d_model]
        x = self.conv2(self.ffn(x), x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, head_num) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, head_num)
        self.conv1 = ResidualLayer(d_model)
        self.encode_decode_attention = MultiHeadAttention(d_model, head_num)
        self.conv2 = ResidualLayer(d_model)

        self.ffn = FeedForwardLayer(d_model, head_num, 4 * d_model)
        self.conv3 = ResidualLayer(d_model)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        x = self.conv1(
            self.attention(x, mask=tgt_mask), x
        )  # get information from generated output
        x = self.conv2(self.encode_decode_attention(x, ctx=memory, mask=src_mask), x)
        x = self.conv3(self.ffn(x), x)
        return x


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
        self.encoder = Encoder(d_model, head_num)
        self.decoder = Decoder(d_model, head_num)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        """
        src_ids: [batch_size, q_len]
        tgt_ids: [batch_size, kv_len]
        """
        src_padding_mask = (src_ids == 0)[:, None, None, :]  # [batch_size, 1, 1, q_len]
        tgt_padding_mask = (tgt_ids == 0)[
            :, None, None, :
        ]  # [batch_size, 1, 1, kv_len]
        kv_len = tgt_ids.size(1)
        tgt_causal_mask = torch.triu(
            torch.ones(kv_len, kv_len, dtype=torch.bool, device=tgt_ids.device),
            diagonal=1,  # diagonal = 1 not mask itself
        )
        src_mask = src_padding_mask  # [batch_size, 1, 1, q_len]
        tgt_mask = tgt_padding_mask | tgt_causal_mask  # [batch_size, 1, q_len, kv_len]
        src_embedding = self.embedding(src_ids)  # [batch_size, q_len, d_model]
        tgt_embedding = self.embedding(tgt_ids)  # [batch_size, kv_len, d_model]
        src_embedding = self.positional_embedding(
            src_embedding
        )  # [batch_size, q_len, d_model]
        tgt_embedding = self.positional_embedding(
            tgt_embedding
        )  # [batch_size, kv_len, d_model]
        src = self.encoder(src_embedding, mask=src_mask)
        tgt = self.decoder(tgt_embedding, memory=src, mask=tgt_mask)
        out = torch.softmax(self.out_proj(tgt), dim=-1, dtype=torch.float32)
        return out
