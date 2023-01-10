import math
import torch
from torch import nn as nn

class PositionalEncoding(nn.Module):
    """PositionalEncoding module from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Modified so that it works with batch_first = True for encoder layer."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (#batches), seq_len, embedding_dim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AttentionBlock(nn.Module):
    """Wraps the nn.MultiheadAttention module from PyTorch to calculate (query, key, value) (with
    target=curr_match_emb, source=past_matches_emb) and perform multiheaded attention."""
    def __init__(self, embed_dim: int, num_heads: int):
            super(AttentionBlock, self).__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            
            self.Q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.K = nn.Linear(embed_dim, embed_dim, bias=False)
            self.V = nn.Linear(embed_dim, embed_dim, bias=False)
            self.mul_head_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, add_zero_attn=True) # Bias = False? 

    def forward(self, past_matches_emb, curr_match_emb, mask):
        """
        Args:
            past_matches_emb: Tensor of shape (#batches, history_len, embed_dim).
            current_match_emb: Tensor of shape (#batches, 2).
            mask: bool Tensor of shape (#bathes, history_len).
        Returns:
            att: Tensor of shape (#batches, embed_dim)."""

        query = self.Q(curr_match_emb).view(-1, 1, self.embed_dim) # Reshaped to (#batches, 1, embed_dim)
        keys = self.K(past_matches_emb) 
        values = self.V(past_matches_emb)

        att = self.mul_head_att(query, keys, values, key_padding_mask=mask)[0] # Shape (batches, 1, embed_dim)
        att = att.view(-1, self.embed_dim)

        return att