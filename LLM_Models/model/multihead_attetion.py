import os
import sys
import math
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(".."))
from .attetion import ScaledDotProductAttention

class MultiHeadAttetion(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttetion, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.Q_weight = nn.Linear(d_model, d_model)
        self.K_weight = nn.Linear(d_model, d_model)
        self.V_weight = nn.Linear(d_model, d_model)
        self.O_weight = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, d_model)
        x = x.view(batch_size, -1, self.num_heads, self.dk)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, dk)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.split_heads(self.Q_weight(Q), batch_size)
        K = self.split_heads(self.K_weight(K), batch_size)
        V = self.split_heads(self.V_weight(V), batch_size)

        # Prepare mask shape for broadcasting if needed
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        attn_weights, attn_output = self.attention(Q, K, V, mask)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        output = self.O_weight(attn_output)
        return attn_weights, output

