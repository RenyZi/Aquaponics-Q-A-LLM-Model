import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(".."))
from .feed_forward import FeedForward
from .multihead_attetion import MultiHeadAttetion  # Fixed spelling

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttetion(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_hidden)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.layer1 = nn.LayerNorm(d_model)
        self.layer2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Multi-head Attention + Add & Norm
        residual = x
        _, attention_output = self.attention(x, x, x, mask)
        x = self.layer1(residual + self.drop1(attention_output))

        # Feed Forward + Add & Norm
        residual = x
        x = self.layer2(residual + self.drop2(self.ffn(x)))

        return x

