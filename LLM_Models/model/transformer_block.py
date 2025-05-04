import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(".."))

from .feed_forward import FeedForward
from .multihead_attetion import MultiHeadAttetion
from .position_encoding import PositionEncoding
from .encoder_layer import EncoderLayer

class Transformer(nn.Module):
  def __init__(self, vocab_size, d_model=256, max_len=512, num_heads=8, num_layers=6, d_hidden=1024, num_classes=2, dropout=0.1):
    super(Transformer, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, d_model)
    self.positionencoding = PositionEncoding(max_len, d_model)
   
    self.layers = nn.ModuleList([
          EncoderLayer(d_model, num_heads, d_hidden, dropout) for _ in range(num_layers)
       ])

   
    self.fc_start = nn.Linear(d_model, 1)
    self.fc_end = nn.Linear(d_model, 1)
   
  def forward(self, x, attention_mask):
    x = self.embeddings(x)
    x = self.positionencoding(x)

    for layer in self.layers:
      x = layer(x, attention_mask)

    start_logits = self.fc_start(x).squeeze(-1)  # Start token prediction
    end_logits = self.fc_end(x).squeeze(-1)      # End token prediction

    return start_logits, end_logits
