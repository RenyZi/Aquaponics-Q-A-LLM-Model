import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
