import torch
from torch import nn
import torch.nn.functional as F


class Lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Lstm_reg, self).__init__()
        # input(batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        s, b, h = x.shape
        x = x.view(-1, h)
        x = self.linear1(x)
        # x = self.linear2(x)
        x = x.view(s, b, -1)
        return x