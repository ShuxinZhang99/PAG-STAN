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
        # x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)  # (batch, station * station, hidden)
        output = F.relu(self.linear1(x))  # (Batch, output_size)
        # # x = self.linear2(x)
        return output
