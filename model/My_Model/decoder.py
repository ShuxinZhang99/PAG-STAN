import torch
from torch import nn
import torch.nn.functional as F
from model.My_Model.GraphLSTM import GConvLSTM


class Decoder(nn.Module):
    def __init__(self, station_num, time_lag, input_sz, hidden_sz):
        super(Decoder, self).__init__()
        self.station_num = station_num
        self.time_lag = time_lag
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.n_layers = 1
        self.layers = nn.ModuleList([GConvLSTM(station_num, input_sz, hidden_sz) for _ in range(self.n_layers)])
        # self.reg = nn.Linear(in_features=time_lag, out_features=1)
        # self.cov3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=7, padding=3)

    def forward(self, ODFlow, adj, edc_output):
        OD_time = ODFlow[:, 2:self.time_lag + 2, :, :]
        for layer in self.layers:
            adjacency = adj
            OD_time, _ = layer(OD_time, adjacency, edc_output)
        # OD_sequence, _ = self.GraphLstm(OD_time, adj, enc_output)  # (batch, time_lag, station, station)

        return OD_time


