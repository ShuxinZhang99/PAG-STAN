import torch
from torch import nn
import torch.nn.functional as F
from model.LSTM.LSTM_layer import Lstm_reg


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super(Model, self).__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.sample_num = 38
        self.input_size = self.station_num * self.sample_num
        self.hidden_size = 128
        self.device = device
        self.Spatial_LSTM = Lstm_reg(input_size=self.input_size, hidden_size=self.hidden_size, output_size=self.station_num *
                                                                                    self.sample_num).to(self.device)
        self.Temporal_LSTM = Lstm_reg(input_size=self.time_lag, hidden_size=self.hidden_size, output_size=time_lag).to(
            self.device)
        self.linear1 = nn.Linear(in_features=self.time_lag, out_features=32).to(self.device)
        self.linear2 = nn.Linear(in_features=1024, out_features=512).to(self.device)
        self.linear3 = nn.Linear(in_features=32, out_features=pre_len)

    def forward(self, flow, inflow, ODRate, adj, external):
        inflow = inflow.to(self.device)

        inflow_week = inflow[:, 0:1, :, :]
        inflow_day = inflow[:, 1:2, :, :]
        inflow_time = inflow[:, 2:self.time_lag+2, :, :]
        B, T, N, S = inflow_time.shape
        # inflow_time = inflow_time.permute(0, 1, 2)  # (batch, station_num, time_lag)
        inflow_spa = inflow_time.reshape(B, T, -1)
        # inflow_tem = inflow_time.reshape(B, -1, T)

        inflow_spa = self.Spatial_LSTM(x=inflow_spa).permute(0, 2, 1)  # (batch, time, station * station)
        # inflow_tem = self.Temporal_LSTM(x=inflow_tem)  # (batch, station * station, time)

        # inflow_st = torch.cat([inflow_spa, inflow_tem], dim=-1)
        output = inflow_spa.reshape(B, N, S, T)  # (batch, station_num * time_lag)
        output = F.relu(self.linear1(output))
        # output = self.linear2(output)
        output = self.linear3(output)
        output = output.reshape(output.size()[0], -1, self.station_num, self.sample_num)
        return output

