import torch
from torch import nn
import torch.nn.functional as F
from model.ResNet.ResNet_layer import ResNetnetwork


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.sample_num = 38
        self.in_channels = 1
        self.out_channels = time_lag
        self.device = device
        self.Conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.Conv_Time = nn.Conv2d(in_channels=self.time_lag, out_channels=1, kernel_size=3, padding=1)
        self.ResNet = ResNetnetwork(station_num=self.station_num, in_channels=self.in_channels, out_channels=self.out_channels)
        self.linear1 = nn.Linear(in_features=self.time_lag * 3, out_features=128).to(self.device)
        # self.linear2 = nn.Linear(in_features=2048, out_features=1024).to(self.device)
        self.linear3 = nn.Linear(in_features=128, out_features=self.pre_len).to(self.device)
        self.active = nn.Tanh()

    def forward(self, flow, inflow, ODRate, adj, external):
        inflow = inflow.to(self.device)

        inflow_week = inflow[:, 0:1, :, :]  # (batch, 1, station, station)
        inflow_day = inflow[:, 1:2, :, :]
        inflow_time = inflow[:, 2:self.time_lag + 2, :, :]  # (batch, time_lag, station, station)
        inflow_week = self.active(self.Conv(inflow_week))
        inflow_day = self.active(self.Conv(inflow_day))
        inflow_time = self.active(self.Conv_Time(inflow_time))
        inflow_week = self.ResNet(inflow_week)  # (batch, time_lag, station, station)
        inflow_day = self.ResNet(inflow_day)
        inflow_time = self.ResNet(inflow_time)
        inflow = torch.cat([inflow_week, inflow_day, inflow_time], dim=1)

        output = inflow.permute(0, 2, 3, 1)
        output = F.relu(self.linear1(output))  # (32, 210)
        # output = self.linear2(output)  # (32, 64)
        output = self.linear3(output)  # (32, 14*pre_len)
        output = output.reshape(output.size()[0], self.pre_len, self.station_num, self.sample_num)  # (32, 60, pre_len)
        return output