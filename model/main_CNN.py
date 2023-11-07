import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.device = device
        self.sample_num = 38
        self.Conv2d = nn.Conv2d(in_channels=1, out_channels=self.time_lag, kernel_size=3, padding=1).to(self.device)
        self.Conv2D = nn.Conv2d(in_channels=self.time_lag, out_channels=self.time_lag, kernel_size=3, padding=1).to(self.device)
        self.linear1 = nn.Linear(in_features= self.time_lag, out_features=256).to(self.device)
        self.linear2 = nn.Linear(in_features=1024, out_features=512).to(self.device)
        self.linear3 = nn.Linear(in_features=256, out_features= self.pre_len).to(self.device)

    def forward(self, flow, inflow, ODRate, adj, external):
        inflow = inflow.to(self.device)
        inflow_week = inflow[:, 0:1, :, :]  # (32, 60, 10)
        inflow_day = inflow[:, 1:2, :, :]
        inflow_time = inflow[:, 2:self.time_lag + 2, :, :]
        inflow_time = self.Conv2D(inflow_time)  # (32, 60)
        inflow_week = self.Conv2d(inflow_week)
        inflow_day = self.Conv2d(inflow_day)
        # inflow = torch.cat([inflow_week, inflow_day, inflow_time], dim=1)  # (64, 3*time_lag, station_num, station_num)

        output = inflow_time.permute(0, 2, 3, 1)
        output = F.relu(self.linear1(output))  # (32, 210)
        # output = self.linear2(output) # (32, 64)
        output = self.linear3(output)  # (32, 14*pre_len)
        output = output.reshape(output.size()[0], self.pre_len, self.station_num, self.sample_num)  # ( 64, 14, pre_len)
        return output
