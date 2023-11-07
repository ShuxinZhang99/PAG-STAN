import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer.Transformer import Transformer


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.device = device
        self.d_model = 512
        self.sample_num = 38

        self.transformer = Transformer()
        self.linear = nn.Conv1d(in_channels=self.sample_num*self.station_num, out_channels=self.d_model, kernel_size=1)
        self.linear1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.sample_num*self.station_num, kernel_size=1).to(self.device)
        self.linear2 = nn.Conv1d(in_channels=self.sample_num*self.station_num, out_channels=self.sample_num*self.station_num, kernel_size=1).to(self.device)
        self.reg1 = nn.Linear(in_features=self.time_lag, out_features=36)
        self.reg2 = nn.Linear(in_features=36, out_features=self.pre_len)

    # def forward(self, inflow, weibo, adj):
    def forward(self, flow, inflow, ODRate, adj, external):
        inflow = inflow.to(self.device)

        inflow_week = inflow[:, 0:1, :, :]  # (batch, 1, station, station)
        inflow_day = inflow[:, 1:2, :, :]
        inflow_time = inflow[:, 2:self.time_lag + 2, :, :]  # (batch, time_lag, station, station)
        inflow_OD = self.linear(inflow_time.reshape(inflow_time.size()[0], -1, self.time_lag))
        # inflow_DO = inflow_OD.permute(0, 2, 1, 3)

        inflow_OD = self.transformer(inflow_OD)  # (batch_size, station_num, d_model)
        # inflow_DO = self.transformer(inflow_DO)

        # inflow_time = torch.cat([inflow_OD, inflow_DO], dim=-1)

        # output = inflow_time.reshape(inflow_time.size()[0], -1)  # (32, 10*76)
        output = self.linear1(inflow_OD)  # (32, 360)
        output = self.linear2(output)
        output = F.relu(self.reg1(output))
        output = self.reg2(output)
        output = output.reshape(output.size()[0], self.pre_len, self.station_num, self.sample_num)  # ( 64, 14, pre_len)
        return output
