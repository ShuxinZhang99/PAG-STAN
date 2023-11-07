import torch
from torch import nn
import torch.nn.functional as F
from model.Informer.origin_Informer import Informer


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.device = device
        self.sample_num = 38
        # self.GCN = GraphConvolution(in_features=self.time_lag, out_features=self.time_lag).to(self.device)
        # self.TCN = TemporalConvNet(num_inputs=self.time_lag, num_channels=[24, 36, 12])
        self.Informer_time = Informer().to(self.device)
        # self.ResNet = ResNetnetwork(Time_window_size=self.time_lag, station_num=self.station_num, in_channels=
        #                             self.in_channels, out_channels=self.out_channels).to(self.device)
        self.linear1 = nn.Linear(in_features=self.time_lag, out_features=128).to(self.device)
        self.linear2 = nn.Linear(in_features=2048, out_features=1024).to(self.device)
        self.linear3 = nn.Linear(in_features=128, out_features=self.pre_len).to(self.device)

    def forward(self, flow, inflow, ODRate, adj, external):
        inflow = inflow.to(self.device)
        inflow_week = inflow[:, 0:1, :, :]
        inflow_day = inflow[:, 1:2, :, :]
        inflow_time = inflow[:, 2:2+self.time_lag, :, :]
        B,T,S,_ = inflow_time.size()
        residual = inflow_time = inflow_time.reshape(B,-1,T)
        inflow_time = self.Informer_time(x_enc=inflow_time, x_dec=inflow_time)  # (64, 276, 10)
        inflow_time += residual
        # inflow_week = self.Informer_week(x_enc=inflow_week)  # (64, 276, 10)
        # inflow_day = self.Informer_day(x_enc=inflow_day)  # (64, 276, 10)
        # inflow = torch.cat([inflow_week, inflow_day, inflow_time], dim=2)  # (64, 61, 36)

        # output = inflow_time.reshape(inflow_time.size()[0], -1)  # (64, 61*36)
        output = F.relu(self.linear1(inflow_time))  # (64, 1024)
        # output = self.linear2(output) # (64, 512)
        output = self.linear3(output)  # (64, 276*pre_len)
        output = output.reshape(output.size()[0], self.pre_len, self.station_num, self.sample_num)  # ( 64, 276, pre_len)
        return output
