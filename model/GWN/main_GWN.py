import torch
from torch import nn
import torch.nn.functional as F
from model.GWN.GWN_layer import gwnet


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device, support):
        super(Model, self).__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.input_size = 12
        self.sample_num = 15
        self.device = device
        self.Graph_WaveNet = gwnet(device=device, num_nodes=self.station_num, supports=support, in_dim=time_lag).to(self.device)
        # linear1的in_features是需要自行确定的
        self.linear1 = nn.Linear(in_features=3, out_features=256).to(self.device)
        self.linear2 = nn.Linear(in_features=1024, out_features=512).to(self.device)
        self.linear3 = nn.Linear(in_features=256, out_features=self.sample_num)

    def forward(self, flow, inflow, ODRate, adj):
        inflow = inflow.to(self.device)
        # adj = adj.to(self.device)

        inflow_week = inflow[:, 0:1, :, :]  # (batch, 1, station, station)
        inflow_day = inflow[:, 1:2, :, :]
        inflow_time = inflow[:, 2:self.time_lag + 2, :, :]
        # inflow_time = inflow_time.permute(0, 1, 2)  # (batch, station_num, time_lag)

        # inflow_week = self.Graph_WaveNet(inflow_week)
        # inflow_day = self.Graph_WaveNet(inflow_day)
        inflow_time = self.Graph_WaveNet(inflow_time)
        # print(inflow_time.shape)

        # inflow_all = torch.cat([inflow_week, inflow_day, inflow_time], dim=2)

        # output = inflow_time.reshape(inflow_time.size()[0], -1)  # (batch, station_num * time_lag)
        # print(inflow_time.shape)
        output = F.relu(self.linear1(inflow_time))
        # output = self.linear2(output)
        output = self.linear3(output)
        output = output.reshape(output.size()[0], self.pre_len, self.station_num, self.sample_num)
        return output
