import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model.ConvLSTM.ConvLSTM import *
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device, padding="SAME"):
        super().__init__()
        self.time_lag = time_lag  # 使用的历史时间步
        self.pre_len = pre_len
        self.station_num = station_num
        self.sample_num = 38
        self.device = device
        self.input_size = (self.station_num, self.sample_num)
        self.input_dim = 1
        self.hidden_dim = 64
        self.kernel_size = 3
        self.num_layers = 2
        self.padding = padding
        self.clstm = ConvLSTM(input_size=self.input_size,
                              input_dim=self.input_dim,
                              hidden_dim=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              num_layers=self.num_layers,
                              padding=self.padding)  # [10,3,46,30]
        # self.convlstm = ConvLSTM(30, 30, 3, 1, True, True, False).to(self.device)
        self.linear1 = nn.Linear(in_features=self.hidden_dim, out_features=64).to(self.device)
        # self.linear4 = nn.Linear(in_features=2048, out_features=1024).to(self.device)
        self.linear2 = nn.Linear(in_features=2048, out_features=1024).to(self.device)
        self.linear3 = nn.Linear(in_features=64, out_features=self.pre_len).to(self.device)

    def forward(self,  flow, inflow, ODRate, adj, external):
        inflow = inflow.to(self.device)

        inflow_week = inflow[:, 0:1, :, :]  # (32, 1, station_num, station_num)
        inflow_day = inflow[:, 1:2, :, :]
        inflow_time = inflow[:, 2:self.time_lag + 2, :, :]  # (batch_size, 12, station_num, station_num)
        # inflow = torch.cat([inflow_week, inflow_day, inflow_time], dim=2)  # （32， 75， 3 * time_lag）
        inflow_time = inflow_time.unsqueeze(2)  # [32,1,75,30]  (batch_size, 1, station_num, time_lag)

        # inflow_day = inflow_day.unsqueeze(1)  # [32,1,75,30]  (batch_size, 1, station_num, time_lag)

        # inflow_week = inflow_week.unsqueeze(1)  # [32,1,75,30]  (batch_size, 1, station_num, time_lag)

        # print('shape', inflow.shape)
        # inflow = inflow.view(-1, 1, 1, 41, 12)
        clstm_time, state = self.clstm(inflow_time)
        # clstm_day, state = self.clstm(inflow_day)
        # clstm_week, state = self.clstm(inflow_week)
        output_time = clstm_time[:, -1, :, :, :]  # [batch_size, hidden_dim, station_num, station_num]
        # output_day = clstm_day[:, -1, :, :, :]
        # output_week = clstm_week[:, -1, :, :, :]
        # output = torch.cat([output_week, output_day, output_time], dim=2)
        output = output_time.permute(0, 2, 3, 1)  # (batch_size, 3*75*36)
        output = F.relu(self.linear1(output))  # (64, 1024)
        # output = self.linear2(output)  # (64, 512)
        output = self.linear3(output)  # (64, 276*pre_len)
        output = output.reshape(output.size()[0], self.pre_len, self.station_num, self.sample_num)  # ( 64, 276, pre_len)
        return output
