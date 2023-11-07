import torch
from torch import nn
import torch.nn.functional as F
from model.ConvLSTM.ConvLSTM import *


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.sample_num = 38
        self.device = device
        self.input_size = (self.station_num, self.sample_num)
        self.input_dim = 16
        self.hidden_dim = 32
        self.kernel_size = 3
        self.num_layers = 1
        self.padding = "SAME"
        self.Conv_OD = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(1, 3, 3), padding=(0,1,1)).to(self.device)
        self.Conv2D = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 3, 3), padding=(0,1,1)).to(self.device)
        self.Conv_TEC = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1).to(self.device)
        self.Conv_GCC = nn.Conv2d(in_channels=self.hidden_dim, out_channels=1, kernel_size=3, padding=1).to(self.device)
        self.clstm = ConvLSTM(input_size=self.input_size,
                              input_dim=self.input_dim,
                              hidden_dim=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              num_layers=self.num_layers,
                              padding=self.padding)
        self.linear1 = nn.Linear(in_features=self.hidden_dim+1, out_features=128).to(self.device)
        self.linear2 = nn.Linear(in_features=1024, out_features=512).to(self.device)
        self.linear3 = nn.Linear(in_features=128, out_features=self.pre_len).to(self.device)

    def forward(self, flow, inflow, ODRate, adj, external):
        inflow = inflow.to(self.device)
        inflow_week = inflow[:, 0:1, :, :]  # (32, 60, 10)
        inflow_day = inflow[:, 1:2, :, :]
        inflow_time = inflow[:, 2:self.time_lag + 2, :, :]

        # LSC
        inflow_OD = inflow_time.unsqueeze(1)  # (batch, 1, time_step, station_num, station_num)
        # inflow_DO = inflow_time.permute(0, 1, 3, 2).unsqueeze(1)
        inflow_OD = F.relu(self.Conv_OD(inflow_OD))  # (batch, 16, time_step, station_num, station_num)
        # inflow_DO = F.relu(self.Conv_OD(inflow_DO))
        # inflow_2OD = torch.cat((inflow_OD, inflow_DO), dim=1)  # (batch, 32, time, station, station)
        inflow_2OD = self.Conv2D(inflow_OD).permute(0, 2, 1, 3, 4)  # (batch_size, time_step, 16, station_num, station_num)

        # TEC
        OD_time, state = self.clstm(inflow_2OD)  # (batch_size, time_step, hidden_dim, station_num, 32)
        OD_time = OD_time[:, -1, :, :, :]  # (batch, hidden_dim, station_num, station_num)
        OD_TEC = self.Conv_TEC(OD_time)  # (batch, hidden, station, sample))

        # GCC
        OD_time = self.Conv_GCC(OD_TEC)  # (batch, 1 , station, sample)
        DO_time = OD_time.permute(0,1,3,2)

        OD_sim = F.softmax(torch.einsum('ijkl,ijlm->ijkm', OD_time, DO_time), dim=2)
        OD_GCC = torch.matmul(OD_sim, OD_time)  # (batch, 1, station, sample)

        # predict
        OD_output = torch.cat((OD_TEC, OD_GCC), dim=1)  # (batch, hidden +1 , station, sample)
        output = OD_output.permute(0, 2, 3, 1)  # (batch_size, station, sample, hidden+1)

        output = F.tanh(self.linear1(output))  # (64, 1024)
        # output = self.linear2(output)  # (64, 512)
        output = self.linear3(output)  # (64, 276*pre_len)
        output = output.reshape(output.size()[0], self.pre_len, self.station_num, self.sample_num)  # ( 64, 276, pre_len)
        return output








