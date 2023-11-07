import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model.ST_ED_RMGC.RMGCN import RMGCN_Block
from model.ST_ED_RMGC.LSTM_layer import Lstm_reg
from model.ST_ED_RMGC.GCN_layers import GraphConvolution


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.ODPairs = 1600
        self.device = device
        self.graph_num = 2
        self.rmgcn = RMGCN_Block(time_lag=self.time_lag, station_num=station_num, graph_num=self.graph_num)
        self.mgcn = GraphConvolution(in_features=1, out_features=1)
        self.lstm = Lstm_reg(input_size=self.station_num * self.station_num, hidden_size=1024, output_size=self.station_num * self.station_num)
        self.linear1 = nn.Linear(in_features=2, out_features=1).to(self.device)
        self.fc = nn.Linear(in_features=2, out_features=1)
        self.BN = nn.BatchNorm1d(self.station_num * self.station_num)
        self.linear2 = nn.Linear(in_features=1, out_features=256).to(self.device)
        self.linear3 = nn.Linear(in_features=256, out_features=self.pre_len).to(self.device)
        self.linear4 = nn.Linear(in_features=self.time_lag, out_features=1)

    def forward(self, inflow, O_adj, D_adj):
        inflow = inflow.to(self.device)  # (Batch, N, 12)
        inflow_time = inflow[:, :, 2:2+self.time_lag]
        O_adj = O_adj.to(self.device)  # (N*K, N)
        D_adj = D_adj.to(self.device)

        # Encoder
        spatial_output = self.rmgcn(inflow_time, O_adj, D_adj)  # (Batch, 900)
        temporal_output = self.lstm(inflow_time.permute(0, 2, 1)).permute(0, 2, 1)  # (Batch, Station * station, 1)
        temporal_output = self.linear4(temporal_output)
        encoder_output = torch.cat((spatial_output, temporal_output), dim=-1)  # (Batch, station * station, 2)

        # Decoder
        residual = decoder_input = F.relu(self.linear1(encoder_output))  # (Batch, N, 1)
        x_O = self.mgcn(decoder_input, O_adj)
        x_D = self.mgcn(decoder_input, D_adj)
        x_fuse = torch.cat([x_O, x_D], dim=-1)
        x = F.relu(self.fc(x_fuse))
        decoder_output = self.BN((x + residual))  # (batch, N, 1)

        # Prediction
        output = F.relu(self.linear2(decoder_output))  # (32, 210)
        output = self.linear3(output) # (32, 64)
        # output = self.linear3(output)  # (32, 14*pre_len)
        output = output.reshape(output.size()[0], -1, self.pre_len)  # ( 64, 14, pre_len)
        return output
