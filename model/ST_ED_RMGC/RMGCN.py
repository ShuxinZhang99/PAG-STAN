import torch
from torch import nn
import torch.nn.functional as F
from model.ST_ED_RMGC.GCN_layers import GraphConvolution
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class RMGCN_Block(Module):
    def __init__(self, time_lag, station_num, graph_num):
        super(RMGCN_Block, self).__init__()
        self.time_lag = time_lag
        self.station_num = station_num
        self.graph_num = graph_num
        self.ODPairs = 1600
        self.gcn_1 = GraphConvolution(in_features=self.time_lag, out_features=32)
        self.gcn_2 = GraphConvolution(in_features=32, out_features=32)
        self.fc = nn.Linear(in_features=32*2, out_features=32)
        self.linear1 = nn.Linear(in_features=32, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.BN = nn.BatchNorm1d(self.station_num * self.station_num)

    def forward(self, x, O_adj, D_adj):
        x_1 = self.gcn_1(x, O_adj)  # (Batch, N, 64)
        x_2 = self.gcn_1(x, D_adj)  # (batch, N, 32)
        x_fuse = torch.cat([x_1, x_2], dim=-1)
        residual = x_fuse = F.relu(self.fc(x_fuse))
        x_11 = self.gcn_2(x_fuse, O_adj)
        x_22 = self.gcn_2(x_fuse, D_adj)
        x_fuse = torch.cat([x_11, x_22], dim=-1)
        x_all = F.relu(self.fc(x_fuse))
        x = self.BN(x_all + residual)

        # output = output.reshape(output.size()[0], -1)  # (batch, N*128)
        output = F.relu(self.linear1(x))  # (batch, N, 1)
        output = self.linear2(output)

        return output

