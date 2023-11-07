import torch
from torch import nn
import torch.nn.functional as F


class DAMGNN(nn.Module):
    def __init__(self, station_num, in_channels, out_channels, adaptadj=True, dropout=0.1):
        super(DAMGNN, self).__init__()
        self.station_num = station_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adaptadj = adaptadj
        self.Theta = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.num_transition_matrix = 4
        self.linear = nn.Linear(out_channels * self.num_transition_matrices, out_channels)
        self.dropout = nn.Dropout(dropout)

        if self.adaptadj:
            self.nodevec1 = nn.Parameter(torch.randn(self.station_num, 32), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(32, self.station_num), requires_grad=True)

    def forward(self, ODFlow, adj, OCoeff_Graph, DCoeff_Graph):
        if self.addaptadj is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # (N, N)
            new_adj = adj + adp

        batch_size, in_channels, origin_stations, destination_stations = ODFlow .shape  # Coeff_Graph

        # O_view
        x = ODFlow.permute(0, 3, 2, 1)  # (batch_size, destination_stations, origin_stations, in_channels)
        x_OD = torch.matmul(new_adj, x)
        y_OD = torch.matmul(OCoeff_Graph, x)
        O_adj = F.relu(self.Theta(x_OD)).reshape(batch_size, -1, origin_stations, destination_stations)
        O_Coeff = F.relu(self.Theta(y_OD)).reshape(batch_size, -1, origin_stations, destination_stations)

        # D_view
        y = ODFlow.permute(0, 2, 3, 1)  # (batch, origin_stations, destination_stations, in_channels)
        x_DO = torch.matmul(new_adj, y)
        y_DO = torch.matmul(DCoeff_Graph, y)
        D_adj = F.relu(self.Theta(x_DO)).reshape(batch_size, -1, origin_stations, destination_stations)
        D_Coeff = F.relu(self.Theta(y_DO)).reshape(batch_size, -1, origin_stations, destination_stations)

        # Concat
        Multi_ODGraph = torch.cat([O_adj, D_adj, O_Coeff, D_Coeff], dim=1).permute(0, 2, 3, 1)
        ODGraph = self.linear(Multi_ODGraph)
        ODGraph = self.dropout(ODGraph)

        return ODGraph
