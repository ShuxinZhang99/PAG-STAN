import torch
from torch import nn
import torch.nn.functional as F
from model.HIAM.module.GCGRU import GCGRUCell
from model.HIAM.module.DualAttention import MultiHeadAttention


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.device = device
        self.TopK = 24
        self.head = 4
        self.gcgru = GCGRUCell(in_channels=self.time_lag * self.TopK, out_channels=self.time_lag * self.TopK,
                               num_nodes=station_num, K=2)
        self.dualattn = MultiHeadAttention(time_lag=self.time_lag, n_heads=self.head, station=self.station_num)
        self.linear = nn.Conv2d(in_channels=2 * self.time_lag, out_channels=self.time_lag, kernel_size=(1, 1))
        self.linear1 = nn.Linear(in_features=self.time_lag, out_features=self.pre_len)

    def forward(self, inflow, DO, IncomOD, ODRatio, adj):
        inflow = inflow.to(self.device)
        ODRatio = ODRatio.to(self.device)
        DO = DO.to(self.device)
        IncomOD = IncomOD.to(self.device)
        adj = adj.to(self.device)

        inflow_time = inflow[:, :, 2:self.time_lag + 2]  # (B,N,T)
        IncomOD_time = IncomOD[:, 2:self.time_lag + 2, :, :]
        DO_time = DO[:, 2:self.time_lag + 2, :, :]
        ODRatio_long = ODRatio[:, 0:self.time_lag, :, :]  # (B, T, N, K)
        ODRatio_short = ODRatio[:, self.time_lag:self.time_lag*2, :, :]

        # Unfinish OD
        inflow_time = inflow_time.unsqueeze(0).expand(ODRatio_short.size(-1), *inflow_time.size()).permute(1, 3, 2,0)
        Unfinish_OD_long = inflow_time * ODRatio_long
        Unfinish_OD_short = inflow_time * ODRatio_short
        Unfinish_OD_long = Unfinish_OD_long.permute(0, 2, 3, 1)
        Unfinish_OD_short = Unfinish_OD_short.permute(0, 2, 3, 1)

        OD_time = IncomOD_time.permute(0, 2, 3, 1)  # (Btach, Station, Topk, Time_lag)
        DO_time = DO_time.permute(0, 2, 3, 1)

        # OD branch
        OD_long, _ = self.gcgru(Unfinish_OD_long, adj)
        OD_short, _ = self.gcgru(Unfinish_OD_short, adj)
        his_OD = torch.cat([OD_long, OD_short], dim=-1).permute(0, 3, 1, 2)  # (b, 2*time, s, topk)
        his_OD = self.linear(his_OD).permute(0, 2, 3, 1)  # (b, s, topk, time)
        InOD_time, _ = self.gcgru(OD_time, adj)
        OD_time = InOD_time + his_OD

        # DO branch
        DO_time, _ = self.gcgru(DO_time, adj)  # (batch, station, topk, time_step)
        # Dual Operation
        enc_first_interact_info_od, enc_first_interact_info_do = self.dualattn(OD_time, DO_time)

        enc_od, _ = self.gcgru(enc_first_interact_info_od, adj)
        enc_do, _ = self.gcgru(enc_first_interact_info_do, adj)

        # enc_second_interact_info_od, enc_second_interact_info_do = self.dualattn(enc_od, enc_do)

        output = F.relu(self.linear1(enc_od))
        output = output.reshape(output.size()[0], self.pre_len, self.station_num, self.TopK)
        return output
