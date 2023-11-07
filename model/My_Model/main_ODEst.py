import torch
from torch import nn
import torch.nn.functional as F
from model.My_Model.ResNet_layer import ResNetnetwork


class ODEst(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, sample_num, device):
        super(ODEst, self).__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.sample_num = sample_num
        self.device = device
        self.linear1 = nn.Linear(in_features=self.sample_num * 2, out_features=512)
        self.linear = nn.Linear(in_features=self.sample_num, out_features=256)
        self.linear2 = nn.Linear(in_features=512, out_features=self.sample_num)
        # 权重参数初始化
        # self.fuse_weigh_week = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(self.device)
        # self.fuse_weigh_day = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(self.device)
        # self.fuse_weigh_week.data.fill_(0.5)
        # self.fuse_weigh_day.data.fill_(0.5)
        # assert self.fuse_weigh_day + self.fuse_weigh_week == 1

    def forward(self, inflow, ODCoeff):
        inflow = inflow.to(self.device)
        ODCoeff = ODCoeff.to(self.device)

        # inflow_week = inflow[:, :, 0:1].unsqueeze(1)
        # inflow_day = inflow[:, :, 1:2].unsqueeze(1)
        inflow_time = inflow[:, :, 2: self.time_lag + 2]

        ODCoeff_Week = ODCoeff[:, 0: self.time_lag, :, :]
        ODCoeff_Day = ODCoeff[:, self.time_lag: 2 * self.time_lag, :, :]
        # ODCoeff_time = ODCoeff[:, 2 * self.time_lag: 3 * self.time_lag, :, :]

        inflow_time = inflow_time.permute(0, 2, 1)
        ODCoeff_Week = ODCoeff_Week.permute(3, 0, 1, 2)
        ODCoeff_Day = ODCoeff_Day.permute(3, 0, 1, 2)

        inflow_Week = torch.mul(ODCoeff_Week, inflow_time).permute(1, 2, 3, 0)
        inflow_Day = torch.mul(ODCoeff_Day, inflow_time).permute(1, 2, 3, 0)
        inflow_Time = torch.cat([inflow_Week, inflow_Day], dim=3)
        inflow_Time = F.relu(self.linear1(inflow_Time))
        OD_time = self.linear2(inflow_Time)
        # OD_time = 0.5 * inflow_Week + 0.5 * inflow_Day

        return OD_time
