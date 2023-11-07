import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math


class ResNetnetwork(nn.Module):

    def __init__(self, station_num, in_channels, out_channels, stride=1):
        # (4, 1, 3, 12)--(Batch, in_channels, station_num, time_window_size)
        super(ResNetnetwork, self).__init__()
        # # 设置可学习权重
        # self.sub_lossW = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.taxi_lossW = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.bus_lossW = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # # 初始化
        # self.sub_lossW.data.fill_(1)
        # self.taxi_lossW.data.fill_(1)
        # self.bus_lossW.data.fill_(1)

        self.station_num = station_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 等宽卷积，输出还是(station_num,Time_window_size)
            # nn.BatchNorm2d(out_channels),  # 参数即为conv2d的outchannel
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=self.out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 等宽卷积，输出还是(station_num,Time_window_size)
            # nn.BatchNorm2d(out_channels)
        )
        # 确保self.left处理过后，大小与原始输入相同
        self.shortcut = nn.Sequential()
        if stride != 1 or self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                          kernel_size=1, stride=1, bias=False),
                # nn.BatchNorm2d(out_channels)
            )
        # 输出层，将通道数变为分类数
        self.relu = nn.ReLU()

    # def forward(self, x, subway_hat, taxi_hat, bus_hat, criterion):  # train
    def forward(self, x):
        # 卷积操作
        out = self.left(x)
        # 实现残差
        out += self.shortcut(x)
        out = self.relu(out)

        return out
        # 展平成一维
        # b, c, station_num, ts = out.shape
        # out = out.view(b, self.out_channels * self.station_num * self.Time_window_size)
        # # 输出
        # out = self.fc1(out)
        # out = self.relu(out)
        # # out = self.fc2(out)
        # # out = self.fc3(out)
        # out = self.linear(out)
        # out1 = out.T[0].T
        # out2 = out.T[1].T
        # out3 = out.T[2].T
        #
        # # out1 = self.tower1(out)
        # # out1 = out1.squeeze(-1).squeeze(-1)
        # # out2 = self.tower2(out)
        # # out2 = out2.squeeze(-1).squeeze(-1)
        # # out3 = self.tower3(out)  # [4, 1]
        # return out1, out2, out3

        # loss = 1 / (2 * math.exp(self.sub_lossW)) * criterion(out1, subway_hat) \
        #        + 1 / (2 * math.exp(self.taxi_lossW)) * criterion(out2, taxi_hat) \
        #        + 1 / (2 * math.exp(self.bus_lossW)) * criterion(out3, bus_hat) +self.sub_lossW + self.taxi_lossW + self.bus_lossW  # Kendall
        # return loss
