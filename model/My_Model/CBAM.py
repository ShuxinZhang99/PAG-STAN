import torch
import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)


def Downsample(in_planes, out_planes, expansion, stride):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes * expansion))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(nn.Conv2d(in_channels, 8, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(8, in_channels, kernel_size=1))
        # self.sigmoid = nn.Softmax()

    def forward(self, x):
        avg_out = self.avg_pool(x)  # (batch, 1, 1, 1)
        avg_out = self.fc(avg_out)
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out  # (batch, 1, 1, 1)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        avg_out = torch.mean(x, dim=3, keepdim=True)
        max_out, _ = torch.max(x, dim=3, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=3).permute(0,3,1,2)  # (batch, 2, 1, time_lag)
        x = self.conv1(x).permute(0,2,3,1)  # （batch, 1, time_lag, 1)
        return x


class CBAM_block(nn.Module):
    expansion = 1
    # Input: (batch, 1, ts, station, station)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM_block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(in_channels=planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = out = self.conv1(x)  # (batch, 1, ts, d_model)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # (batch, 1, ts, d_model)
        out = self.bn2(out)

        out = self.ca(out) * out  # (batch, 1, ts, station, station)
        out = self.sa(out) * out  # (batch, 1, ts, station, station)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class TemporalAttention(nn.Module):
#     def __init__(self, in_planes, ratio=4):
#         super(TemporalAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d((61, 1))
#         self.max_pool = nn.AdaptiveMaxPool2d((61, 1))
#
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, 8, kernel_size=1, bias=False),
#                                 nn.ReLU(),
#                                 nn.Conv2d(8, in_planes, kernel_size=1, bias=False))
#         # self.sigmoid = nn.Softmax()
#
#     def forward(self, x):
#         avg_out = self.avg_pool(x)
#         avg_out = self.fc(avg_out)
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return out  # (batch_size, out_channels, 1, 1)
