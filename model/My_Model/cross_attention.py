import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K):
        '''
        :param attn_mask: [batch_size, num_stations, time_steps]
        :param Q: [batch_size, n_heads, station_num, station_num, d_k]
        :param K: [batch_size, n_heads, station_num, station_num, d_k]
        :param V: [batch_size, n_heads, num_stations, num_stations, d_v]
        :return:
        '''
        # scores: [batch_size, n_heads, station, station, station]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # if self.attn_mask is not None:
        attn = nn.Softmax(dim=1)(scores)
        # context = torch.matmul(attn, V)  # [batch_size, n_heads, station, station, d_v]
        return attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, station):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.station = station

        self.W_day = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_week = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_time = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.linear1 = nn.Conv2d(in_channels=n_heads * 3, out_channels=n_heads, kernel_size=1)
        self.linear2 = nn.Conv2d(in_channels=n_heads, out_channels=n_heads, kernel_size=1)

    def forward(self, flow_time, flow_day, flow_week):
        '''
        :param flow_time: [batch_size, len_q(station_number), d_model]
        :param flow_day: [batch_size, len_k, d_model]
        :param flow_week: [batch_size, len_v, d_model]
        :return:
        '''
        residual, batch_size = flow_time, flow_time.size(0)  # (batch, time_lag, d_model)
        # (batch, station, station, d_model)
        Q_day = K_day = self.W_day(flow_day).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Q: [batch_size, n_heads, station, station, d_k]
        Q_week = K_week = self.W_week(flow_week).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        Q_time = K_time = V_time = self.W_time(flow_time).reshape(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # attn_scores: [batch_size, n_heads, station * sample, station * sample]
        attn_day = ScaledDotProductAttention(self.d_k)(Q_day, K_day)
        attn_week = ScaledDotProductAttention(self.d_k)(Q_week, K_week)
        attn_time = ScaledDotProductAttention(self.d_k)(Q_time, K_time)
        attn_fuse = torch.cat([attn_week, attn_day, attn_time], dim=1)  # (batch, n_head*3, self.time_lag, self.d_v)
        attn_fuse = self.linear1(attn_fuse)
        # attn = self.linear2(attn_fuse)
        #
        context = torch.matmul(attn_fuse, V_time)  # [batch_size, n_heads, station_D*station_O, d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)  # (batch, station, station, d_model)
        output = self.fc(context)  # [batch_size, time_lag, d_model]

        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn_fuse

