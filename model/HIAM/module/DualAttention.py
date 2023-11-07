import torch
import math
import sys
import os
import copy
from torch.nn import functional as F
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        :param attn_mask: [batch_size, num_stations, time_steps]
        :param Q: [batch_size, n_heads, station_num, station_num, d_k]
        :param K: [batch_size, n_heads, station_num, station_num, d_k]
        :param V: [batch_size, n_heads, num_stations, num_stations, d_v]
        :return:
        '''
        # scores: [batch_size, n_heads, station, station, station]
        d_k = K.size()[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # if self.attn_mask is not None:
        attn = nn.Softmax(dim=1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, station, station, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, time_lag, n_heads, station):
        super(MultiHeadAttention, self).__init__()
        self.station = station
        self.n_heads = n_heads
        self.d_model = 128
        self.time_lag = time_lag
        self.d_k = self.d_model // self.n_heads
        self.d_v = self.d_model // self.n_heads
        self.W_od = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_do = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)
        self.linear = nn.Linear(self.d_model, self.time_lag)
        self.embedding = nn.Conv2d(in_channels=self.time_lag, out_channels=self.d_model, kernel_size=(1, 1))

    def forward(self, hidden_od, hidden_do):
        '''
        :param attn_mask:
        :param hidden_od: [batch_size, station_num, topk, d_model]
        :param hidden_do: [batch_size, station_num, topk, d_model]
        :return:
        '''
        hidden_od = self.embedding(hidden_od.permute(0, 3, 1, 2))
        hidden_od = hidden_od.permute(0, 2, 3, 1)
        hidden_do = self.embedding(hidden_do.permute(0, 3, 1, 2))
        hidden_do = hidden_do.permute(0, 2, 3, 1)
        residual_od, batch_size, k = hidden_od, hidden_od.size(0), hidden_od.size(2)
        residual_do = hidden_do
        Q_od = K_od = V_od = self.W_od(hidden_od).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        Q_do = K_do = V_do = self.W_do(hidden_do).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Q: [batch_size, n_heads, station, station, d_k]

        # [batch_size, n_heads, seq_len, seq_len]
        context_od, attn_od = ScaledDotProductAttention()(Q_od, K_do, V_do)
        context_do, attn_do = ScaledDotProductAttention()(Q_do, K_od, V_od)
        context_od = context_od.transpose(1, 2).reshape(batch_size, self.station, -1, self.n_heads * self.d_v)  # (batch, station, station, d_model)
        context_do = context_do.transpose(1, 2).reshape(batch_size, self.station, -1, self.n_heads * self.d_v)
        output_od = self.fc(context_od)  # [batch_size, station, topk, d_model]
        output_do = self.fc(context_do)

        output_od = nn.LayerNorm(self.d_model).cuda()(output_od + residual_od)
        output_do = nn.LayerNorm(self.d_model).cuda()(output_do + residual_do)

        output_od = self.linear(output_od)
        output_do = self.linear(output_do)
        return output_od, output_do

