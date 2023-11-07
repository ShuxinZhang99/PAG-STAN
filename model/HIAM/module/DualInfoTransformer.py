from torch_geometric import nn as gnn
from torch import nn
import torch
import math
import sys
import os
import copy
from torch.nn import functional as F


class DualInfoTransformer(nn.Module):
    def __init__(self, h, d_nodes, d_model, d_channel):
        "Take in model size and number of heads."
        super(DualInfoTransformer, self).__init__()
        # assert d_model % h == 0

        self.d_nodes = d_nodes
        self.d_model = d_model
        self.d_channel = d_channel
        self.d_k = d_channel // h
        self.h = h

        self.od_linears = self.clones(
            nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=d_channel, kernel_size=1),
                nn.PReLU(d_channel),
                nn.Conv1d(in_channels=d_channel, out_channels=d_channel, kernel_size=1),
                nn.PReLU(d_channel)), 3)
        self.do_linears = self.clones(
            nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=d_channel, kernel_size=1),
                nn.PReLU(d_channel),
                nn.Conv1d(in_channels=d_channel, out_channels=d_channel, kernel_size=1),
                nn.PReLU(d_channel)),3)
        self.od_conv = nn.Sequential(
            nn.Conv1d(in_channels=d_channel, out_channels=d_channel, kernel_size=1),
            nn.PReLU(d_channel),
            nn.Conv1d(in_channels=d_channel, out_channels=d_model, kernel_size=1),
            nn.PReLU(d_model)
        )
        self.do_conv = nn.Sequential(
            nn.Conv1d(in_channels=d_channel, out_channels=d_channel, kernel_size=1),
            nn.PReLU(d_channel),
            nn.Conv1d(in_channels=d_channel, out_channels=d_model, kernel_size=1),
            nn.PReLU(d_model)
        )

    def clones(self, module, N):
        "Produce N identical layers"
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def attention(self, query, key, value):
        "compute scaled dot-product aatention"
        d_k = query.size(-1)  # batch, heads, num_nodes, num_units / heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # batch, heads, num_nodes, num_nodes
        p_attn = F.softmax(scores, dim=-1)  # batch, heads, num_nodes, num_nodes
        return torch.matmul(p_attn, value)  # batch, heads, num_nodes, num_units / heads

    def MultiHeadAttention(self, hid_od, hid_do):
        hid_od = hid_od.view(-1, self.d_model, self.d_nodes)
        hid_do = hid_do.view(-1, self.d_model, self.d_nodes)
        nbatches = hid_od.size(0)

        # do all the linear projections in batch from d_model => h * d_k
        odquery, odkey, odvalue = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
             for l, x in zip(self.od_linears, (hid_od, hid_od, hid_od))]  # batch, heads, num_nodes, num_units / heads
        doquery, dokey, dovalue = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
             for l, x in zip(self.do_linears, (hid_do, hid_do, hid_do))]

        # apply attention on all the projectiond vectors in batch
        attn_od = self.od_conv(
            self.attention(query=odquery,
                           key=dokey,
                           value=dovalue).transpose(-2, -1).contiguous().view(-1, self.d_channel ,self.d_nodes)
        )  # batch, heads, 1, num_units / heads
        attn_do = self.do_conv(
            self.attention(query=doquery,
                           key=odkey,
                           value=odvalue).transpose(-2, -1).contiguous().view(-1, self.d_channel, self.d_nodes)
        )

        return attn_od.view(nbatches, self.d_nodes, -1, self.d_model), attn_do.view(nbatches, self.d_nodes, -1, self.d_model)

    def forward(self, hidden_states_od, hidden_states_do):
        return self.MultiHeadAttention(hidden_states_od, hidden_states_do)