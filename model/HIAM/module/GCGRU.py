from torch_geometric import nn as gnn
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
import torch
import random
import math
import sys
import os
import copy

from model.HIAM.module.rgcn import RGCNConv
from model.HIAM.module.GCN_layer import GraphConvolution
from torch.nn.functional import dropout


def zoneout(prev_h, next_h, rate, training=True):
    """TODO: Docstring for zoneout.
     :prev_h: TODO
     :next_h: TODO
     :p: when p = 1, all new elements should be droped
         when p = 0, all new elements should be maintained
     :returns: TODO
     """
    if training:
        next_h = (1 - rate) * dropout(next_h - prev_h, rate) + prev_h
    else:
        next_h = rate * prev_h + (1 - rate) * next_h

    return next_h


class KStepRGCN(nn.Module):
    """docstring for KStepRGCN"""

    def __init__(self, in_channels, out_channels, K):
        super(KStepRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        # self.rgcn_layers = nn.ModuleList([
        #     RGCNConv(in_channels,
        #              out_channels,
        #              num_relations,
        #              num_bases,
        #              bias)
        # ] + [
        #     RGCNConv(out_channels,
        #              out_channels,
        #              num_relations,
        #              num_bases,
        #              bias) for _ in range(self.K - 1)
        # ])
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(in_features=in_channels,
                             out_features=out_channels)
        ] + [
            GraphConvolution(in_features=out_channels,
                             out_features=out_channels) for _ in range(self.K - 1)
        ])
        self.mediate_activation = nn.PReLU()

    def forward(self, x, adj):
        for i in range(self.K):
            # x = self.rgcn_layers[i](x=x,
            #                         edge_index=edge_index,
            #                         edge_attr=edge_attr,
            #                         edge_norm=None)
            x = self.gcn_layers[i](x=x, adj=adj)
            # print(x.shape)
            # not final layer, add relu
            if i != self.K - 1:
                x = self.mediate_activation(x)
        return x


class GCGRUCell(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_nodes,
                 dropout_type=None,
                 dropout_prob=0.0,
                 K=1,
                 global_fusion=False):  # num_relations=3, num_bases=3,

        super(GCGRUCell, self).__init__()
        self.num_chunks = 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.num_relations = num_relations
        # self.num_bases = num_bases
        self.num_nodes = num_nodes
        self.global_fusion = global_fusion
        self.cheb_i = KStepRGCN(in_channels,
                                out_channels * self.num_chunks,
                                K=K)  # num_relations=num_relations, num_bases=num_bases,
        self.cheb_h = KStepRGCN(out_channels,
                                out_channels * self.num_chunks,
                                K=K)  # num_relations=num_relations, num_bases=num_bases,

        self.bias_i = Parameter(torch.Tensor(self.out_channels))
        self.bias_r = Parameter(torch.Tensor(self.out_channels))
        self.bias_n = Parameter(torch.Tensor(self.out_channels))
        self.dropout_prob = dropout_prob
        self.dropout_type = dropout_type

        self.reset_parameter()

    def reset_parameter(self):
        init.ones_(self.bias_i)
        init.ones_(self.bias_r)
        init.ones_(self.bias_n)

        if self.global_fusion is True:
            init.ones_(self.bias_i_g)
            init.ones_(self.bias_r_g)
            init.ones_(self.bias_n_g)

    def forward(self, inputs, adj, hidden=None):  # edge_index, edge_attr
        """TODO: Docstring for forward.
                :inputs: TODO
                :hidden: TODO
                :returns: TODO
                """
        topk = inputs.size(2)
        inputs = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        if hidden is None:
            hidden = torch.zeros([inputs.size(0), inputs.size(1), self.out_channels],
                                 dtype=inputs.dtype,
                                 device=inputs.device)
        gi = self.cheb_i(inputs, adj)
        gh = self.cheb_h(hidden, adj)
        # print('gi:', gi.shape)
        # print('hidden:', hidden.shape)
        # print('gh:', gh.shape)
        i_r, i_i, i_n = gi.chunk(3, -1)  # 在给定维度上将输入张量进行分块儿
        h_r, h_i, h_n = gh.chunk(3, -1)

        resetgate = torch.sigmoid(i_r + h_r + self.bias_r)
        inputgate = torch.sigmoid(i_i + h_i + self.bias_i)
        newgate = torch.tanh(i_n + resetgate * h_n + self.bias_n)
        next_hidden = (1 - inputgate) * newgate + inputgate * hidden

        output = next_hidden

        if self.dropout_type == 'zoneout':
            next_hidden = zoneout(prev_h=hidden,
                                  next_h=next_hidden,
                                  rate=self.dropout_prob,
                                  training=self.training)

        elif self.dropout_type == 'dropout':
            next_hidden = F.dropout(next_hidden,
                                    self.dropout_prob,
                                    self.training)
        output = output.view(output.size(0), output.size(1), topk, -1)

        return output, next_hidden
