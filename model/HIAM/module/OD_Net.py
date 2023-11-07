from torch_geometric import nn as gnn
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
import torch
import math
import random
import sys
import os
import copy

from model.HIAM.module.GCGRU import GCGRUCell


class ODNet(nn.Module):

    def __init__(self, num_nodes, output_dim, num_units, input_dim, num_rnn_layers, seq_len, horizon):
        super(ODNet, self).__init__()
        self.num_nodes = num_nodes
        self.num_output_dim = output_dim
        self.num_units = num_units
        self.finished_input_dim = input_dim
        self.unfinished_input_dim = input_dim
        self.num_rnn_layers = num_rnn_layers
        self.seq_len = seq_len
        self.horizon = horizon
        # self.num_relations = 1
        self.K = 2
        # self.num_bases = 1

        self.dropout_type = None
        self.dropout_prob = 0.0

        self.global_fusion = False

        self.encoder_first_finished_cells = GCGRUCell(self.finished_input_dim,  # in_channels
                                                      self.num_units,  # out_channels
                                                      self.num_nodes,  # station_num
                                                      self.dropout_type,
                                                      self.dropout_prob,
                                                      K=self.K,
                                                      global_fusion=self.global_fusion)
        self.encoder_first_unfinished_cells = GCGRUCell(self.unfinished_input_dim,
                                                        self.num_units,
                                                        self.num_nodes,
                                                        self.dropout_type,
                                                        self.dropout_prob,
                                                        K=self.K,
                                                        global_fusion=self.global_fusion)
        self.encoder_first_short_his_cells = GCGRUCell(self.unfinished_input_dim,
                                                       self.num_units,
                                                       self.num_nodes,
                                                       self.dropout_type,
                                                       self.dropout_prob,
                                                       K=self.K,
                                                       global_fusion=self.global_fusion)

        self.unfinished_output_layer = nn.Conv2d(in_channels=self.num_units * 2,
                                                 out_channels=self.num_units,
                                                 kernel_size=1)
        self.unfinished_hidden_layer = nn.Conv2d(in_channels=self.num_units * 2,
                                                 out_channels=self.num_units,
                                                 kernel_size=1)

        self.encoder_second_cells = nn.ModuleList([GCGRUCell(self.num_units,
                                                             self.num_units,
                                                             self.num_nodes,
                                                             self.dropout_type,
                                                             self.dropout_prob,
                                                             K=self.K,
                                                             global_fusion=self.global_fusion)
                                                   for _ in range(self.num_rnn_layers - 1)])
        self.decoder_first_cells = GCGRUCell(self.finished_input_dim,
                                             self.num_units,
                                             self.num_nodes,
                                             self.dropout_type,
                                             self.dropout_prob,
                                             K=self.K,
                                             global_fusion=self.global_fusion)
        self.decoder_second_cells = nn.ModuleList([GCGRUCell(self.num_units,
                                                             self.num_units,
                                                             self.num_nodes,
                                                             self.dropout_type,
                                                             self.dropout_prob,
                                                             K=self.K,
                                                             global_fusion=self.global_fusion)
                                                   for _ in range(self.num_rnn_layers - 1)])

        self.output_type = 'fc'
        if self.output_type == 'fc':
            self.output_layer = nn.Linear(self.num_units, self.num_output_dim)

    def encoder_first_layer(self, IncomOD, UnLOD, UnSOD, adj, finished_hidden, long_his_hidden, short_his_hidden):
        finished_out, finished_hidden = self.encoder_first_finished_cells(inputs=IncomOD,
                                                                          adj=adj,
                                                                          hidden=finished_hidden)  # GCGRUCell
        enc_first_hidden = finished_hidden
        enc_first_out = finished_out

        long_his_out, long_his_hidden = self.encoder_first_unfinished_cells(inputs=UnLOD,
                                                                            adj=adj,
                                                                            hidden=long_his_hidden)  # GCGRUCell

        short_his_out, short_his_hidden = self.encoder_first_unfinished_cells(inputs=UnSOD,
                                                                              adj=adj,
                                                                              hidden=short_his_hidden)  # GCGRUCell

        hidden_fusion = torch.cat([long_his_hidden, short_his_hidden], -1).permute(0, 3, 1, 2)

        long_his_weight = torch.sigmoid(self.unfinished_hidden_layer(hidden_fusion)).permute(0, 2, 3, 1)
        short_his_weight = torch.sigmoid(self.unfinished_output_layer(hidden_fusion)).permute(0, 2, 3, 1)

        unfinished_hidden = long_his_weight * long_his_hidden + short_his_weight * short_his_hidden
        unfinished_out = long_his_weight * long_his_out + short_his_weight * short_his_out

        enc_first_out = enc_first_out + unfinished_out
        enc_first_hidden = enc_first_hidden + unfinished_hidden

        return enc_first_out, finished_hidden, long_his_hidden, short_his_hidden, enc_first_hidden

    def encoder_second_layer(self,
                             index,
                             first_out,
                             adj,
                             enc_second_hidden):
        enc_second_out, enc_second_hidden = self.encoder_second_cells[index](inputs=first_out,
                                                                             adj=adj,
                                                                             hidden=enc_second_hidden)  # GCGRU
        return enc_second_out, enc_second_hidden

    def decoder_first_layer(self,
                            decoder_input,
                            adj,
                            dec_first_hidden):
        dec_first_out, dec_first_hidden = self.decoder_first_cells(inputs=decoder_input,
                                                                   adj=adj,
                                                                   hidden=dec_first_hidden)
        return dec_first_out, dec_first_hidden

    def decoder_second_layer(self,
                             index,
                             decoder_first_out,
                             adj,
                             dec_second_hidden):
        dec_second_out, dec_second_hidden = self.decoder_second_cells[index](inputs=decoder_first_out,
                                                                             adj=adj,
                                                                             hidden=dec_second_hidden)
        return dec_second_out, dec_second_hidden
