from torch_geometric import nn as gnn
from torch import nn
import torch
import sys
import os

from model.HIAM.module.GCGRU import GCGRUCell


class DONet(nn.Module):
    def __init__(self, num_nodes, output_dim, rnn_units, input_dim, num_rnn_layers, seq_len, horizon):
        super(DONet, self).__init__()
        self.num_nodes = num_nodes
        self.num_output_dim = output_dim
        self.num_units = rnn_units
        self.num_finished_input_dim = input_dim
        self.num_rnn_layers = num_rnn_layers
        self.seq_len = seq_len
        self.horizon = horizon
        # self.num_relations = 1
        self.K = 2
        # self.num_bases = 1
        self.use_curriculum_learning = 'use_curriculum_learning'
        # self.cl_decay_steps
        self.dropout_type = None
        self.dropout_prob = 0.0
        self.use_input = True
        self.global_fusion = False

        self.encoder_first_cells = GCGRUCell(self.num_finished_input_dim,  # in_channel
                                             self.num_units,  # out_channel
                                             self.num_nodes,
                                             self.dropout_type,
                                             self.dropout_prob,
                                             K=self.K,
                                             global_fusion=self.global_fusion)
        self.encoder_second_cells = nn.ModuleList([GCGRUCell(self.num_units,
                                                             self.num_units,
                                                             self.num_nodes,
                                                             self.dropout_type,
                                                             self.dropout_prob,
                                                             K=self.K,
                                                             global_fusion=self.global_fusion)
                                                   for _ in range(self.num_rnn_layers - 1)])

        self.decoder_first_cells = GCGRUCell(self.num_finished_input_dim,
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

    def encoder_first_layer(self, DO, adj, enc_first_hidden):
        enc_first_out, enc_first_hidden = self.encoder_first_cells(inputs=DO,
                                                                   adj=adj,
                                                                   hidden=enc_first_hidden)
        return enc_first_out, enc_first_hidden

    def encoder_second_layer(self, index, encoder_first_out, adj, enc_second_hidden):
        enc_second_out, enc_second_hidden = self.encoder_second_cells[index](inputs=encoder_first_out,
                                                                             adj=adj,
                                                                             hidden=enc_second_hidden)
        return enc_second_out, enc_second_hidden

    def decoder_first_layer(self, decoder_input, adj, dec_first_hidden):
        dec_first_out, dec_first_hidden = self.decoder_first_cells(inputs=decoder_input,
                                                                   adj=adj,
                                                                   hidden=dec_first_hidden)
        return dec_first_out, dec_first_hidden

    def decoder_second_layer(self, index, decoder_first_out, adj, dec_second_hidden):
        dec_second_out, dec_second_hidden = self.decoder_second_cells[index](inputs=decoder_first_out,
                                                                             adj=adj,
                                                                             hidden=dec_second_hidden)
        return dec_second_out, dec_second_hidden
