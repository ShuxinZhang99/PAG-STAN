from torch_geometric import nn as gnn
from torch import nn
import torch
import random
import math
import sys
import os

from model.HIAM.module.OD_Net import ODNet
from model.HIAM.module.DO_Net import DONet
from model.HIAM.module.DualInfoTransformer import DualInfoTransformer


class Net(nn.Module):
    def __init__(self, num_nodes, output_dim, rnn_units, input_dim, num_rnn_layers, seq_len, horizon,
                 head=4, channel=256, use_curriculum_learning=False, use_input=True):
        super(Net, self).__init__()

        self.OD = ODNet(num_nodes, output_dim, rnn_units, input_dim, num_rnn_layers, seq_len, horizon)
        self.DO = DONet(num_nodes, output_dim, rnn_units, input_dim, num_rnn_layers, seq_len, horizon)

        self.num_nodes = num_nodes,  # station_num
        self.num_output_dim = output_dim,  # net_output
        self.num_units = rnn_units,  # GCGRU_Out_Channel
        self.num_finished_input_dim = input_dim  # GCGRU_input_dim
        self.num_unfinished_input_dim = input_dim
        self.num_rnn_layers = num_rnn_layers

        self.seq_len = seq_len
        self.horizon = horizon
        self.head = head
        self.d_channel = channel

        self.use_curriculum_learning = use_curriculum_learning
        self.use_input = use_input

        self.mediate_activation = nn.PReLU(self.num_units)

        self.global_step = 0
        self.cl_decay_steps = 2

        self.encoder_first_interact = DualInfoTransformer(
            h=self.head,
            d_nodes=self.num_nodes,
            d_model=self.num_units,
            d_channel=self.d_channel)
        self.decoder_first_interact = DualInfoTransformer(
            h=self.head,
            d_nodes=self.num_nodes,
            d_model=self.num_units,
            d_channel=self.d_channel)
        self.encoder_second_interact = nn.ModuleList([DualInfoTransformer(
            h=self.head,
            d_nodes=self.num_nodes,
            d_model=self.num_units,
            d_channel=self.d_channel) for _ in range(self.num_rnn_layers - 1)])
        self.decoder_second_interact = nn.ModuleList([DualInfoTransformer(
            h=self.head,
            d_nodes=self.num_nodes,
            d_model=self.num_units,
            d_channel=self.d_channel) for _ in range(self.num_rnn_layers - 1)])

    @staticmethod
    def inverse_sigmoid_scheduler_sampling(step, k):
        """TODO: Docstring for linear_scheduler_sampling.
                :returns: TODO
                """
        try:
            return k / (k + math.exp(step / k))
        except OverflowError:
            return float('inf')

    def encoder_od_do(self, IncomOD, UnLOD, UnSOD, DO, adj):
        """
               Encodes input into hidden state on one branch for T steps.
               Return: hidden state on one branch."""
        enc_hidden_od = [None] * self.num_rnn_layers
        enc_hidden_do = [None] * self.num_rnn_layers

        finished_hidden_od = None
        long_his_hidden_od = None
        short_his_hidden_od = None

        encoder_first_out_od, finished_hidden_od, long_his_hidden_od, short_his_hidden_od, enc_first_hidden_od = \
            self.OD.encoder_first_layer(
                IncomOD,
                UnLOD,
                UnSOD,
                adj,
                finished_hidden_od,
                long_his_hidden_od,
                short_his_hidden_od)

        enc_first_out_do, enc_first_hidden_do = self.DO.encoder_first_layer(
            DO,
            adj,
            enc_hidden_do[0])

        print(enc_first_hidden_od.shape, enc_first_hidden_do.shape)
        enc_first_interact_info_od, enc_first_interact_info_do = self.encoder_first_interact(
            enc_first_hidden_od,
            enc_first_hidden_do)

        enc_hidden_od[0] = enc_first_hidden_od + enc_first_interact_info_od
        enc_hidden_do[0] = enc_first_hidden_do + enc_first_interact_info_do

        enc_mid_out_od = encoder_first_out_od + enc_first_interact_info_od
        enc_mid_out_do = enc_first_out_do + enc_first_interact_info_do

        for index in range(self.num_rnn_layers - 1):
            enc_mid_out_od = self.mediate_activation(enc_mid_out_od)
            enc_mid_out_do = self.mediate_activation(enc_mid_out_do)

            enc_mid_out_od, enc_mid_hidden_od = self.OD.encoder_second_layer(
                index,
                enc_mid_out_od,
                adj,
                enc_hidden_od)
            enc_mid_out_do, enc_mid_hidden_do = self.DO.encoder_second_layer(
                index,
                enc_mid_out_do,
                adj,
                enc_hidden_do)

            enc_mid_interact_info_od, enc_mid_interact_info_do = self.encoder_second_interact[index](
                enc_mid_hidden_od,
                enc_mid_hidden_do)

            enc_hidden_od[index + 1] = enc_mid_hidden_od + enc_mid_interact_info_od
            enc_hidden_do[index + 1] = enc_mid_hidden_do + enc_mid_interact_info_do

        return enc_hidden_od, enc_hidden_do

    def scheduled_sampling(self, out, label, GO, use_truth_sequence=False):
        if use_truth_sequence:
            # Feed the prev label as the next input
            decoder_input = label
        else:
            # detach from history as input
            decoder_input = out.detach()
        if not self.use_input:
            decoder_input = GO.detach()

        return decoder_input

    def decoder_od_do(self, IncomOD, adj, enc_hidden_od, enc_hidden_do):
        predictions_od = []
        predictions_do = []

        GO_od = torch.zeros([enc_hidden_od[0].size()[0], enc_hidden_od[0].size()[1], enc_hidden_od[0].size()[2], self.num_output_dim],
                            dtype=enc_hidden_od[0].dtype,
                            device=enc_hidden_od[0].device)
        GO_do = torch.zeros([enc_hidden_do[0].size()[0], enc_hidden_do[0].size()[1], enc_hidden_do[0].size()[2], self.num_output_dim],
                            dtype=enc_hidden_do[0].dtype,
                            device=enc_hidden_do[0].device)

        dec_input_od = GO_od
        dec_hidden_od = enc_hidden_od

        dec_input_do = GO_do
        dec_hidden_do = enc_hidden_do

        for t in range(self.horizon):
            dec_first_out_od, dec_first_hidden_od = self.OD.decoder_first_layer(
                decoder_input=dec_input_od,
                adj=adj,
                dec_first_hidden=dec_hidden_od[0])
            dec_first_out_do, dec_first_hidden_do = self.DO.decoder_first_layer(
                decoder_input=dec_input_do,
                adj=adj,
                dec_first_hidden=dec_hidden_do[0])
            dec_first_interact_info_od, dec_first_interact_info_do = self.decoder_first_interact(
                dec_first_hidden_od,
                dec_first_hidden_do)

            dec_hidden_od[0] = dec_first_hidden_od + dec_first_interact_info_od
            dec_hidden_do[0] = dec_first_hidden_do + dec_first_interact_info_do
            dec_mid_out_od = dec_first_out_od + dec_first_interact_info_od
            dec_mid_out_do = dec_first_out_do + dec_first_interact_info_do

            for index in range(self.num_rnn_layers - 1):
                dec_mid_out_od = self.mediate_activation(dec_mid_out_od)
                dec_mid_out_do = self.mediate_activation(dec_mid_out_do)
                dec_mid_out_od, dec_mid_hidden_od = self.OD.decoder_second_layer(
                    index,
                    dec_mid_out_od,
                    adj,
                    dec_hidden_od[index])
                dec_mid_out_do, dec_mid_hidden_do = self.DO.decoder_second_layer(
                    index,
                    dec_mid_out_do,
                    adj,
                    dec_hidden_do[index])

                dec_second_interact_info_od, dec_second_interact_info_do = self.decoder_second_interact[index](
                    dec_mid_hidden_od,
                    dec_mid_hidden_do)

                dec_hidden_od[index + 1] = dec_mid_hidden_od + dec_second_interact_info_od
                dec_hidden_do[index + 1] = dec_mid_hidden_do + dec_second_interact_info_do
                dec_mid_out_od = dec_mid_out_od + dec_second_interact_info_od
                dec_mid_out_do = dec_mid_out_do + dec_second_interact_info_do

            nbatch = dec_mid_out_od.size()[0]
            dec_mid_out_od = dec_mid_out_od.reshape(-1, self.num_units)
            dec_mid_out_do = dec_mid_out_do.reshape(-1, self.num_units)

            dec_mid_out_od = self.OD.output_layer(dec_mid_out_od).view(nbatch, self.num_nodes, -1, self.num_output_dim)
            dec_mid_out_do = self.DO.output_layer(dec_mid_out_do).view(nbatch, self.num_nodes, -1, self.num_output_dim)

            predictions_od.append(dec_mid_out_od)
            predictions_do.append(dec_mid_out_do)

            dec_input_od = self.scheduled_sampling(dec_mid_out_od, IncomOD, GO_od)
            dec_input_do = self.scheduled_sampling(dec_mid_out_do, IncomOD, GO_do)

        if self.training:
            self.global_step += 1

        return torch.stack(predictions_od).transpose(0, 1), torch.stack(predictions_do).transpose(0, 1)

    def forward(self, IncomOD, UnLOD, UnSOD, DO, adj):
        enc_hiddens_od, enc_hiddens_do = self.encoder_od_do(IncomOD=IncomOD,
                                                            UnLOD=UnLOD,
                                                            UnSOD=UnSOD,
                                                            DO=DO,
                                                            adj=adj)
        predictions_od, predictions_do = self.decoder_od_do(IncomOD,
                                                            adj,
                                                            enc_hiddens_od,
                                                            enc_hiddens_do)

        return predictions_od, predictions_do
