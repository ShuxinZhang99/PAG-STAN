import torch
from torch import nn
import torch.nn.functional as F
from model.My_Model.mask_encoder import ST_Masked_Encoder_layer
from model.My_Model.main_ODEst import ODEst
from model.My_Model.Heterogeneous_information_fusion_block import HIFBlock
from model.ResNet.ResNet_layer import ResNetnetwork


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.sample_num = 38
        self.d_model = 256
        self.hidden_dim = 24
        self.device = device
        self.encoder = ST_Masked_Encoder_layer()
        self.HIFB = HIFBlock(time_lag=self.time_lag, station_num=self.station_num, sample_num=self.sample_num, device=device).to(self.device)
        self.biLSTM = nn.LSTM(input_size=self.station_num*self.sample_num, hidden_size=self.d_model, bidirectional=True, batch_first=True, dropout=0.2).to(self.device)
        self.enc_fc = nn.Conv2d(in_channels=self.time_lag, out_channels=2, kernel_size=1).to(self.device)
        self.enc_hid = nn.Linear(in_features=self.sample_num*self.station_num, out_features=self.d_model).to(self.device)
        self.Conv2d = nn.Conv2d(in_channels=2*self.time_lag, out_channels=self.time_lag, kernel_size=3, padding=1).to(self.device)
        self.linear1 = nn.Linear(in_features=2 * self.d_model, out_features=self.station_num * self.sample_num)
        self.linear2 = nn.Linear(in_features=self.time_lag, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=self.pre_len)
        # Estimation
        self.estimation = ODEst(self.time_lag, self.pre_len, self.station_num, self.sample_num, self.device)

    def forward(self, OD, adj, external):  #, inflow, ODCoeff,  external):

        # inflow = inflow.to(self.device)
        adj = adj.to(self.device)
        OD = OD.to(self.device)
        residual = OD_time = OD[:, 2: self.time_lag + 2, :, :]

        # complete_OD_Estimation
        # residual = OD_time = self.estimation(inflow, ODCoeff)
        emb_OD = OD_time.reshape(OD_time.shape[0], OD_time.shape[1], -1)

        # encoder
        enc_output = self.encoder(OD, OD_time, adj)  # (batch, time_lag, station, sample)
        # enc_output = self.enc_fc(enc_output)
        # enc_output = enc_output.reshape(enc_output.shape[0], enc_output.shape[1], -1)
        # enc_output = self.enc_hid(enc_output).permute(1,0,2).contiguous()

        # decoder
        dec_OD, _ = self.biLSTM(emb_OD)  # (batch, time_step, 2*d_model)
        dec_output = F.relu(self.linear1(dec_OD))

        dec_output = dec_output.reshape(dec_output.shape[0], dec_output.shape[1], self.station_num, self.sample_num)

        # fusion
        OD_dense = torch.cat([enc_output, dec_output], dim=1)
        OD_dense = self.Conv2d(OD_dense)

        # External
        External_out = self.HIFB(external)
        dec_output = 0.85 * OD_dense + 0.15 * External_out

        OD_dense = residual + dec_output
        OD_output = OD_dense.permute(0, 2, 3, 1)
        OD_output = F.relu(self.linear2(OD_output))
        OD_output = self.linear3(OD_output)
        # enc_output = F.relu(self.linear1(enc_output))  # (32, 210)
        # output = self.linear2(enc_output)  # (32, 14*pre_len)

        OD_Pre = OD_output.reshape(OD_output.size()[0], self.pre_len, self.station_num, self.sample_num)  # (64, 14, pre_len)

        return OD_Pre
