import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.My_Model.cross_attention import ScaledDotProductAttention, MultiHeadAttention
from model.My_Model.cross_spraseattn import FullAttention, ProbAttention, AttentionLayer
from model.My_Model.GraphLSTM import GConvLSTM
from model.ConvLSTM.ConvLSTM import *
from model.My_Model.CBAM import CBAM_block
from model.My_Model.GCN_layer import GraphConvolution

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

station_num = 62
time_lag = 12
pre_len = 1
d_model = 256
d_ff = 128
d_k = d_v = 64
n_heads = 4
n_layers = 1
sample_num = 38
hidden_dim = 24


class MaskOD(nn.Module):
    def __init__(self, prob_replace_rand=0.8):
        super(MaskOD, self).__init__()
        self.prob_replace_rand = prob_replace_rand
        self.device = device

    def forward(self, OD_FLOW):
        OD_FLOW = OD_FLOW.to(self.device)
        labels_mask = torch.full(OD_FLOW.shape, self.prob_replace_rand).to(self.device)
        labels_unmask = torch.zeros(OD_FLOW.shape).to(self.device)
        mask = torch.where(OD_FLOW == 0, labels_mask, labels_unmask).to(self.device)  # 将流量为0的OD给定概率0.8
        masked_indices = torch.bernoulli(mask).bool().to(self.device)  # 对于OD流量小于0的OD以0.8概率抽取出来，以便后续赋予随机值
        random_words = torch.rand(size=OD_FLOW.shape).to(self.device)

        OD_FLOW[masked_indices] = random_words[masked_indices]

        return OD_FLOW


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class ST_Masked_Encoder_layer(nn.Module):
    def __init__(self):
        super(ST_Masked_Encoder_layer, self).__init__()
        self.station_num = station_num
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.device = device
        self.sample_num = sample_num
        self.input_size = (self.station_num, self.sample_num)
        self.gclstm = GConvLSTM(input_size=self.input_size, input_dim=1, hidden_dim=hidden_dim, num_nodes=station_num, num_layers=2)
        # self.bilstm = nn.LSTM(input_size=self.station_num*self.sample_num,
        # hidden_size=self.station_num*self.sample_num, bidirectional=True, batch_first=True)
        self.Attn = AttentionLayer(
            ProbAttention(factor=5, n_heads=n_heads, attention_dropout=0.1, output_attention=False), d_model=d_model,
            n_heads=n_heads)
        self.pos_fnn = PoswiseFeedForwardNet()
        # self.mask_operation = MaskOD().to(self.device)
        self.cbam_his = CBAM_block(inplanes=1, planes=self.time_lag).to(self.device)
        self.cbam_time = CBAM_block(inplanes=self.time_lag, planes=self.time_lag).to(self.device)
        # self.embedding = GraphConvolution(in_features=hidden_dim * sample_num,
        #                                   out_features=d_model, num_nodes=self.station_num).to(self.device)
        self.embedding_fc = nn.Conv1d(in_channels=sample_num*hidden_dim*station_num, out_channels=d_model, kernel_size=3, padding=1).to(self.device)
        # self.conv2d = nn.Conv2d(in_channels=1, out_channels=self.time_lag, kernel_size=(3,3), padding=1).to(self.device)
        # self.embedding_his = nn.Conv1d(in_channels=2*sample_num*station_num, out_channels=d_model, kernel_size=1).to(self.device)
        # self.reg = nn.Conv1d(in_channels=3*d_model, out_channels=d_model, kernel_size=1).to(self.device)
        self.linear1 = nn.Linear(in_features=d_model, out_features=station_num * sample_num).to(self.device)
        # self.linear2 = nn.Linear(in_features=d_model, out_features=2048).to(self.device)
        # self.linear3 = nn.Linear(in_features=hidden_dim, out_features=256)
        # self.linear4 = nn.Linear(in_features=256, out_features=self.time_lag * self.sample_num)
        self.fc = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(in_features=2 * self.station_num * self.sample_num, out_features=self.station_num*self.sample_num)

    def forward(self, ODFlow, Complete_OD, adj):
        # Masked_OD = self.mask_operation(ODFlow)
        # Masked_OD_Time = self.mask_operation(Complete_OD)
        OD_Weekly = ODFlow[:, 0:1, :, :]  # (batch, 1, station, station)
        OD_Daily = ODFlow[:, 1:2, :, :]  # (batch, 1, station, station)
        residual = Complete_OD
        # OD_Time = Masked_OD[:, 2:self.time_lag + 2, :, :]  # (batch, time_lag, station, station)

        # CBAM
        OD_Weekly = self.cbam_his(OD_Weekly).unsqueeze(2)  # (batch, time_lag, 1, station, sample)
        # OD_Weekly = OD_Weekly.reshape(OD_Weekly.size()[0], OD_Weekly.size()[1], -1)
        OD_Daily = self.cbam_his(OD_Daily).unsqueeze(2)
        # OD_Daily = OD_Daily.reshape(OD_Daily.size()[0], OD_Daily.size()[1], -1)
        OD_Time = self.cbam_time(Complete_OD).unsqueeze(2)

        # GCLSTM
        OD_Week_hidden, (h_week, c_week) = self.gclstm(OD_Weekly, adj)  # (batch, time, 2*station*sample)  (2, batch, station*sample)
        OD_Day_hidden, (h_day, c_day) = self.gclstm(OD_Daily, adj)
        # print(OD_Week_hidden.shape, h_week.shape)
        # h_t = 0.25 * h_week + 0.75 * h_day
        # c_t = 0.25 * c_week + 0.75 * c_day
        h_t = torch.cat([h_week, h_day], dim=1)  # (batch, 2*hidden_dim, station, sample)
        h_t = self.fc(h_t)  # (batch, station, sample_num)
        # h_t = h_t.reshape(h_t.size()[0], h_t.size()[1], station_num, sample_num)
        c_t = torch.cat([c_week, c_day], dim=1)
        c_t = self.fc(c_t)  # (batch, station, sample_num)
        # c_t = c_t.reshape(c_t.size()[0], h_t.size()[1], station_num, sample_num)
        OD_Time_hidden, _ = self.gclstm(OD_Time, adj, (h_t, c_t))
        # 最后时刻的隐藏状态

        # encoder
        OD_Weekly = OD_Week_hidden.reshape(OD_Week_hidden.size()[0], -1, self.time_lag)
        OD_Daily = OD_Day_hidden.reshape(OD_Day_hidden.size()[0], -1, self.time_lag)
        OD_Time = OD_Time_hidden.reshape(OD_Time_hidden.size()[0], -1, self.time_lag)
        OD_Time = self.embedding_fc(OD_Time)  # (batch, d_model, time_lag)
        OD_Weekly = self.embedding_fc(OD_Weekly)
        OD_Daily = self.embedding_fc(OD_Daily)
        # OD_His = 0.25 * OD_Weekly + 0.75 * OD_Daily

        OD_context, enc_attn = self.Attn(OD_Time, OD_Time, OD_Time, OD_Daily, OD_Weekly)  # (batch, time_lag, d_model)
        OD_context = self.pos_fnn(OD_context)  # (batch, time_lag, d_model)
        # OD_context = F.relu(self.linear1(OD_context))
        # OD_Hidden = OD_context.reshape(OD_context.size()[0], -1, self.station_num, self.sample_num)

        OD_context = F.relu(self.linear1(OD_context))
        OD_output = OD_context.reshape(OD_context.size()[0], -1, self.station_num, self.sample_num)
        OD_output += residual

        return OD_output
