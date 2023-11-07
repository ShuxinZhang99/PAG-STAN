import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

# Transformer Parameters
time_steps = 12
d_model = 512  # Embedding size
Embed = d_model // time_steps
d_ff = 256  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of encoder of decoder layer
n_heads = 8  # number of multi-head attention
station = 15
sample_num = 15


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=61):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [seq_len, batch_size, d_model]
        :return:
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q, d_model = seq_q.size()
    batch_size, len_k, d_model = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0)  # [batch_size, len_k, d_model]
    return pad_attn_mask  # .expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    """
    :param seq: [batch_size, num_stations, time_steps]
    :return: subsequence_mask: [batch_size, num_stations, time_steps]
    """
    attn_shape = [seq.size()[0], seq.size()[1], seq.size()[1]]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None):
        '''
        :param attn_mask: [batch_size, num_stations, time_steps]
        :param Q: [batch_size, n_heads, station_num, station_num, d_k]
        :param K: [batch_size, n_heads, station_num, station_num, d_k]
        :param V: [batch_size, n_heads, num_stations, num_stations, d_v]
        :return:
        '''
        # scores: [batch_size, n_heads, station, station, station]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # if self.attn_mask is not None:
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn = nn.Softmax(dim=1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, station, station, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, flow_time, flow_day, flow_week, attn_mask=None):
        '''
        :param attn_mask:
        :param flow_time: [batch_size, len_q(station_number), d_model]
        :param flow_day: [batch_size, len_k, d_model]
        :param flow_week: [batch_size, len_v, d_model]
        :return:
        '''
        batch_size = flow_time.size(0)
        residual = flow_time.view(batch_size, time_steps, d_model)
        flow_day = flow_day.view(batch_size, time_steps, d_model)
        flow_week = flow_week.view(batch_size, time_steps, d_model)
        flow_time = flow_time.view(batch_size, time_steps, d_model)
        Q = self.W_Q(flow_day).view(batch_size, time_steps, n_heads, d_k).permute(0, 2, 1, 3)
        # Q: [batch_size, n_heads, station, station, d_k]
        K = self.W_k(flow_week).view(batch_size, time_steps, n_heads, d_k).permute(0, 2, 1, 3)
        V = self.W_V(flow_time).view(batch_size, time_steps, n_heads, d_v).permute(0, 2, 1, 3)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, time_steps, n_heads * d_v)  # (batch, station, station, d_model)
        output = self.fc(context)  # [batch_size, station, station, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


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


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, flow_time):
        '''
        :param flow_week:
        :param flow_day:
        :param flow_time: [batch_size, num_station, d_model]
        :return:
        '''
        enc_outputs, attn = self.enc_self_attn(flow_time, flow_time, flow_time)
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, num_station, station , d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, flow_time):
        # flow_time = self.pos_emb(flow_time.transpose(0, 1)).transpose(0, 1)
        # flow_day = self.pos_emb(flow_day.transpose(0, 1)).transpose(0, 1)
        # flow_week = self.pos_emb(flow_week.transpose(0, 1)).transpose(0, 1)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(flow_time)
            enc_self_attns.append(enc_self_attns)
            flow_time = enc_outputs
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs):
        '''
        :param dec_self_attn_mask:
        :param dec_inputs: [batch_size, num_stations, d_model]
        :param enc_outputs: [batch_size, num_stations, d_model]
        :return: [batch_size, num_stations, d_model]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        # dec_outputs: [batch_size, num_stations, d_model]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        # dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).cuda()
        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        # dec_self_attn_mask = torch.gt(dec_self_attn_subsequence_mask, 0).cuda()
        # dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_inputs, enc_outputs)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
            dec_inputs = dec_outputs
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, 1, bias=False).cuda()

    def forward(self, flow_time):
        enc_outputs, _ = self.encoder(flow_time)
        dec_outputs, _, _ = self.decoder(flow_time, enc_outputs)  # (batch, station * station, d_model)
        outputs = dec_outputs.reshape(dec_outputs.size()[0], d_model, time_steps)

        return outputs