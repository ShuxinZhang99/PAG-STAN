import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Informer.Encoder import Encoder, EncoderLayer, EncoderStack, ConvLayer
from model.Informer.Attention import FullAttention, ProbAttention, AttentionLayer
from model.Informer.Decoder import Decoder, DecoderLayer


class Informer(nn.Module):
    def __init__(self, factor=5, time_lag=12, station_num=62, sample_num=38, d_model=512, n_heads=8, e_layers=3, d_ff=128, d_layers=2,
                 dropout=0.0, attn='prob', output_attention=False, distil=False):
        super(Informer, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        Attn = ProbAttention if self.attn == 'prob' else FullAttention
        self.tokenEmbedding = nn.Conv1d(in_channels=station_num * sample_num, out_channels=d_model, kernel_size=3, padding=1)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=self.output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    # activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    c_in=d_model
                ) for l in range(e_layers-1)
            ] if distil else None
        )
        self.projection = nn.Linear(d_model, station_num*sample_num, bias=True)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=self.output_attention),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(time_lag)
        )

    def forward(self, x_enc, x_dec):
        x_enc = self.tokenEmbedding(x_enc)  # (batch_size, d_model, seq_len)
        enc_out, attns = self.encoder(x_enc)  # (batch_size, d_model, *seq_len) 经过蒸馏
        x_dec = self.tokenEmbedding(x_dec)  # (batch_size, d_model, seq_len)
        dec_out = self.decoder(x_dec, enc_out)
        dec_out = self.projection(dec_out.transpose(2, 1)).permute(0, 2, 1)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [:, -self.pred_len:, :]  # [B, L, D]


# class InformerStack(nn.Module):
#     def __init__(self, c_out, pre_len, factor=5,
#                  d_model=512, n_heads=8, e_layers=[3, 2, 1], d_ff=512,
#                  dropout=0.0, attn='prob', output_attention=False,
#                  distil=True, mix=True, device=torch.device('cuda:0')):
#         super(InformerStack, self).__init__()
#         self.pred_len = pre_len
#         self.attn = attn
#         self.output_attention = output_attention
#
#         Attn = ProbAttention if attn == 'prob' else FullAttention
#
#         # Encoder
#         inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
#         encoders = [
#             Encoder(
#                 [
#                     EncoderLayer(
#                         AttentionLayer(
#                             Attn(factor, attention_dropout=dropout, output_attention=output_attention),
#                             d_model, n_heads, mix=False),
#                         d_model,
#                         d_ff,
#                         dropout=dropout,
#                         # activation=activation
#                     ) for l in range(el)
#                 ],
#                 [
#                     ConvLayer(
#                         d_model
#                     ) for l in range(el - 1)
#                 ] if distil else None,
#                 norm_layer=torch.nn.LayerNorm(d_model)
#             ) for el in e_layers]
#         self.encoder = EncoderStack(encoders, inp_lens)
#
#         self.projection = nn.Linear(d_model, c_out, bias=True)
#
#     def forward(self, x_enc):
#         enc_out, attns = self.encoder(x_enc)
#         print(enc_out.size())
#         dec_out = self.projection(enc_out)
#
#         if self.output_attention:
#             return dec_out, attns
#         else:
#             return dec_out


