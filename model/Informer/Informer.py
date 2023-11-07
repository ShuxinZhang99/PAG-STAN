import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Informer.Encoder import Encoder, EncoderLayer, EncoderStack, ConvLayer
from model.Informer.Attention import FullAttention, ProbAttention, AttentionLayer
from model.TCN import TemporalConvNet


class Informer(nn.Module):
    def __init__(self, c_out, factor=5, time_lag=12,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn='prob', output_attention=False,
                 distil=True):
        super(Informer, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        Attn = ProbAttention if self.attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=self.output_attention),
                                   d_model, n_heads, time_lag),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    # activation=activation
                ) for l in range(e_layers)
            ],
            [
                TemporalConvNet(
                    num_inputs=c_out, num_channels=[24, 12]
                ) for l in range(e_layers - 1)
                # ConvLayer(
                #     c_in=c_out
                # ) for l in range(e_layers-1)
            ] if distil else None
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc):
        dec_out, attns = self.encoder(x_enc)
        # print(dec_out.size())
        # dec_out = self.projection(enc_out)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, c_out, pre_len, factor=5,
                 d_model=512, n_heads=8, e_layers=[3, 2, 1], d_ff=512,
                 dropout=0.0, attn='prob', output_attention=False,
                 distil=True, mix=True, device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = pre_len
        self.attn = attn
        self.output_attention = output_attention

        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        # activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc):
        enc_out, attns = self.encoder(x_enc)
        print(enc_out.size())
        dec_out = self.projection(enc_out)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


