import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  stride=1,
                                  padding_mode='circular')

        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x = self.TCN(x)  # (batch_size, d_model, seq_len)
        x = self.downConv(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, time_lag=12, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        d_ff = d_ff or 4*d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, stride=1)
        self.norm1 = nn.LayerNorm(time_lag)
        self.norm2 = nn.LayerNorm(time_lag)
        self.dropout = nn.Dropout(dropout)
        # self.activation = F.relu() if activation == 'relu' else F.gelu()

    def forward(self, x):
        # x [B, S, L]
        new_x, attn = self.attention(x, x, x)

        y = x = x + self.dropout(new_x)
        y = self.dropout(F.relu(self.conv1(y)))
        y = self.dropout(self.conv2(y))

        return (x+y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x):
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[2]//(2**i_len)
            x_s, attn = encoder(x[:, :, -inp_len])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns