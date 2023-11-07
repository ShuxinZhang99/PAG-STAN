import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.Encoder import Encoder, EncoderLayer, EncoderStack, ConvLayer
from model.Attention import FullAttention, ProbAttention, AttentionLayer


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    # 裁剪多出来的padding
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 factor=5, time_lag=12, d_model=512, n_heads=8, d_ff=512, embed=144,
                 attn='prob', output_attention=False,
                 ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        ))
        # 输出的size为（batch_size, output_channels, seq_len + padding）
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()

        self.attn = attn
        self.output_attention = output_attention

        Attn = ProbAttention if self.attn == 'prob' else FullAttention
        # Encoder
        self.encoder = EncoderLayer(
                    AttentionLayer(Attn(factor, attention_dropout=dropout, output_attention=self.output_attention),
                                   d_model, n_heads, time_lag, embed),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    # activation=activation
                )

        self.init_weights()

    def init_weights(self):
        # 参数初始化
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # param x : size of (Batch_size, num_stations, seq_len)
        x = self.encoder(x)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class t_former(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs: int 输入通道数
        :param num_channels: list 每层的hidden_channel数， 例如[10, 10, 10]表示有3个隐层，每层hidden_channel = 10
        :param kernel_size: 卷积核尺寸
        :param dropout:  drop_out比率
        """
        super(t_former, self).__init__()

        # Encoder
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数： 1， 2， 4， 8....
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channels, seq_len)
        :return: size of (Batch, output_channels, seq_len)
        """
        inflow = self.network(x.permute(0, 2, 1))
        output = inflow.permute(0, 2, 1)
        return output