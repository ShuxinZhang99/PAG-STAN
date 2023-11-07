import torch
from torch import nn
import torch.nn.functional as F
from model.My_Model.CBAM import ChannelAttention

class HIFBlock(nn.Module):
    def __init__(self, time_lag, station_num, sample_num, device):
        super(HIFBlock, self).__init__()
        self.time_lag = time_lag
        self.station_num = station_num
        self.sample_num = sample_num
        self.device = device
        self.d_model = 64
        self.d_features = 9
        self.channel_attn = ChannelAttention(in_channels=self.d_model)
        self.embedding = nn.Conv2d(in_channels=1, out_channels=self.d_model, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=5, padding=2)
        self.linear1 = nn.Linear(in_features=2048, out_features=self.station_num*self.sample_num)
        self.linear = nn.Linear(in_features=self.d_model * self.d_features, out_features=2048)

    def forward(self, external):
        external = external.to(self.device)
        external_time = external[:, :, 2: self.time_lag + 2]  # (batch, d_features, ts)
        # external_com = external_time[:, 1:2, :]
        external_soc = external_time  # [:, 0:1, :]
        # external_date = external_time[:, 2:, :]

        # external_com = self.embedding(external_com.unsqueeze(1))  # batch, d_model, 1, ts
        external_soc = self.embedding(external_soc.unsqueeze(1))
        # external_date = self.embedding(external_date.unsqueeze(1))

        # external_com_attn = self.channel_attn(external_com)
        external_soc_attn = self.channel_attn(external_soc)
        # external_date_attn = self.channel_attn(external_date)

        # external_com = external_com_attn * external_com
        external_soc = external_soc_attn * external_soc
        # external_date = external_date_attn * external_date

        # external_com3 = self.conv3(external_com)
        # external_com5 = self.conv5(external_com)
        external_soc3 = self.conv3(external_soc)
        external_soc5 = self.conv5(external_soc)
        # external_date3 = self.conv3(external_date)
        # external_date5 = self.conv5(external_date)

        # external_com = external_com3 + external_com5  # (batch, d_model, 1, ts)
        external_soc = external_soc3 + external_soc5
        # external_date = external_date3 + external_date5

        # external = torch.cat([external_com, external_soc], dim=2)  # (batch, d_model, 11, ts)
        external = external_soc.reshape(external_soc.size()[0], external_soc.size()[3], -1)
        external = F.relu(self.linear(external))
        external_output = self.linear1(external)

        output = external_output.reshape(external_output.size()[0], external_output.size()[1], self.station_num, self.sample_num)

        return output


