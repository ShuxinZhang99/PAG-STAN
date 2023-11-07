import torch
import torch.nn as nn
from torch.autograd import Variable
from model.My_Model.GCN_layer import GraphConvolution


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, num_nodes):
        """
        :param input_size: station_num
        :param input_dim: the channel of input xt  每一行输入元素的个数（预测的时间窗）
        :param hidden_dim: the channel of state h and c
        """

        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # input (batch, station, sample_num)
        self.gcn = GraphConvolution(in_features=(self.input_dim + self.hidden_dim) * self.input_size[1],
                                    out_features=4 * hidden_dim * self.input_size[1], num_nodes=self.num_nodes, addaptadj=True)

    def forward(self, xt, adj, state):
        """
        :param xt: (batch_size, channels, height, weight)  输入时间序列本身
        :param state: include c(t-1) and h(t-1)
        :return: c_next, h_next
        """
        c, h = state

        # concatenate h and xt along channel axis
        com_input = torch.cat([xt, h], dim=1)
        com_input = com_input.reshape(com_input.size()[0], self.input_size[0], -1)
        com_outputs = self.gcn(com_input, adj)
        com_outputs = com_outputs.reshape(com_outputs.size()[0], -1, self.input_size[0], self.input_size[1])
        temp_i, temp_f, temp_o, temp_g = torch.split(com_outputs, self.hidden_dim, dim=1)

        i = torch.sigmoid(temp_i)  # i = sigmoid(W_xi * x_t + W_hi * H_t-1 + W_ci o C_t-1 + b_i)
        f = torch.sigmoid(temp_f)
        o = torch.sigmoid(temp_o)
        g = torch.tanh(temp_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return c_next, h_next

    def init_state(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.input_size[0], self.input_size[1])).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.input_size[0], self.input_size[1])).cuda())
