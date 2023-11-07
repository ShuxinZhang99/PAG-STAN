import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os


class Traffic_ODCoeff(Dataset):
    def __init__(self, time_interval, time_lag, tg_in_one_day, forecast_day_number, inflow_data, pre_len, is_train=True,
                 is_val=False, val_rate=0.2):
        super().__init__()
        # 此部分的作用是将数据集划分为训练集、验证集、测试集。
        # 完成后X的维度为 num*276*30，30代表10个时间步*3个模式Y的维度为 num*276*1
        # X中包含上周同一时段的10个时间步、前一天同一时段的10个时间步以及临近同一时段的10个时间步
        # Y为276个车站未来1个时间步
        self.time_interval = time_interval  # 时间段长度
        self.time_delay = 60 // time_interval
        self.time_lag = time_lag  # 步长
        self.tg_in_one_day = tg_in_one_day  # 一天内包含的步长
        self.forecast_day_number = forecast_day_number
        self.tg_in_one_week = self.tg_in_one_day * 7
        self.inflow_data = inflow_data  # (time_lag*day_num) * station * station,
        # self.Edge_index = np.loadtxt(edge_index, delimiter=",")

        self.max_inflow = np.max(self.inflow_data)
        self.min_inflow = np.min(self.inflow_data)
        self.is_train = is_train
        self.is_val = is_val
        self.val_rate = val_rate
        self.pre_len = pre_len

        # Normalization
        self.inflow_data_norm = inflow_data
        self.inflow_data_norm = np.zeros(
            (self.inflow_data.shape[0], self.inflow_data.shape[1], self.inflow_data.shape[2]))
        for i in range(self.inflow_data.shape[0]):
            for j in range(self.inflow_data.shape[1]):
                for k in range(self.inflow_data.shape[2]):
                    self.inflow_data_norm[i, j, k] = round(
                        (self.inflow_data[i, j, k] - self.min_inflow) / 10000, 5)
        if self.is_train:
            self.start_index = self.tg_in_one_week + self.time_lag
            self.end_index = self.inflow_data.shape[0] - self.tg_in_one_day * self.forecast_day_number - self.pre_len
        else:
            self.start_index = self.inflow_data.shape[0] - self.tg_in_one_day * self.forecast_day_number
            self.end_index = self.inflow_data.shape[0] - self.pre_len

        self.X = [[] for index in range(self.start_index, self.end_index)]
        self.Y = []
        self.Y_original = []
        # print(self.start_index, self.end_index)
        for index in range(self.start_index, self.end_index):
            temp1 = self.inflow_data_norm[index - self.tg_in_one_week - self.time_lag: index - self.tg_in_one_week, :, :]  # 上周预测时段
            temp2 = self.inflow_data_norm[index - self.tg_in_one_day - self.time_lag: index - self.tg_in_one_day, :, :]  # 昨天预测时段
            # 上周同一时间段
            # temp3 = self.inflow_data_norm[:, index - self.tg_in_one_day: index - self.tg_in_one_day + 1]
            # 昨天同一时段
            # temp2 = self.inflow_data_norm[:, index - self.tg_in_one_day - self.time_lag: index - self.tg_in_one_day]
            temp3 = self.inflow_data_norm[index - self.time_lag - self.time_delay: index - self.time_delay, :, :]  # 邻近几个时间段的进站量
            temp = np.concatenate((temp1, temp2, temp3), axis=0).tolist()  # (time_lag + 2, station, station)
            self.X[index - self.start_index] = temp
            self.Y.append(self.inflow_data_norm[index:index + self.pre_len, :, :])
        self.X, self.Y = torch.from_numpy(np.array(self.X)), torch.from_numpy(np.array(self.Y))  # (num, 276, time_lag)
        # print("X.shape", self.X.shape, "Y.shape", self.Y.shape)
        # self.edge = torch.tensor(self.Edge_index).type(torch.long)
        # self.edge = self.edge.t()

        # if val is not zero
        if self.val_rate * len(self.X) != 0:
            val_len = int(self.val_rate * len(self.X))
            train_len = len(self.X) - val_len
            if self.is_val:
                self.X = self.X[-val_len:]
                self.Y = self.Y[-val_len:]
            else:
                self.X = self.X[:train_len]
                self.Y = self.Y[:train_len]
        print("X.shape", self.X.shape, "Y.shape", self.Y.shape)

        if not self.is_train:
            for index in range(self.start_index, self.end_index):
                self.Y_original.append(
                    self.inflow_data[index:index + self.pre_len, :, :])  # the predicted inflow before normalization
            self.Y_original = torch.from_numpy(np.array(self.Y_original))

    def get_max_min_inflow(self):
        return self.max_inflow, self.min_inflow

    def __getitem__(self, item):
        if self.is_train:
            return self.X[item], self.Y[item]
        else:
            return self.X[item], self.Y[item], self.Y_original[item]

    def __len__(self):
        return len(self.X)

