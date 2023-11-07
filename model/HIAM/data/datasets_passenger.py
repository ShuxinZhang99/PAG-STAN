import torch
from torch.utils.data import Dataset
import numpy as np

"""
Parameter:
time_interval, time_lag, tg_in_one_day, forecast_day_number, is_train=True, is_val=False, val_rate=0.1, pre_len
"""


class Traffic_passenger_flow(Dataset):
    def __init__(self, time_interval, time_lag, tg_in_one_day, forecast_day_number, inflow_data, pre_len, is_train=True,
                 is_val=False, val_rate=0.2):
        super().__init__()
        # 此部分的作用是将数据集划分为训练集、验证集、测试集。
        # 完成后X的维度为 num*276*30，30代表10个时间步*3个模式Y的维度为 num*276*1
        # X中包含上周同一时段的10个时间步、前一天同一时段的10个时间步以及临近同一时段的10个时间步
        # Y为276个车站未来1个时间步
        self.time_interval = time_interval
        self.time_lag = time_lag
        self.tg_in_one_day = tg_in_one_day
        self.forecast_day_number = forecast_day_number
        self.tg_in_one_week = self.tg_in_one_day * self.forecast_day_number
        self.inflow_data = np.loadtxt(inflow_data, delimiter=",")  # (276*num), num is the total inflow numbers in the 25 workdays
        # self.Edge_index = np.loadtxt(edge_index, delimiter=",")

        self.max_inflow = np.max(self.inflow_data)
        self.min_inflow = np.min(self.inflow_data)
        self.is_train = is_train
        self.is_val = is_val
        self.val_rate = val_rate
        self.pre_len = pre_len

        # Normalization
        self.inflow_data_norm = np.zeros((self.inflow_data.shape[0], self.inflow_data.shape[1]))
        for i in range(len(self.inflow_data)):
            for j in range(len(self.inflow_data[0])):
                self.inflow_data_norm[i, j] = round(
                    (self.inflow_data[i, j] - self.min_inflow) / (self.max_inflow - self.min_inflow), 5)
        if self.is_train:
            self.start_index = self.tg_in_one_week + self.time_lag
            self.end_index = len(self.inflow_data[0]) - self.tg_in_one_day * self.forecast_day_number - self.pre_len
        else:
            self.start_index = len(self.inflow_data[0]) - self.tg_in_one_day * self.forecast_day_number
            self.end_index = len(self.inflow_data[0]) - self.pre_len

        self.X = [[] for index in range(self.start_index, self.end_index)]
        self.Y = []
        self.Y_original = []
        # print(self.start_index, self.end_index)
        for index in range(self.start_index, self.end_index):
            temp1 = self.inflow_data_norm[:, index - self.tg_in_one_week: index - self.tg_in_one_week + 1]  # 上周同一时段
            # temp1 = self.inflow_data_norm[:, index - self.tg_in_one_week: index - self.tg_in_one_week + 1]
            # temp2 = self.inflow_data_norm[:, index - self.tg_in_one_day: index - self.tg_in_one_day + 1]
            temp2 = self.inflow_data_norm[:, index - self.tg_in_one_day: index - self.tg_in_one_day + 1]  # 昨天同一时段
            temp3 = self.inflow_data_norm[:, index - self.time_lag: index]  # 邻近几个时间段的进站量
            temp = np.concatenate((temp1, temp2, temp3), axis=1).tolist()
            self.X[index - self.start_index] = temp
            self.Y.append(self.inflow_data_norm[:, index:index + self.pre_len])
        self.X, self.Y = torch.from_numpy(np.array(self.X)), torch.from_numpy(np.array(self.Y))  # (num, 276, time_lag)
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
                    self.inflow_data[:, index:index + self.pre_len])  # the predicted inflow before normalization
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

# if __name__ == '__main__':
# 	# the following is an example
#
# 	# train
# 	inflow_data = Traffic_inflow(time_interval=15, time_lag=5, tg_in_one_day=72, forecast_day_number=5,
# 								inflow_data="./enter_data/in_15min.csv", pre_len=2, is_train=True, is_val=False, val_rate=0.1)
# 	train_inflow_X, train_inflow_Y = inflow_data.__getitem__(1)
# 	print("train_inflow_X.shape:", train_inflow_X.shape, "train_inflow_Y.shape:", train_inflow_Y.shape)
#
# 	# val
# 	inflow_data = Traffic_inflow(time_interval=15, time_lag=5, tg_in_one_day=72, forecast_day_number=5,
# 								inflow_data="./enter_data/in_15min.csv", pre_len=2, is_train=True, is_val=True, val_rate=0.1)
# 	train_inflow_X, train_inflow_Y = inflow_data.__getitem__(1)
# 	print("train_inflow_X.shape:", train_inflow_X.shape, "train_inflow_Y.shape:", train_inflow_Y.shape)
#
# 	# test
# 	inflow_data = Traffic_inflow(time_interval=15, time_lag=5, tg_in_one_day=72, forecast_day_number=5,
# 								inflow_data="./enter_data/in_15min.csv", pre_len=2, is_train=False, is_val=False, val_rate=0)
# 	train_inflow_X, train_inflow_Y, train_inflow_Y_original = inflow_data.__getitem__(1)
# 	print("train_inflow_X.shape:", train_inflow_X.shape, "train_inflow_Y.shape:", train_inflow_Y.shape,"train_inflow_Y_original.shape:", train_inflow_Y_original.shape)
