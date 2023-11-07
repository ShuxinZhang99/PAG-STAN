import numpy as np
import os, time, torch

import pandas as pd
import seaborn as sns
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import GetLaplacian
from model.ST_ED_RMGC.data.get_dataloader import get_inflow_dataloader
from model.ST_ED_RMGC.mian_model import Model
import matplotlib.pyplot as plt
from utils.metrics import Metrics, Metrics_1d

device = torch.device("cuda:0")  # if torch.cuda.is_available() else "cpu")
print(device)

epoch_num = 1000
lr = 0.0001
time_interval = 15
time_lag = 12
tg_in_one_day = 68
forecast_day_number = 7
pre_len = 1
batch_size = 64
station_num = 62

inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow = \
    get_inflow_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                          forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)


# get normalized adj
O_adjacency = np.loadtxt('D:/ZSX/Paper3/model/ST_ED_RMGC/data/O_Neighbor_Graph.csv', delimiter=",")
D_adjacency = np.loadtxt('D:/ZSX/Paper3/model/ST_ED_RMGC/data/D_Neighbor_Graph.csv', delimiter=",")
# ODBased_matrix = np.loadtxt('D:/ZSX/Paper3/model/ST_ED_RMGC/data/pattern.csv', delimiter=",")
O_adjacency = torch.tensor(GetLaplacian(O_adjacency).get_normalized_adj(station_num * station_num)).type(torch.float32).to(device)
D_adjacency = torch.tensor(GetLaplacian(D_adjacency).get_normalized_adj(station_num * station_num)).type(torch.float32).to(device)
# MultiGraph = torch.cat((O_adjacency, D_adjacency), dim=0)
# MultiGraph = torch.tensor(MultiGraph).type(torch.float32).to(device)

global_start_time = time.time()
writer = SummaryWriter()

model = Model(time_lag, pre_len, station_num, device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss().to(device)

path = 'D:/ZSX/Paper3/model/ST_ED_RMGC/save_model/ST-ED-RMGC-15min_2023_02_14_16_42_35/model_dict_checkpoint_130_0.00007955.pth'
checkpoint = torch.load(path)
# model_dict = model.state_dict()
# model_dict.update(checkpoint)
# pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
model.load_state_dict(checkpoint, strict=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# test
result = []
result_original = []

# if not os.path.exists('result/prediction'):
#     os.makedirs('result/prediction/')
# if not os.path.exists('result/original'):
#     os.makedirs('result/original/')
with torch.no_grad():
    model.eval()
    test_loss = 0
    for inflow_te in enumerate(inflow_data_loader_test):  # , enumerate(weibo_data_loader_test),enumerate(confirmed_data_loader_test)):
        i_batch, (test_inflow_X, test_inflow_Y, test_inflow_Y_original) = inflow_te
        # i_batch, (test_weibo_X, test_weibo_Y, test_weibo_Y_original) = weibo_te
        # i_batch, (test_confirmed_X, test_confirmed_Y, test_confirmed_Y_original) = confirmed_te
        test_inflow_X, test_inflow_Y = test_inflow_X.type(torch.float32).to(device), test_inflow_Y.type(
            torch.float32).to(device)
        # test_weibo_X, test_weibo_Y = test_weibo_X.type(torch.float32).to(device), test_weibo_Y.type(torch.float32).to(
        #     device)
        # test_confirmed_X, test_confirmed_Y = test_confirmed_X.type(torch.float32).to(device), test_confirmed_Y.type(
        #     torch.float32).to(
        #     device)

        # target = model(test_inflow_X, test_outflow_X, adjacency)
        target = model(test_inflow_X, O_adjacency, D_adjacency)  #, test_weibo_X, test_confirmed_X)  # Graph: edge_index)
        loss = mse(input=test_inflow_Y, target=target)
        test_loss += loss.item()

        # evaluate on original scale
        # 获取result (batch, 276, pre_len)
        clone_prediction = (target.cpu().detach().numpy().copy() * (max_inflow - min_inflow)) + min_inflow  # clone(): Copy the tensor and allocate the new memory
        # print(clone_prediction.shape)
        for i in range(clone_prediction.shape[0]):
            result.append(clone_prediction[i])  # （64， 1， 41， 41）

        # 获取result_original
        test_inflow_Y_original = test_inflow_Y_original.cpu().detach().numpy()
        # print(test_inflow_Y_original.shape)  # (batch_size, pre_len, station, station)
        for i in range(test_inflow_Y_original.shape[0]):
            result_original.append(test_inflow_Y_original[i])

    print(np.array(result).shape, np.array(result_original).shape)
    # 取整&负数取0
    result = np.array(result).astype(int)
    result[result < 0] = 0
    result_original = np.array(result_original).astype(int)
    result_original[result_original < 0] = 0

    result = np.array(result).reshape(-1, station_num, station_num)
    result_original = result_original.reshape(-1, station_num, station_num)
    print(result.shape, result_original.shape)

    # 取多个时刻进行画图
    x = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    y = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    # for i in range(result.shape[0]):
    x[0].append(result[3])
    y[0].append(result_original[3])
    x[1].append(result[12])
    y[1].append(result_original[12])
    x[2].append(result[20])
    y[2].append(result_original[20])
    x[3].append(result[29])
    y[3].append(result_original[29])
    x[4].append(result[37])
    y[4].append(result_original[37])
    x[5].append(result[46])
    y[5].append(result_original[46])
    x[6].append(result[54])
    y[6].append(result_original[54])
    x[7].append(result[63])
    y[7].append(result_original[63])
    x[8].append(result[71])
    y[8].append(result_original[71])
    x[9].append(result[80])
    y[9].append(result_original[80])
    x[10].append(result[88])
    y[10].append(result_original[88])
    x[11].append(result[97])
    y[11].append(result_original[97])
    x[12].append(result[105])
    y[12].append(result_original[105])
    x[13].append(result[114])
    y[13].append(result_original[114])

    x = np.array(x).squeeze()
    y = np.array(y).squeeze()

    RMSE_All = []
    R2_All = []
    MAE_All = []
    WMAPE_All = []
    for i in range(result.shape[0]):
        RMSE, R2, MAE, WMAPE = Metrics(result_original[i], result[i]).evaluate_performance()
        RMSE_All.append(RMSE)
        R2_All.append(R2)
        MAE_All.append(MAE)
        WMAPE_All.append(WMAPE)

    RMSE_Mean = np.mean(RMSE_All)
    R2_Mean = np.mean(R2_All)
    MAE_Mean = np.mean(MAE_All)
    WMAPE_Mean = np.mean(WMAPE_All)

    avg_test_loss = test_loss / len(inflow_data_loader_test)
    print('test Loss:', avg_test_loss)

    RMSE_y0, R2_y0, MAE_y0, WMAPE_y0 = Metrics(y[0], x[0]).evaluate_performance()
    RMSE_y1, R2_y1, MAE_y1, WMAPE_y1 = Metrics(y[1], x[1]).evaluate_performance()
    RMSE_y2, R2_y2, MAE_y2, WMAPE_y2 = Metrics(y[2], x[2]).evaluate_performance()
    RMSE_y3, R2_y3, MAE_y3, WMAPE_y3 = Metrics(y[3], x[3]).evaluate_performance()
    RMSE_y4, R2_y4, MAE_y4, WMAPE_y4 = Metrics(y[4], x[4]).evaluate_performance()
    RMSE_y5, R2_y5, MAE_y5, WMAPE_y5 = Metrics(y[5], x[5]).evaluate_performance()
    RMSE_y6, R2_y6, MAE_y6, WMAPE_y6 = Metrics(y[6], x[6]).evaluate_performance()
    RMSE_y7, R2_y7, MAE_y7, WMAPE_y7 = Metrics(y[7], x[7]).evaluate_performance()
    RMSE_y8, R2_y8, MAE_y8, WMAPE_y8 = Metrics(y[8], x[8]).evaluate_performance()
    RMSE_y9, R2_y9, MAE_y9, WMAPE_y9 = Metrics(y[9], x[9]).evaluate_performance()
    RMSE_y10, R2_y10, MAE_y10, WMAPE_y10 = Metrics(y[10], x[10]).evaluate_performance()
    RMSE_y11, R2_y11, MAE_y11, WMAPE_y11 = Metrics(y[11], x[11]).evaluate_performance()
    RMSE_y12, R2_y12, MAE_y12, WMAPE_y12 = Metrics(y[12], x[12]).evaluate_performance()
    RMSE_y13, R2_y13, MAE_y13, WMAPE_y13 = Metrics(y[13], x[13]).evaluate_performance()

ALL = [RMSE_Mean, MAE_Mean, WMAPE_Mean]

y0_ALL = [RMSE_y0, MAE_y0, WMAPE_y0]
y1_ALL = [RMSE_y1, MAE_y1, WMAPE_y1]
y2_ALL = [RMSE_y2, MAE_y2, WMAPE_y2]
y3_ALL = [RMSE_y3, MAE_y3, WMAPE_y3]
y4_ALL = [RMSE_y4, MAE_y4, WMAPE_y4]
y5_ALL = [RMSE_y5, MAE_y5, WMAPE_y5]
y6_ALL = [RMSE_y6, MAE_y6, WMAPE_y6]
y7_ALL = [RMSE_y7, MAE_y7, WMAPE_y7]
y8_ALL = [RMSE_y8, MAE_y8, WMAPE_y8]
y9_ALL = [RMSE_y9, MAE_y9, WMAPE_y9]
y10_ALL = [RMSE_y10, MAE_y10, WMAPE_y10]
y11_ALL = [RMSE_y11, MAE_y11, WMAPE_y11]
y12_ALL = [RMSE_y12, MAE_y12, WMAPE_y12]
y13_ALL = [RMSE_y13, MAE_y13, WMAPE_y13]

np.savetxt('result/result_rmgcn/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_ALL.txt', ALL)
np.savetxt('result/result_rmgcn/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y0_ALL.txt', y0_ALL)
np.savetxt('result/result_rmgcn/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y1_ALL.txt', y1_ALL)
np.savetxt('result/result_rmgcn/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y2_ALL.txt', y2_ALL)
np.savetxt('result/result_rmgcn/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y3_ALL.txt', y3_ALL)
np.savetxt('result/result_rmgcn/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y4_ALL.txt', y4_ALL)
# np.savetxt('result/result_cnn/X_original.txt', y)
# np.savetxt('result/result_cnn/X_prediction.txt', x)

print("ALL:", ALL)
print("y0_ALL:", y0_ALL)
print("y1_ALL:", y1_ALL)
print("y2_ALL:", y2_ALL)
print("y3_ALL:", y3_ALL)
print("y4_ALL:", y4_ALL)
print("y5_ALL:", y5_ALL)
print("y6_ALL:", y6_ALL)
print("y7_ALL:", y7_ALL)
print("y8_ALL:", y8_ALL)
print("y9_ALL:", y9_ALL)
print("y10_ALL:", y10_ALL)
print("y11_ALL:", y11_ALL)
print("y12_ALL:", y12_ALL)
print("y13_ALL:", y13_ALL)

print("end")

x = x[11]
y = y[11]

x_11 = pd.DataFrame(x, columns=None)
x_11.to_csv('result/result_rmgcn/OD_Prediction.csv', encoding='gbk')
y_11 = pd.DataFrame(y, columns=None)
y_11.to_csv('result/result_rmgcn/OD_Actual.csv', encoding='gbk')
# L1, = plt.plot(x, color="r")
# L2, = plt.plot(y, color="y")
# plt.legend([L1, L2], ["pre", "actual"], loc='best')
# plt.show()

sns.set()
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示，必须放在sns.set之后
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(x, ax=ax,vmin=0,vmax=20,cmap='YlOrRd',annot=False,linewidths=2,cbar=True)

ax.set_title('OD_Flow') #plt.title('热图'),均可设置图片标题
ax.set_ylabel('Origin')  #设置纵轴标签
ax.set_xlabel('Destination')  #设置横轴标签
plt.show()