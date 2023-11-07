import numpy as np
import os, time, torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import GetLaplacian

from model.ResNet.main_ResNet import Model
import matplotlib.pyplot as plt
from utils.metrics import Metrics, Metrics_1d
from data.get_dataloader import get_inflow_dataloader, get_WeiBo_dataloder, get_Confirmed_dataloder
from model.ASTGNN.Norm_Matrix import sym_norm_Adj

device = torch.device("cuda:0")  # if torch.cuda.is_available() else "cpu")
print(device)

epoch_num = 1000
lr = 0.0001
time_interval = 10
time_lag = 12
tg_in_one_day = 102
forecast_day_number = 7
pre_len = 1
batch_size = 64
station_num = 61

inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow = \
    get_inflow_dataloader(time_interval=10, time_lag=time_lag, tg_in_one_day=102,
                          forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
weibo_data_loader_train, weibo_data_loader_val, weibo_data_loader_test, max_weibo, min_weibo = \
    get_WeiBo_dataloder(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                        forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
confirmed_data_loader_train, confirmed_data_loader_val, confirmed_data_loader_test, max_confirmed, min_confirmed = \
    get_Confirmed_dataloder(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                            forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)

# get normalized adj
adjacency = np.loadtxt('./data/adjacency_20.csv', delimiter=",")
similarity = np.loadtxt('./data/similarity_matrix.csv', delimiter=",")
ODBased_matrix = np.loadtxt('./data/ODcoffe_matrix.csv', delimiter=",")
# eigenmaps = np.loadtxt('./data/eigenmaps_matrix.csv', delimiter=",")
# adjacency = sym_norm_Adj(np.array(adjacency))
adjacency = np.array(adjacency)
similarity = np.array(similarity)
ODBased_matrix = np.array(ODBased_matrix)
# eigenmaps = np.array(eigenmaps)
# eigenmaps = torch.tensor(eigenmaps).type(torch.float32).to(device)
MultiGraph = np.stack((adjacency, similarity, ODBased_matrix), axis=0)
MultiGraph = torch.tensor(MultiGraph).type(torch.float32).to(device)
adjacency = torch.tensor(GetLaplacian(adjacency).get_normalized_adj(station_num)).type(torch.float32).to(device)

global_start_time = time.time()
writer = SummaryWriter()

model = Model(time_lag, pre_len, station_num, device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss().to(device)

path = 'D:/BC_notes/python/Paper2/save_model/10min/ResNet_10min_2022_07_06_14_35_01/model_dict_checkpoint_260_0.00013107.pth'
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
    for inflow_te, weibo_te, confirmed_te in zip(enumerate(inflow_data_loader_test), enumerate(weibo_data_loader_test),
                                                 enumerate(confirmed_data_loader_test)):
        i_batch, (test_inflow_X, test_inflow_Y, test_inflow_Y_original) = inflow_te
        i_batch, (test_weibo_X, test_weibo_Y, test_weibo_Y_original) = weibo_te
        i_batch, (test_confirmed_X, test_confirmed_Y, test_confirmed_Y_original) = confirmed_te
        test_inflow_X, test_inflow_Y = test_inflow_X.type(torch.float32).to(device), test_inflow_Y.type(
            torch.float32).to(device)
        test_weibo_X, test_weibo_Y = test_weibo_X.type(torch.float32).to(device), test_weibo_Y.type(torch.float32).to(
            device)
        test_confirmed_X, test_confirmed_Y = test_confirmed_X.type(torch.float32).to(device), test_confirmed_Y.type(
            torch.float32).to(
            device)

        # target = model(test_inflow_X, test_outflow_X, adjacency)
        target = model(test_inflow_X, test_weibo_X, test_confirmed_X)  # Graph: edge_index)
        loss = mse(input=test_inflow_Y, target=target)
        test_loss += loss.item()

        # evaluate on original scale
        # 获取result (batch, 276, pre_len)
        clone_prediction = (target.cpu().detach().numpy().copy() * (max_inflow - min_inflow)) + min_inflow  # clone(): Copy the tensor and allocate the new memory
        # print(clone_prediction.shape)  # (16, 276, 1)
        for i in range(clone_prediction.shape[0]):
            result.append(clone_prediction[i])

        # 获取result_original
        test_inflow_Y_original = test_inflow_Y_original.cpu().detach().numpy()
        # print(test_inflow_Y_original.shape)  # (16, 276, 1)
        for i in range(test_inflow_Y_original.shape[0]):
            result_original.append(test_inflow_Y_original[i])

    print(np.array(result).shape, np.array(result_original).shape)
    # 取整&负数取0
    result = np.array(result).astype(int)
    result[result < 0] = 0
    result_original = np.array(result_original).astype(int)
    result_original[result_original < 0] = 0

    # 取多个车站进行画图
    x = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    y = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(result.shape[0]):
        x[0].append(result[i][7][0])
        y[0].append(result_original[i][7][0])
        x[1].append(result[i][9][0])
        y[1].append(result_original[i][9][0])
        x[2].append(result[i][13][0])
        y[2].append(result_original[i][13][0])
        x[3].append(result[i][14][0])
        y[3].append(result_original[i][14][0])
        x[4].append(result[i][20][0])
        y[4].append(result_original[i][20][0])
        x[5].append(result[i][22][0])
        y[5].append(result_original[i][22][0])
        x[6].append(result[i][27][0])
        y[6].append(result_original[i][27][0])
        x[7].append(result[i][30][0])
        y[7].append(result_original[i][30][0])
        x[8].append(result[i][34][0])
        y[8].append(result_original[i][34][0])
        x[9].append(result[i][37][0])
        y[9].append(result_original[i][37][0])
        x[10].append(result[i][10][0])
        y[10].append(result_original[i][10][0])
        x[11].append(result[i][11][0])
        y[11].append(result_original[i][11][0])
        x[12].append(result[i][12][0])
        y[12].append(result_original[i][12][0])
        x[13].append(result[i][13][0])
        y[13].append(result_original[i][13][0])

    result = np.array(result).reshape(station_num, -1)
    result_original = result_original.reshape(station_num, -1)
    print(result.shape, result_original.shape)

    RMSE, R2, MAE, WMAPE = Metrics(result_original, result).evaluate_performance()

    avg_test_loss = test_loss / len(inflow_data_loader_test)
    print('test Loss:', avg_test_loss)

    RMSE_y0, R2_y0, MAE_y0, WMAPE_y0 = Metrics_1d(y[0], x[0]).evaluate_performance()
    RMSE_y1, R2_y1, MAE_y1, WMAPE_y1 = Metrics_1d(y[1], x[1]).evaluate_performance()
    RMSE_y2, R2_y2, MAE_y2, WMAPE_y2 = Metrics_1d(y[2], x[2]).evaluate_performance()
    RMSE_y3, R2_y3, MAE_y3, WMAPE_y3 = Metrics_1d(y[3], x[3]).evaluate_performance()
    RMSE_y4, R2_y4, MAE_y4, WMAPE_y4 = Metrics_1d(y[4], x[4]).evaluate_performance()
    RMSE_y5, R2_y5, MAE_y5, WMAPE_y5 = Metrics_1d(y[5], x[5]).evaluate_performance()
    RMSE_y6, R2_y6, MAE_y6, WMAPE_y6 = Metrics_1d(y[6], x[6]).evaluate_performance()
    RMSE_y7, R2_y7, MAE_y7, WMAPE_y7 = Metrics_1d(y[7], x[7]).evaluate_performance()
    RMSE_y8, R2_y8, MAE_y8, WMAPE_y8 = Metrics_1d(y[8], x[8]).evaluate_performance()
    RMSE_y9, R2_y9, MAE_y9, WMAPE_y9 = Metrics_1d(y[9], x[9]).evaluate_performance()
    RMSE_y10, R2_y10, MAE_y10, WMAPE_y10 = Metrics_1d(y[10], x[10]).evaluate_performance()
    RMSE_y11, R2_y11, MAE_y11, WMAPE_y11 = Metrics_1d(y[11], x[11]).evaluate_performance()
    RMSE_y12, R2_y12, MAE_y12, WMAPE_y12 = Metrics_1d(y[12], x[12]).evaluate_performance()
    RMSE_y13, R2_y13, MAE_y13, WMAPE_y13 = Metrics_1d(y[13], x[13]).evaluate_performance()

ALL = [RMSE, MAE, WMAPE]
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

np.savetxt('result/result_LSTM/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_ALL.txt', ALL)
np.savetxt('result/result_LSTM/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y0_ALL.txt', y0_ALL)
np.savetxt('result/result_LSTM/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y1_ALL.txt', y1_ALL)
np.savetxt('result/result_LSTM/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y2_ALL.txt', y2_ALL)
np.savetxt('result/result_LSTM/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y3_ALL.txt', y3_ALL)
np.savetxt('result/result_LSTM/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_y4_ALL.txt', y4_ALL)
np.savetxt('result/result_LSTM/X_original.txt', y)
np.savetxt('result/result_LSTM/X_prediction.txt', x)

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
L1, = plt.plot(x, color="r")
L2, = plt.plot(y, color="y")
plt.legend([L1, L2], ["pre", "actual"], loc='best')
plt.show()
