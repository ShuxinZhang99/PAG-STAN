import numpy as np
import os, time, torch
from model.HIAM.utils.utils import GetLaplacian
from torch.utils.tensorboard import SummaryWriter
from model.HIAM.module.main_model import Model
from utils.earlystopping import EarlyStopping
from data.get_dataloader import get_inflow_dataloader, get_DO_dataloader, get_OD_dataloader, get_ODRatio_dataloader, \
    get_ComOD_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
epoch_num = 1000
lr = 0.00005
time_interval = 60
time_lag = 10
tg_in_one_day = 17
forecast_day_number = 7
pre_len = 1
batch_size = 24
station_num = 62
model_type = "HIAM-60min"
TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
save_dir = './save_model/' + model_type + '_' + TIMESTAMP
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

DO_data_loader_train, DO_data_loader_val, DO_data_loader_test, DO_max_inflow, DO_min_inflow = \
   get_DO_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                     forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow = \
    get_inflow_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                          forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
OD_data_loader_train, OD_data_loader_val, OD_data_loader_test, OD_max_inflow, OD_min_inflow = \
    get_OD_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                      forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
ODRatio_data_loader_train, ODRatio_data_loader_val, ODRatio_data_loader_test, Ratio_max_inflow, Ratio_min_inflow = \
    get_ODRatio_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                           forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
ComOD_data_loader_train, ComOD_data_loader_val, ComOD_data_loader_test, ComOD_max_inflow, ComOD_min_inflow = \
    get_ComOD_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                         forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)

# # get multi_graph
adjacency = np.loadtxt('./data/adjacency.csv', delimiter=",")
# similarity = np.loadtxt('./data/similarity_matrix.csv', delimiter=",")
# ODBased_matrix = np.loadtxt('./data/ODcoffe_matrix.csv', delimiter=",")
# # eigenmaps = np.loadtxt('./data/eigenmaps_matrix.csv', delimiter=",")
# adjacency = np.array(adjacency)  # 对称归一化矩阵
# similarity = np.array(similarity)
# ODBased_matrix = np.array(ODBased_matrix)
# # eigenmaps = np.array(eigenmaps)
# # eigenmaps = torch.tensor(eigenmaps).type(torch.float32).to(device)
# MultiGraph = np.stack((adjacency, similarity, ODBased_matrix), axis=0)
# MultiGraph = torch.tensor(MultiGraph).type(torch.float32).to(device)
adjacency = torch.tensor(GetLaplacian(adjacency).get_normalized_adj(station_num)).type(torch.float32).to(device)
# # ASTGNN不用该步骤

global_strat_time = time.time()
writer = SummaryWriter()

model = Model(time_lag, pre_len, station_num, device)
print(model)

model = model.to(device)
# model_dict = model.state_dict()
# for k, v in model_dict.items():
#     print(k)  # 只打印key值，不打印具体参数。
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss().to(device)

temp_time = time.time()
early_stopping = EarlyStopping(patience=100, verbose=True)

for epoch in range(0, epoch_num):
    # model train
    train_loss = 0
    model.train()
    for inflow_tr, DO_tr, OD_tr, ComOD_tr, ODRatio_tr in zip(enumerate(inflow_data_loader_train),
                                                             enumerate(DO_data_loader_train),
                                                             enumerate(OD_data_loader_train),
                                                             enumerate(ComOD_data_loader_train),
                                                             enumerate(ODRatio_data_loader_train)):
        i_batch, (train_inflow_X, train_inflow_Y) = inflow_tr
        i_batch, (train_DO_X, train_DO_Y) = DO_tr
        i_batch, (train_OD_X, train_OD_Y) = OD_tr
        i_batch, (train_ComOD_X, train_ComOD_Y) = ComOD_tr
        i_batch, (train_ODRatio_X, train_ODRatio_Y) = ODRatio_tr
        train_inflow_X, train_inflow_Y = train_inflow_X.type(torch.float32).to(device), train_inflow_Y.type(
            torch.float32).to(device)
        train_DO_X, train_DO_Y = train_DO_X.type(torch.float32).to(device), train_DO_Y.type(
            torch.float32).to(device)
        train_OD_X, train_OD_Y = train_OD_X.type(torch.float32).to(device), train_OD_Y.type(
            torch.float32).to(device)
        train_ComOD_X, train_ComOD_Y = train_ComOD_X.type(torch.float32).to(device), train_ComOD_Y.type(
            torch.float32).to(device)
        train_ODRatio_X, train_ODRatio_Y = train_ODRatio_X.type(torch.float32).to(device), train_ODRatio_Y.type(
            torch.float32).to(device)
        # target = model(train_inflow_X, train_outflow_X, adjacency)
        target = model(train_inflow_X, train_DO_X, train_OD_X, train_ODRatio_X, adjacency)
        # train_weibo_X, train_confirmed_X, adjacency)  #, MultiGraph, eigenmaps)  # graph: edge_index
        loss = mse(input=train_ComOD_Y, target=target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        # model validation
        model.eval()
        val_loss = 0
        for inflow_val, DO_val, OD_val, ComOD_val, ODRatio_val in zip(enumerate(inflow_data_loader_train),
                                                                      enumerate(DO_data_loader_train),
                                                                      enumerate(OD_data_loader_train),
                                                                      enumerate(ComOD_data_loader_train),
                                                                      enumerate(ODRatio_data_loader_train)):
            i_batch, (val_inflow_X, val_inflow_Y) = inflow_val
            i_batch, (val_DO_X, val_DO_Y) = DO_val
            i_batch, (val_OD_X, val_OD_Y) = OD_val
            i_batch, (val_ComOD_X, val_ComOD_Y) = OD_val
            i_batch, (val_ODRatio_X, val_ODRatio_Y) = ODRatio_val
            val_inflow_X, val_inflow_Y = val_inflow_X.type(torch.float32).to(device), val_inflow_Y.type(
                torch.float32).to(device)
            val_DO_X, val_DO_Y = val_DO_X.type(torch.float32).to(device), val_DO_Y.type(
                torch.float32).to(device)
            val_OD_X, val_OD_Y = val_OD_X.type(torch.float32).to(device), val_OD_Y.type(
                torch.float32).to(device)
            val_ComOD_X, val_ComOD_Y = val_ComOD_X.type(torch.float32).to(device), val_ComOD_Y.type(
                torch.float32).to(device)
            val_ODRatio_X, val_ODRatio_Y = val_ODRatio_X.type(torch.float32).to(device), val_ODRatio_Y.type(
                torch.float32).to(device)
            # target = model(train_inflow_X, train_outflow_X, adjacency)
            target = model(val_inflow_X, val_DO_X, val_OD_X, val_ODRatio_X, adjacency)
            loss = mse(input=val_ComOD_Y, target=target)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(inflow_data_loader_train)
    avg_val_loss = val_loss / len(inflow_data_loader_val)
    writer.add_scalar("loss_train", avg_train_loss, epoch)
    writer.add_scalar("loss_eval", avg_val_loss, epoch)
    print('epoch:', epoch, 'train Loss:', avg_train_loss, 'val Loss:', avg_val_loss)

    if epoch > 0:
        model_dict = model.state_dict()
        # for k, v in model_dict.items():
        #     print(k)  # 只打印key值，不打印具体参数。
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping(avg_val_loss, model_dict, model, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    # 每10个epoch打印一次训练时间
    if epoch % 10 == 0:
        print("time for 10 epoches:", round(time.time() - temp_time, 2))
        temp_time = time.time()
global_end_time = time.time() - global_strat_time
print("global end time:", global_end_time)
