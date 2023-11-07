import numpy as np
import os, time, torch
from torch.utils.tensorboard import SummaryWriter
from model.My_Model.main_model import Model
from utils.earlystopping import EarlyStopping
from data.get_dataloader import get_inflow_dataloader, get_OD_dataloader, get_ODRatio_dataloader, \
    get_external_dataloader
from utils.utils import GetLaplacian

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
epoch_num = 1000
lr = 0.0001
time_interval = 30
time_lag = 12
tg_in_one_day = 34
forecast_day_number = 7
pre_len = 1
batch_size = 16
station_num = 62
model_type = "FullyDate_30min"
TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
save_dir = './save_model/' + model_type + '_' + TIMESTAMP
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow = \
#     get_inflow_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
#                           forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
OD_data_loader_train, OD_data_loader_val, OD_data_loader_test, max_OD, min_OD = \
    get_OD_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                      forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
# ODRatio_data_loader_train, ODRatio_data_loader_val, ODRatio_data_loader_test, max_ODRatio, min_ODRatio = \
#     get_ODRatio_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
#                            forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)
External_loader_train, External_loader_val, External_loader_test, max_External, min_External = \
    get_external_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                            forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)

# # get multi_graph
adjacency = np.loadtxt('D:/ZSX/Paper3/data/adjacency_20.csv', skiprows=0, delimiter=",")
adjacency = torch.tensor(GetLaplacian(adjacency).get_normalized_adj(station_num)).type(torch.float32).to(device)
mask = np.loadtxt('D:/ZSX/Paper3/data/mask_index.csv', delimiter=',')
mask = torch.tensor(mask).bool().to(device)

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
    # for inflow_tr, OD_tr, ODRatio_tr, External_tr in zip(enumerate(inflow_data_loader_train),
    #                                                      enumerate(OD_data_loader_train),
    #                                                      enumerate(ODRatio_data_loader_train),
    #                                                      enumerate(External_loader_train)):
    for OD_tr, External_tr in zip(enumerate(OD_data_loader_train), enumerate(External_loader_train)):
        # i_batch, (train_inflow_X, train_inflow_Y) = inflow_tr
        i_batch, (train_OD_X, train_OD_Y) = OD_tr
        # i_batch, (train_ODRatio_X, train_ODRatio_Y) = ODRatio_tr
        i_batch, (train_External_X, train_External_Y) = External_tr
        # train_inflow_X, train_inflow_Y = train_inflow_X.type(torch.float32).to(device), train_inflow_Y.type(
        #     torch.float32).to(device)
        train_OD_X, train_OD_Y = train_OD_X.type(torch.float32).to(device), train_OD_Y.type(torch.float32).to(device)
        # train_ODRatio_X, train_ODRatio_Y = train_ODRatio_X.type(torch.float32).to(device), train_ODRatio_Y.type(
        #     torch.float32).to(device)
        train_External_X, train_External_Y = train_External_X.type(torch.float32).to(device), train_External_Y.type(
            torch.float32).to(device)
        # target = model(train_inflow_X, train_outflow_X, adjacency)
        # target = model(train_OD_X)
        target = model(train_OD_X, adjacency, train_External_X)  # ,train_inflow_X, train_ODRatio_X,  train_External_X)
        loss_O = []
        loss_D = []
        for i in range(train_OD_Y.size()[0]):
            train_OD_Y = train_OD_Y.squeeze()
            target = target.squeeze()
            # print(train_OD_Y[i].shape)
            loss_o = mse(input=torch.sum(train_OD_Y[i], dim=0), target=torch.sum(target[i], dim=0))
            loss_d = mse(input=torch.sum(train_OD_Y[i], dim=1), target=torch.sum(target[i], dim=1))
            loss_O.append(loss_o)
            loss_D.append(loss_d)
        loss_O = sum(loss_O) / len(loss_O)
        loss_D = sum(loss_D) / len(loss_D)
        # target = torch.masked_fill(input=target, mask=~mask, value=0).to(device)
        # train_OD_Y = torch.masked_fill(input=train_OD_Y, mask=~mask, value=0).to(device)
        loss = mse(input=train_OD_Y, target=target) + loss_O + loss_D
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        # model validation
        model.eval()
        val_loss = 0
        # for inflow_val, OD_val, ODRatio_val, External_val in zip(enumerate(inflow_data_loader_val),
        #                                                          enumerate(OD_data_loader_val),
        #                                                          enumerate(ODRatio_data_loader_val),
        #                                                          enumerate(External_loader_val)):
        for OD_val, External_val in zip(enumerate(OD_data_loader_val), enumerate(External_loader_val)):
            # i_batch, (val_inflow_X, val_inflow_Y) = inflow_val
            i_batch, (val_OD_X, val_OD_Y) = OD_val
            # i_batch, (val_ODRatio_X, val_ODRatio_Y) = ODRatio_val
            i_batch, (val_External_X, val_External_Y) = External_val

            # val_inflow_X, val_inflow_Y = val_inflow_X.type(torch.float32).to(device), val_inflow_Y.type(
            #     torch.float32).to(device)
            val_OD_X, val_OD_Y = val_OD_X.type(torch.float32).to(device), val_OD_Y.type(torch.float32).to(device)
            # val_ODRatio_X, val_ODRatio_Y = val_ODRatio_X.type(torch.float32).to(device), val_ODRatio_Y.type(
            #     torch.float32).to(device)
            val_External_X, val_External_Y = val_External_X.type(torch.float32).to(device), val_External_Y.type(
                torch.float32).to(device)
            # target = model(val_inflow_X)
            target = model(val_OD_X, adjacency, val_External_X)  # ,val_inflow_X, val_ODRatio_X, val_External_X)  # val_weibo_X, val_confirmed_Y, adjacency)
            loss_O = []
            loss_D = []
            for i in range(val_OD_Y.size()[0]):
                val_OD_Y = val_OD_Y.squeeze()
                target = target.squeeze()
                loss_o = mse(input=torch.sum(val_OD_Y[i], dim=0), target=torch.sum(target[i], dim=0))
                # loss_d = mse(input=torch.sum(val_OD_Y[i], dim=-1), target=torch.sum(target[i], dim=-1))
                loss_O.append(loss_o)
                # loss_D.append(loss_d)
            loss_O = sum(loss_O) / len(loss_O)
            # loss_D = sum(loss_D) / len(loss_D)
            # target = torch.masked_fill(input=target, mask=mask, value=0).to(device)
            # val_OD_Y = torch.masked_fill(input=val_OD_Y, mask=mask, value=0).to(device)
            loss = mse(input=val_OD_Y, target=target)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(OD_data_loader_train)
    avg_val_loss = val_loss / len(OD_data_loader_val)
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
