import numpy as np
import os, time, torch
from torch.utils.tensorboard import SummaryWriter
from model.ST_ED_RMGC.mian_model import Model
from utils.earlystopping import EarlyStopping
from model.ST_ED_RMGC.data.get_dataloader import get_inflow_dataloader
from utils.utils import GetLaplacian


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(device)
epoch_num = 1000
lr = 0.00005
time_interval = 15
time_lag = 12
tg_in_one_day = 68
forecast_day_number = 7
pre_len = 1
batch_size = 48
station_num = 62
model_type = "ST-ED-RMGC-15min"
TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
save_dir = './save_model/' + model_type + '_' + TIMESTAMP
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow = \
    get_inflow_dataloader(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day, \
                          forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)

# # get multi_graph
O_adjacency = np.loadtxt('D:/ZSX/Paper3/model/ST_ED_RMGC/data/O_Neighbor_Graph.csv', delimiter=",")
D_adjacency = np.loadtxt('D:/ZSX/Paper3/model/ST_ED_RMGC/data/D_Neighbor_Graph.csv', delimiter=",")
# ODBased_matrix = np.loadtxt('D:/ZSX/Paper3/model/ST_ED_RMGC/data/pattern.csv', delimiter=",")
O_adjacency = torch.tensor(GetLaplacian(O_adjacency).get_normalized_adj(station_num * station_num)).type(torch.float32).to(device)
D_adjacency = torch.tensor(GetLaplacian(D_adjacency).get_normalized_adj(station_num * station_num)).type(torch.float32).to(device)
# OD_pattern = torch.tensor(GetLaplacian(ODBased_matrix).get_normalized_adj(station_num * station_num)).type(torch.float32).to(device)
# # eigenmaps = np.loadtxt('./data/eigenmaps_matrix.csv', delimiter=",")
# adjacency = np.array(adjacency)  # 对称归一化矩阵
# similarity = np.array(similarity)
# ODBased_matrix = np.array(ODBased_matrix)
# # eigenmaps = np.array(eigenmaps)
# # eigenmaps = torch.tensor(eigenmaps).type(torch.float32).to(device)
#MultiGraph = torch.cat((O_adjacency, D_adjacency), dim=0)
# MultiGraph = torch.tensor(MultiGraph).type(torch.float32).to(device)

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
    for inflow_tr in enumerate(inflow_data_loader_train):  # ,enumerate(weibo_data_loader_train),enumerate(confirmed_data_loader_train)):
        i_batch, (train_inflow_X, train_inflow_Y) = inflow_tr
        #i_batch, (train_weibo_X, train_weibo_Y) = weibo_tr
        #i_batch, (train_confirmed_X, train_confirmed_Y) = confirmed_tr
        # i_batch, (train_outflow_X, train_outflow_Y) = outflow_tr
        train_inflow_X, train_inflow_Y = train_inflow_X.type(torch.float32).to(device), train_inflow_Y.type(
            torch.float32).to(device)
        # train_weibo_X, train_weibo_Y = train_weibo_X.type(torch.float32).to(device), train_weibo_Y.type(
        #     torch.float32).to(device)
        # train_confirmed_X, train_confirmed_Y = train_confirmed_X.type(torch.float32).to(device), train_confirmed_Y.type(
        #     torch.float32).to(device)
        # target = model(train_inflow_X, train_outflow_X, adjacency)
        target = model(train_inflow_X, O_adjacency, D_adjacency)  # train_weibo_X, train_confirmed_X, adjacency)  #, MultiGraph, eigenmaps)  # graph: edge_index
        # loss_O = []
        # loss_D = []
        # for i in range(train_inflow_Y.size()[0]):
        #     train_inflow_Y = train_inflow_Y.squeeze()
        #     target = target.squeeze()
        #     loss_o = mse(input=torch.sum(train_inflow_Y[i],dim=0), target=torch.sum(target[i], dim=0))
        #     loss_d = mse(input=torch.sum(train_inflow_Y[i],dim=1), target= torch.sum(target[i], dim=1))
        #     loss_O.append(loss_o)
        #     loss_D.append(loss_d)
        # loss_O = sum(loss_O) / len(loss_O)
        # loss_D = sum(loss_D) / len(loss_D)
        loss = mse(input=train_inflow_Y, target=target)  # + loss_O + loss_D
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        # model validation
        model.eval()
        val_loss = 0
        for inflow_val in enumerate(inflow_data_loader_val):  # ,enumerate(weibo_data_loader_val),enumerate(confirmed_data_loader_val)):
            i_batch, (val_inflow_X, val_inflow_Y) = inflow_val
            # i_batch, (val_weibo_X, val_weibo_Y) = weibo_val
            # i_batch, (val_confirmed_X, val_confirmed_Y) = confirmed_val
            val_inflow_X, val_inflow_Y = val_inflow_X.type(torch.float32).to(device), val_inflow_Y.type(
                torch.float32).to(device)
            # val_weibo_X, val_weibo_Y = val_weibo_X.type(torch.float32).to(device), val_weibo_Y.type(torch.float32).to(
            #     device)
            # val_confirmed_X, val_confirmed_Y = val_confirmed_X.type(torch.float32).to(device), val_confirmed_Y.type(
            #     torch.float32).to(
            #     device)
            # target = model(val_inflow_X, val_outflow_X, adjacency)
            target = model(val_inflow_X, O_adjacency, D_adjacency) # , val_weibo_X, val_confirmed_Y, adjacency)  #, MultiGraph, eigenmaps)  # graph: edge_index
            # loss_O = []
            # loss_D = []
            # for i in range(val_inflow_Y.size()[0]):
            #     val_inflow_Y = val_inflow_Y.squeeze()
            #     target = target.squeeze()
            #     loss_o = mse(input=torch.sum(val_inflow_Y[i], dim=0), target=torch.sum(target[i], dim=0))
            #     loss_d = mse(input=torch.sum(val_inflow_Y[i], dim=1), target=torch.sum(target[i], dim=1))
            #     loss_O.append(loss_o)
            #     loss_D.append(loss_d)
            # loss_O = sum(loss_O) / len(loss_O)
            # loss_D = sum(loss_D) / len(loss_D)
            loss = mse(input=val_inflow_Y, target=target)  #  + loss_O + loss_D
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
