from data.datasets import Traffic_inflow
from data.datasets_inflow import Traffic_inbound_flow
from data.datasets_ODCoeff import Traffic_ODCoeff
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import pickle

# datalist = os.listdir('D:/ZSX/Paper3/data/19NewYear(60min)')
# print(datalist)
# datalist.sort(key=lambda x: int(x[0:8]))  # 排序
# # 初始化n个空列表存放n天的数据
# for i in range(0, len(datalist)):
#     globals()['ODFlow_' + str(i)] = []
#
# for i in range(0, len(datalist)):
#     file = np.loadtxt('D:/ZSX/Paper3/data/19NewYear(60min)/' + datalist[i], dtype=str, encoding='gbk', delimiter=',')
#     for line in file:
#         # line = line.strip().split(',')
#         line = [int(float(x)) for x in line]
#         # print(line)
#         globals()['ODFlow_' + str(i)].append(line)
#     print("已导入第" + str(i) + "个OD数据" + "  " + datalist[i])
#
# cal = 0
# for i in range(len(datalist)):
#     globals()['ODFlow_' + str(i)] = np.array(globals()['ODFlow_' + str(i)], np.float32)
#     globals()['ODFlow_' + str(i)] = np.expand_dims(globals()['ODFlow_' + str(i)], 0)
#     if cal == 0:
#         ODFlow = globals()['ODFlow_' + str(i)]
#     else:
#         ODFlow = np.concatenate((ODFlow, globals()['ODFlow_' + str(i)]), axis=0)
#     cal += 1
device = torch.device("cuda:0")  # if torch.cuda.is_available() else "cpu")

inflow_data = 'D:/ZSX/Paper3/data/20200106-20200430-30min.csv'

external_factors = 'D:/ZSX/Paper3/data/Date_Factors_30min.csv'

with open('D:/ZSX/Paper3/data/FullyCompressOD_30min.pickle', 'rb') as file:
    ODFlow = pickle.load(file)
    # ODFlow = np.array(ODFlow)

with open('D:/ZSX/Paper3/data/FullyCompressODRate_30min.pickle', 'rb') as file:
    ODRatio = pickle.load(file)
    # ODRatio = np.array(ODRatio)


def get_OD_dataloader(time_interval=60, time_lag=12, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                      batch_size=64):
    # train inflow data loader
    print("train OD")
    inflow_train = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                  forecast_day_number=forecast_day_number,
                                  pre_len=pre_len, inflow_data=ODFlow, is_train=True, is_val=False, val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    OD_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val OD")
    inflow_val = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                forecast_day_number=forecast_day_number,
                                pre_len=pre_len, inflow_data=ODFlow, is_train=True, is_val=True, val_rate=0.2)
    OD_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test OD")
    inflow_test = Traffic_inflow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                 forecast_day_number=forecast_day_number,
                                 pre_len=pre_len, inflow_data=ODFlow, is_train=False, is_val=False, val_rate=0)
    OD_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return OD_data_loader_train, OD_data_loader_val, OD_data_loader_test, max_inflow, min_inflow


def get_external_dataloader(time_interval=60, time_lag=12, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                            batch_size=64):
    # train inflow data loader
    print("train external")
    inflow_train = Traffic_inbound_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                        forecast_day_number=forecast_day_number, pre_len=pre_len,
                                        inflow_data=external_factors, is_train=True, is_val=False, val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    external_factor_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val external")
    inflow_val = Traffic_inbound_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                      forecast_day_number=forecast_day_number, pre_len=pre_len,
                                      inflow_data=external_factors, is_train=True, is_val=True, val_rate=0.2)
    external_factor_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test external")
    inflow_test = Traffic_inbound_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                       forecast_day_number=forecast_day_number, pre_len=pre_len,
                                       inflow_data=external_factors, is_train=False, is_val=False, val_rate=0)
    external_factor_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return external_factor_loader_train, external_factor_loader_val, external_factor_loader_test, max_inflow, min_inflow


def get_ODRatio_dataloader(time_interval=60, time_lag=12, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                           batch_size=64):
    # train inflow data loader
    print("train ODRatio")
    inflow_train = Traffic_ODCoeff(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                   forecast_day_number=forecast_day_number,
                                   pre_len=pre_len, inflow_data=ODRatio, is_train=True, is_val=False, val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    ODRatio_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val ODRatio")
    inflow_val = Traffic_ODCoeff(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                 forecast_day_number=forecast_day_number,
                                 pre_len=pre_len, inflow_data=ODRatio, is_train=True, is_val=True, val_rate=0.2)
    ODRatio_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test ODRatio")
    inflow_test = Traffic_ODCoeff(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                  forecast_day_number=forecast_day_number,
                                  pre_len=pre_len, inflow_data=ODRatio, is_train=False, is_val=False, val_rate=0)
    ODRatio_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return ODRatio_data_loader_train, ODRatio_data_loader_val, ODRatio_data_loader_test, max_inflow, min_inflow


def get_inflow_dataloader(time_interval=60, time_lag=12, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                          batch_size=64):
    # train inflow data loader
    print("train inflow")
    inflow_train = Traffic_inbound_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                        forecast_day_number=forecast_day_number,
                                        pre_len=pre_len, inflow_data=inflow_data, is_train=True, is_val=False,
                                        val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    inflow_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val inflow")
    inflow_val = Traffic_inbound_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                      forecast_day_number=forecast_day_number,
                                      pre_len=pre_len, inflow_data=inflow_data, is_train=True, is_val=True,
                                      val_rate=0.2)
    inflow_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test inflow")
    inflow_test = Traffic_inbound_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                       forecast_day_number=forecast_day_number,
                                       pre_len=pre_len, inflow_data=inflow_data, is_train=False, is_val=False,
                                       val_rate=0)
    inflow_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow
