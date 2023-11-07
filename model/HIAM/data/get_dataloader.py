from model.HIAM.data.datasets_OD import Traffic_ODFlow
from model.HIAM.data.datasets_passenger import Traffic_passenger_flow
from model.HIAM.data.datasets_rate import Traffic_ODRatio

from torch.utils.data import DataLoader
import torch
import pickle
import numpy as np
import os

with open('D:/ZSX/Paper3/model/HIAM/data/IncompressOD_60min.pickle', 'rb') as file:
    IncompressOD = pickle.load(file)
    # IncompressOD = np.array(IncompressOD)

with open('D:/ZSX/Paper3/model/HIAM/data/UnfinODRatio_60min.pickle', 'rb') as file:
    UnfinODRatio = pickle.load(file)
    # UnfinODRatio = np.array(UnfinODRatio)

with open('D:/ZSX/Paper3/model/HIAM/data/CompressDO_60min.pickle', 'rb') as file:
    DO = pickle.load(file)
    # DO = np.array(DO)

with open('D:/ZSX/Paper3/model/HIAM/data/FullyOD_60min.pickle', 'rb') as file:
    OD = pickle.load(file)
    # OD = np.array(OD)

inflow_data = 'D:/ZSX/Paper3/model/HIAM/data/20200106-20200430-Unfin60min.csv'


def get_inflow_dataloader(time_interval=60, time_lag=10, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                          batch_size=64):
    # train inflow data loader
    print("train inflow")
    inflow_train = Traffic_passenger_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                  forecast_day_number=forecast_day_number,
                                  pre_len=pre_len, inflow_data=inflow_data, is_train=True, is_val=False, val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    inflow_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val inflow")
    inflow_val = Traffic_passenger_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                forecast_day_number=forecast_day_number,
                                pre_len=pre_len, inflow_data=inflow_data, is_train=True, is_val=True, val_rate=0.2)
    inflow_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test inflow")
    inflow_test = Traffic_passenger_flow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                 forecast_day_number=forecast_day_number,
                                 pre_len=pre_len, inflow_data=inflow_data, is_train=False, is_val=False, val_rate=0)
    inflow_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow


def get_OD_dataloader(time_interval=60, time_lag=10, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                      batch_size=64):
    # train inflow data loader
    print("train InODFlow")
    inflow_train = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                  forecast_day_number=forecast_day_number,
                                  pre_len=pre_len, inflow_data=IncompressOD, is_train=True, is_val=False, val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    inflow_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val InODFlow")
    inflow_val = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                forecast_day_number=forecast_day_number,
                                pre_len=pre_len, inflow_data=IncompressOD, is_train=True, is_val=True, val_rate=0.2)
    inflow_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test InODFlow")
    inflow_test = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                 forecast_day_number=forecast_day_number,
                                 pre_len=pre_len, inflow_data=IncompressOD, is_train=False, is_val=False, val_rate=0)
    inflow_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow


def get_ComOD_dataloader(time_interval=60, time_lag=10, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                         batch_size=64):
    # train inflow data loader
    print("train ComOD")
    inflow_train = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                  forecast_day_number=forecast_day_number,
                                  pre_len=pre_len, inflow_data=OD, is_train=True, is_val=False, val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    inflow_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val ComOD")
    inflow_val = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                forecast_day_number=forecast_day_number,
                                pre_len=pre_len, inflow_data=OD, is_train=True, is_val=True, val_rate=0.2)
    inflow_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test ComOD")
    inflow_test = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                 forecast_day_number=forecast_day_number,
                                 pre_len=pre_len, inflow_data=OD, is_train=False, is_val=False, val_rate=0)
    inflow_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow


def get_ODRatio_dataloader(time_interval=60, time_lag=10, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                           batch_size=64):
    # train inflow data loader
    print("train Ratio")
    inflow_train = Traffic_ODRatio(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                  forecast_day_number=forecast_day_number,
                                  pre_len=pre_len, inflow_data=UnfinODRatio, is_train=True, is_val=False, val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    inflow_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val Ratio")
    inflow_val = Traffic_ODRatio(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                forecast_day_number=forecast_day_number,
                                pre_len=pre_len, inflow_data=UnfinODRatio, is_train=True, is_val=True, val_rate=0.2)
    inflow_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test Ratio")
    inflow_test = Traffic_ODRatio(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                 forecast_day_number=forecast_day_number,
                                 pre_len=pre_len, inflow_data=UnfinODRatio, is_train=False, is_val=False, val_rate=0)
    inflow_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow


def get_DO_dataloader(time_interval=60, time_lag=10, tg_in_one_day=17, forecast_day_number=7, pre_len=1,
                      batch_size=64):
    # train inflow data loader
    print("train DO")
    inflow_train = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                  forecast_day_number=forecast_day_number,
                                  pre_len=pre_len, inflow_data=DO, is_train=True, is_val=False, val_rate=0.2)
    max_inflow, min_inflow = inflow_train.get_max_min_inflow()
    inflow_data_loader_train = DataLoader(inflow_train, batch_size=batch_size, shuffle=False)

    # validation inflow data loader
    print("val DO")
    inflow_val = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                forecast_day_number=forecast_day_number,
                                pre_len=pre_len, inflow_data=DO, is_train=True, is_val=True, val_rate=0.2)
    inflow_data_loader_val = DataLoader(inflow_val, batch_size=batch_size, shuffle=False)

    # test inflow data loader
    print("test DO")
    inflow_test = Traffic_ODFlow(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                 forecast_day_number=forecast_day_number,
                                 pre_len=pre_len, inflow_data=DO, is_train=False, is_val=False, val_rate=0)
    inflow_data_loader_test = DataLoader(inflow_test, batch_size=batch_size, shuffle=False)

    return inflow_data_loader_train, inflow_data_loader_val, inflow_data_loader_test, max_inflow, min_inflow
