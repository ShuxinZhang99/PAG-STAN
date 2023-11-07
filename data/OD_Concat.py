from data.datasets import Traffic_inflow
from torch.utils.data import DataLoader
import torch
import pickle
import numpy as np
import os

datalist = os.listdir('D:/ZSX/Paper3/data/COVID-19/15MIN')
print(datalist)
datalist.sort(key=lambda x: int(x[0:8]))  # 排序
# 初始化n个空列表存放n天的数据
for i in range(0, len(datalist)):
    globals()['ODFlow_' + str(i)] = []

for i in range(0, len(datalist)):
    file = np.loadtxt('D:/ZSX/Paper3/data/COVID-19/15MIN/' + datalist[i], dtype=str, encoding='gbk', delimiter=',')
    for line in file:
        # line = line.strip().split(',')
        line = [int(float(x)) for x in line]
        # print(line)
        globals()['ODFlow_' + str(i)].append(line)
    print("已导入第" + str(i) + "个OD数据" + "  " + datalist[i])

cal = 0
for i in range(len(datalist)):
    globals()['ODFlow_' + str(i)] = np.array(globals()['ODFlow_' + str(i)], np.float32)
    globals()['ODFlow_' + str(i)] = np.expand_dims(globals()['ODFlow_' + str(i)], 0)
    if cal == 0:
        ODFlow = globals()['ODFlow_' + str(i)]
    else:
        ODFlow = np.concatenate((ODFlow, globals()['ODFlow_' + str(i)]), axis=0)
    cal += 1

OD_Files = open('./OD_15min.pickle', 'wb')
pickle.dump(ODFlow, OD_Files)
OD_Files.close()
