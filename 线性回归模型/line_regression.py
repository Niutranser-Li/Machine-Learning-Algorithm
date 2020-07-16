import os, sys
import numpy as np
from matplotlib.font_manager import *
import matplotlib.pyplot as plt

def dataset_norm(data):
    """
    特征值归一化，能够将所有特征值均归一化至同样的范围内，能够加快梯度下降算法收敛速度
    :param data: 数据值
    :return: 归一化后的结果
    """
    data_max = np.max(data)
    data_min = np.min(data)
    return (data - data_min) / (data_max - data_min)

def Split_DataSet(xData, yData, split_ratio = 0.8):
    """
    将数据集切分为训练数据及测试数据
    :param xData:特征集合
    :param yData:标签集合
    :param splir_ratio: 切分比例，默认值为0.8
    :return: 特征集合以及标签集合的训练集合校验集
    """
    split_idx = np.int(xData.shape[0] * split_ratio)
    xTrain = xData[:split_idx]
    yTrain = yData[:split_idx]
    xValid = xData[split_idx:]
    yValid = yData[split_idx:]
    return xTrain, yTrain, xValid, yValid

def Load_DataSet(file_route):
    if not os.path.exists(file_route):
        raise ValueError("error! not found the dataset route: %s" % file_route)
    file_name = os.path.basename(file_route)
    Feature_Num = len(open(file_route, "r", encoding="utf-8").readline().strip().split('\t')) - 1
    print("train set name is: %s and check feature number is: %d" % (file_name, Feature_Num))

    XArr, YArr = list(), list()
    filein = open(file_route, "r", encoding="utf-8")
    success_load = 0
    for line in filein:
        line_ext = list()
        cur_line = line.strip().split("\t")
        if len(cur_line) != Feature_Num+1: continue
        success_load += 1
        for feat_idx in range(Feature_Num):
            line_ext.append(float(cur_line[feat_idx]))
        XArr.append(line_ext)
        YArr.append(float(cur_line[-1]))
    filein.close()
    print("successful load sample line number: %d" % (success_load))
    return XArr, YArr

def delta_wb(x, y, prey):
    """
    计算w和b每次更新的幅度
    :param x: 真实的x值
    :param y: 预测的y值
    :param prey: 真实的y值
    :return: w,b 应该更新的值
    """
    delta_w = (y - prey) * x
    delta_b = y - prey
    return delta_w, delta_b

def cal_loss(y, prey):
    return (y - prey)**2 * (1/2)

def train_regression_model(epoch, alpha, x_train, y_train, x_val, y_val, model_w, model_b):
    """
    使用梯度下降的方法训练线性回归模型
    :param epoch: 模型训练的轮数
    :param alpha: 模型训练过程中，参数更新的学习率
    :param x_train: 训练集合X
    :param y_train: 训练集合Y
    :param mode_w: 特征权重集合
    :param model_b: 模型偏移量
    :return:
    """
    loss = []
    sample_acc = len(y_train)
    valid_x, valid_y = x_val, y_val
    for epoch_idx in range(epoch):
        epoch_loss = 0
        for idx in range(sample_acc):
            pred_y = np.dot(model_w, x_train[idx]) + model_b
            delta_w, delta_b = delta_wb(x_train[idx], pred_y, y_train[idx])
            model_w = model_w - alpha * delta_w
            model_b = model_b - alpha * delta_b
            epoch_loss += cal_loss(y_train[idx], pred_y)
        loss.extend(epoch_loss / sample_acc)
        val_loss = regression_model_val(valid_x, valid_y, model_w, model_b)
        val_loss = round(val_loss, 3)
        print("train line regression model [ %d / %d ]: loss = %s  average loss: %s val loss: %s" % (epoch_idx + 1, epoch, str(epoch_loss), str(epoch_loss / sample_acc), val_loss))
    return model_w, model_b, loss

def regression_model_val(x_val, y_val, w, b):
    y_pred = np.dot(w, x_val.T) + b
    num = y_pred.shape[1]
    y_pred = y_pred[0]
    sum_loss = 0
    for idx in range(num):
        temp_loss = y_pred[idx] - y_val[idx]
        sum_loss += temp_loss
    return sum_loss

def stand_regression_norm(xArr, yArr):
    """
    计算常规线性回归方法的权重: 正规函数方法
    :param xArr: 特征值矩阵
    :param yArr: 标签矩阵
    :return: 普通线性回归方法的权重矩阵
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        raise ValueError("matrix is not a -1!")
    ws = xTx.I * (xMat.T * yMat)
    return ws

def train():
    x, y = Load_DataSet("./dataset/abalone.txt")
    norm_x = dataset_norm(x)
    norm_y = dataset_norm(y)

    x_train, y_train, x_val, y_val = Split_DataSet(norm_x, norm_y)
    train_number = x_train.shape[0]
    print("train dataset shape: " + str(x_train.shape))
    print("valid dataset shape: " + str(x_val.shape))

    np.random.seed(1)
    w = np.random.random((1, 8))
    b = np.random.random((1))

    w, b, train_loss = train_regression_model(1000, 0.01, x_train, y_train, x_val, y_val, w, b)
    val_loss = regression_model_val(x_val, y_val, w, b)

train()