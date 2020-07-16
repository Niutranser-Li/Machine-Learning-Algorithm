import numpy as np
import os,sys
"""
本代码完成的主要任务为使用局部加权线性回归模型(Locally Weighted Linear Regression, LWLR)完成预测鲍鱼年龄的任务
LWLR模型的好处: 熟悉普通线性回归可以让我们知道,其经常会出现欠拟合的情况, 即模型拟合的曲线与样本点的真是分布还是存在一定偏差
而解决模型欠拟合的方式有很多种, 如过采样/增加模型复杂程度等,但是一味的增加模型复杂程度则又可能会造成过拟合情况的发生
使用LWLR模型能够解决这一问题.
数据集介绍: 鲍鱼年龄数据集, 数据集中包含9列内容, 用于使模型根据鲍鱼的一些特征预测鲍鱼年龄
数据集各列的属性为: 性别/长度/直径/高度/整体重量/去壳后重量/脏器重量/壳的重量/环数(鲍鱼年龄)
"""

def LWLR_Model(standPoint, xTrain, yTrain, k=1.0):
    """
    局部加权特征回归的模型部分, 对于每一个待预测点, 都将其根据附近的每个点赋予一定的权重, 并基于最小均方差进行普通的线性回归
    (个人理解: 是一种根据真实样本点,对模型拟合的曲线进行纠正的方法, 使用周围的样本点对预测样本点位置进行纠正)
    :param standPoint: 附近的样本点
    :param xTrain: 样本特征数据, 即为样本的feature
    :param yTrain: 每个样本对应的标签, 即样本的Lable
    :param k: 关于权重矩阵(高斯核)的一个参数, 与权重的衰减速率有关(k越小, 权重衰减越快, 图像越瘦)
    :return: standPoint * ws 数据点与具有权重的系数相乘得到的预测点
    :Note: 权重计算公式: w = e ^ ( (x^(i)-x)^2 / -2 * k^2)
    其中, x为某个预测点, x^(i)为样本点, 样本点距离距离预测点距离越大, 则这个样本点贡献的回归误差值越大(权值越大), 越远则贡献的误差越小。
    预测点可以选取样本点, 其中k是带宽参数, 控制w(钟型函数, 高斯分布)的宽窄程度, 类似于高斯分布的标准差.
    算法思路: 假设预测点取样本点中的第i个样本(共有m个样本), 从头到尾遍历1到m个样本点(含有第i个), 算出每一个样本点与预测点的距离,
    也可以计算出每个样本贡献误差的权值, 可以看出w是一个有m个元素的向量, 写成对角矩阵的方式.
    """
    x_Mat = np.mat(xTrain)
    y_Mat = np.mat(yTrain).T
    # print(x_Mat.shape)
    # print(y_Mat.shape)
    # print(y_Mat)
    # print(standPoint)
    # 获取训练数据集的行数, 也就是样本个数
    train_num = np.shape(x_Mat)[0]
    # print("training dataset lines number: %d" % (train_num))
    # 使用eye()函数生成一个主对角线元素为1, 其它位置元素为0的二维数组, 使用np.mat创建权重矩阵weight
    weights = np.mat(np.eye(train_num))
    for idx in range(train_num):
        # 计算standPoint与其它样本点之间的距离, 然后计算每个样本点对standPoint的误差贡献值
        diff_point = standPoint - x_Mat[idx,:]
        # if idx % 1000 == 0:
        #     print("calculate weight process: %d" % (idx))
        # print(diff_point.shape)
        # print(type(diff_point))
        # print((diff_point*diff_point.T / (-2.0 * k**2)).tolist()[0][0])
        # k控制高斯权重衰减速度
        # weights[idx, idx] = np.exp(diff_point.T * diff_point / (-2.0 * k**2))
        weights[idx, idx] = np.exp((diff_point*diff_point.T / (-2.0 * k**2)).tolist()[0][0])
    # print(weights.shape)
    # 根据最小二乘法公式,
    # 可以推导出ws = (X.T * W * X)^-1 * X.T * W * Y
    xTx = x_Mat.T * (weights * x_Mat)
    if np.linalg.det(xTx) == 0.0:
        raise Exception("the martix is singular, can not do inverse!")
    ws = xTx.I * (x_Mat.T * (weights * y_Mat))
    # print(ws)
    # print(standPoint * ws)
    return standPoint * ws

def LWLR_Test(standArr, xArr, yArr, k=1.0):
    """
    测试局部线性回归模型, 对于数据集中的每个点调用LWLR_Model()函数
    :param standArr: 测试集
    :param xArr: 样本的特征数据, 即为feature
    :param yArr: 样本的类别标签, 即为label
    :param k: 控制核函数的衰减速率
    :return: 预测点的估计值
    """
    # stand_num = standArr.shape[0]
    stand_num = np.shape(standArr)[0]
    print("stand num: {}".format(stand_num))
    pre_y = np.zeros(stand_num)
    for idx in range(stand_num):
        print("[%d/%d]" % (idx, stand_num))
        pre_y[idx] = LWLR_Model(standArr[idx], xArr, yArr, k)
    return pre_y

def calculate_err(yArr, pre_yArr):
    """
    返回最终预测的y值与真实的y值之间的误差
    :param yArr: 样本的真实值
    :param pre_yArr: 样本的预测值
    :return: 预测的总体误差
    """
    print(yArr.shape)
    print(pre_yArr.shape)
    return ((yArr - pre_yArr)**2).sum()

def Load_DataSet(file_route):
    """
    加载训练数据集
    :param file_route: 训练数据集的路径
    :return: 加载后的数据集
    """
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

def run_regression():
    # 加载鲍鱼数据集
    sampleX, sampleY = Load_DataSet("./dataset/abalone.txt")
    # 使用不同的核参数进行预测, 取后100个样本作为测试集, 剩余为训练集
    # sampleX = np.mat(sampleX)
    # sampleY = np.mat(sampleY)
    pre_yHat_A = LWLR_Test(sampleX[len(sampleX)-100:], sampleX[:-100], sampleY[:-100], 0.1)
    pre_yHat_B = LWLR_Test(sampleX[len(sampleX) - 100:], sampleX[:-100], sampleY[:-100], 1)
    pre_yHat_C = LWLR_Test(sampleX[len(sampleX) - 100:], sampleX[:-100], sampleY[:-100], 10)
    # print(type(sampleY))
    print("pre_yHat_A Loss: {}".format(calculate_err(np.array(sampleY[len(sampleX)-100:]), pre_yHat_A)))
    print("pre_yHat_B Loss: {}".format(calculate_err(np.array(sampleY[len(sampleX) - 100:]), pre_yHat_B)))
    print("pre_yHat_B Loss: {}".format(calculate_err(np.array(sampleY[len(sampleX) - 100:]), pre_yHat_C)))
    # print(pre_yHat_A)

run_regression()