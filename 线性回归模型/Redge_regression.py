import numpy as np
import os,sys

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

def Redge_regression(x_Mat, y_Mat, lamda=0.2):
    """
    这个函数给出了给定lamda下的岭回归求解。如果使用之前用的线性回归方法是不可行的, 因为计算(XTX)^-1会出现错误.
    如果特征比样本点还多, 也就是说数据的矩阵不是满秩矩阵。非满秩矩阵在求逆的时候会出现问题.
    使用岭回归方法可以避免这一问题.
    :param x_Mat: 样本的特征数据, 即为feature
    :param y_Mat: 样本的标签数据, 即为label
    :param lamda: L2 范数的系数
    :return: 经过岭回归公式计算得到的回归系数
    """
    xTx = x_Mat.T * x_Mat
    denom = xTx + np.eye(np.shape(x_Mat)[1]) *lamda
    # 检查行列式是否为0, 即矩阵是否可逆, 行列式为0的话就不可逆, 行列式不为0的话则可逆
    if np.linalg.det(denom) == 0.0:
        raise Exception("the martix is singular, can not do inverse!")
    ws = denom.I * (x_Mat.T * y_Mat)
    return ws

def Redge_Test(x_Arr, y_Arr):
    """
    测试岭回归结果
    :param x_Arr: 样本数据特征, 即为feature
    :param y_Arr: 样本数据的类别标签
    :return: 将所有的回归系数输出到矩阵并返回
    """
    x_Mat = np.mat(x_Arr)
    y_Mat = np.mat(y_Arr).T
    # 计算Y的均值
    y_mean = np.mean(y_Mat, 0)
    y_Mat = y_Mat - y_mean
    x_mean = np.mean(x_Mat, 0)
    x_var = np.var(x_Mat, 0)
    x_Mat = (x_Mat - x_mean) / x_var
    ws = Redge_regression(x_Mat, y_Mat, np.exp(1-10))
    return ws

def run_redge_regression():
    sampleX, sampleY = Load_DataSet("./dataset/abalone.txt")
    ws = Redge_Test(sampleX, sampleY)

run_redge_regression()