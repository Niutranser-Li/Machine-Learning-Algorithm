import numpy as np
import os, sys

"""
程序主要功能: 使用KNN方法解决海伦小姐的约会预测(改进约会网站的配对效果)。
数据集介绍: 
数据集产生的由来是这样的, 说海伦小姐收集约会的数据已经有一段时间了，她和不同的人约了1000次会之后(震惊脸！),收集了1000条数据
整个数据集分为四列, 前三列分别为与她约会男士的特征, 最后一列为标签(海伦小姐是否喜欢)
每列所表示的特征为: 每年获得的飞行常客里程数 / 玩游戏视频所消耗时间百分比 / 每周消费的冰淇淋公升数 / 海伦小姐是否喜欢
其中第四列为标签列, 可以有三种取值(didntLike(不喜欢的人) / smallDoses(魅力一般的人) / largeDoses(极具魅力的人))

使用常规的KNN方法根据三个特征预测样本点所代表的人是否是海伦喜欢的人。
"""

def loadDataSet(dataset_route):
    """
    加载训练集合
    :param dataset_route: 训练集合路径
    :return: Feature_Matrix: 特征矩阵  Y_Matrix: 标签矩阵
    """
    if not os.path.exists(dataset_route):
        raise Exception("error! not found the dataset file route: %s" % (dataset_route))
    filein = open(dataset_route, "r", encoding="utf-8")
    memory_lines = filein.readlines()
    line = memory_lines[0]
    feature_num = len(line.strip().split("\t")) - 1
    filein.close()

    line_num = len(memory_lines)
    print("Read DataSet Successful! DataSet Size: %d, Feature Num: %d" % (line_num, feature_num))
    feature_matrix, label_matrix = list(), list()
    for idx, line in enumerate(memory_lines):
        line = line.strip()
        line_ext = line.split("\t")
        feature_matrix.append(line_ext[:-1])
        if line_ext[-1] == "didntLike": label_matrix.append(1)
        if line_ext[-1] == "smallDoses": label_matrix.append(2)
        if line_ext[-1] == "largeDoses": label_matrix.append(3)
    feature_matrix = np.array(feature_matrix).astype(np.float)
    label_matrix = np.array(label_matrix)
    return feature_matrix, label_matrix

def feature_normlization(dataSet):
    """
    特征归一化，在KNN分类算法过程中，若不进行特征值归一化，则容易导致若某个属性的特征值与其它特征值差异较大
    主要造成的影响: 1) 特征值的取值范围较大往往会极大地增大两个样本间的距离度量结果
    2) 特征之间的值差异较大, 造成梯度下降速度变慢

    归一化方法: x = (x - minValue) / (maxValue - minValue) -> [0, 1]
    :param dataSet: 特征矩阵
    :return: 经过归一化的特征矩阵
    """
    min_vals = dataSet.min(0)
    max_vals = dataSet.max(0)

    ranges = max_vals - min_vals
    norm_Matrix = np.zeros(np.shape(dataSet))
    rows = dataSet.shape[0]
    # np.tile() 方法可以实现矩阵的平铺, 第一个参数为平铺操作的基准数据, 后面第一个参数为沿着Y轴平铺的倍数, 第二个参数为沿着X轴平铺的倍数
    norm_Matrix = dataSet - np.tile(min_vals, (rows, 1))
    norm_Matrix = norm_Matrix / np.tile(ranges, (rows, 1))
    return norm_Matrix, ranges, min_vals

def split_train_test(X_Matrix, Y_Matrix, ratio):
    """
    交叉验证切分函数, 将特征矩阵和标签矩阵按照一定比例进行切分
    一部分作为训练集, 另一部分作为测试集
    :param X_Matrix: 待切分的特征矩阵
    :param Y_Matrix: 待切分的标签矩阵
    :param ratio: 切分比例
    :return: X_Train: 训练特征矩阵 X_Valid: 测试特征矩阵  Y_Train: 训练标签矩阵  Y_Valid: 测试标签矩阵
    """
    X_rows = X_Matrix.shape[0]
    train_rows = np.int(X_rows * ratio)
    X_Train = X_Matrix[:train_rows]
    Y_Train = Y_Matrix[:train_rows]
    X_Valid = X_Matrix[train_rows:]
    Y_Valid = Y_Matrix[train_rows:]
    return X_Train, X_Valid, Y_Train, Y_Valid

def sample_kNN_Classifier(TestX_Matrix, TrainX_Matrix, TrainY_Matrix, K):
    """
    对待分类的测试集执行KNN分类, 距离度量采用欧式距离来完成.
    :param TestX_Matrix: 测试集特征矩阵
    :param TrainX_Matrix: 训练集特征矩阵
    :param TrainY_Matrix: 训练集标签矩阵(最终根据找到的最近的K个样本的分类, 执行加权多数表决的方式作为最终分类结果)
    :param K: 设置每次预测选定最近的多少个点.
    :return: 返回分类结果
    """
    print("TrainX_Matrix Shape: {}".format(TrainX_Matrix.shape))
    print("TestX_Matrix Shape: {}".format(TestX_Matrix.shape))
    distance = np.sqrt(np.sum((TrainX_Matrix - TestX_Matrix)**2, axis=1))
    ind = np.argsort(distance)
    class_count = {}
    for idx in range(K):
        vote = TrainY_Matrix[ind[idx]]
        class_count[vote] = class_count.get(vote, 0) + 1
    class_count = sorted(class_count.items(), key=lambda item:item[1], reverse=True)
    return class_count[0][0]

X_Matrix, Y_Matrix = loadDataSet("./dataset/datingTestSet.txt")
xNorm_Matrix, Range, min_vals = feature_normlization(X_Matrix)
X_train, X_valid, Y_train, Y_valid = split_train_test(X_Matrix, Y_Matrix, 0.8)
print("x_train: " + str(X_train.shape))
print("x_valid: " + str(X_valid.shape))
print("y_train: " + str(Y_train.shape))
print("y_valid: " + str(Y_valid.shape))
lines = int(X_valid.shape[0])
accurate = 0
for idx in range(lines):
    class_result = sample_kNN_Classifier(X_train, X_valid[idx, :], Y_train, K=10)
    accurate += int(Y_valid[idx] == class_result)
acc = accurate / len(X_valid) * 100.0
print(str(acc) + "%")