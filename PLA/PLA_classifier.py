import numpy as np
import os, sys

"""
使用感知机(Perception Learning Algorithm, PLA)实现IRIS(鸢尾花卉)数据集的二分类问题
已知鸢尾花数据集中共有150条数据,每列之间使用","进行分隔,各列所表示的属性如下所示:
1 - SepaLengthCm: 花萼长度, 单位cm
2 - SepalWidthCm: 花萼宽度, 单位cm
3 - PatalLengthCm: 花瓣长度, 单位cm
4 - PetalWidthCm: 花瓣宽度, 单位cm
5 - Species: 鸢尾花种类(三类)
我们将类别为: Iris-setosa 划分为正类, 其它种类划分为父类,具体代码如下所示
"""
class PLA_Classifier(object):
    def __init__(self, learning_rate=0.01, n_iter=30, random_seed=1):
        """
        PLA分类初始化函数
        :param learning_rate: 学习率, 每次模型权重以及偏置的更新幅度
        :param n_iter: 模型学习轮数: 表示模型在数据集上学习多少次, 每次过一遍全部数据集
        :param random_seed: 随机种子，用于随机生成初始化权重, 种子相同,每次生成的伪随机数相同
        """
        self.learning_rate = learning_rate
        self.train_epoch = n_iter
        self.random_seed = random_seed

    def model_training(self, train_X, train_Y, valid_X, valid_Y):
        """
        PLA 模型训练函数
        :param train_X: 训练集合X(里面存的都是特征值), 矩阵维度为: sample * feature_num
        :param train_Y: 训练集合Y(里面存的都是标签值), 矩阵维度: 1 * sample
        :param valid_X: 校验集合X
        :param valid_Y: 校验集合Y
        :return: None
        """
        print("loading PLA model to classifier! train_X shape: %s train_Y shape: %s" % (str(train_X.shape), str(train_Y.shape)))
        weight_dimen = train_X.shape[1]
        print("PLA Model Weight dimension: 1 * %s" % weight_dimen)
        random_generate = np.random.RandomState(self.random_seed)
        # model_weight = random_generate.normal(loc=0.0, scale=0.01, size=weight_dimen)
        model_weight = np.mat(np.ones((1, weight_dimen)))
        model_bias = 0
        train_sample_num = int(train_X.shape[0])

        for epoch_idx in range(1, self.train_epoch+1):
            error_num = 0
            loss = 0
            for x, y in zip(train_X, train_Y):
                y_predict = float(x * model_weight.T) + model_bias
                if y * y_predict < 0:
                    error_num += 1
                    model_weight += self.learning_rate * x * y
                    model_bias += self.learning_rate * y
                    loss += y_predict * y * (-1)
                else: continue
            error_ratio = round(error_num / train_sample_num * 100.0, 2)
            loss = round(loss, 1)
            print("training epoch:[%d/%d], error ratio: %s%s loss: %s" % (epoch_idx, self.train_epoch, str(error_ratio), "%", str(loss)))
        print("training finish!")
        self.model_valid(valid_X, valid_Y, model_weight, model_bias)

    def model_valid(self, X_Valid, Y_Valid, model_weight, model_bias):
        """
        PLA 模型校验函数, 使用校验集合, 计算训练完成后模型的泛化性
        :param X_Valid: 校验集合X
        :param Y_Valid: 校验集合Y
        :param model_weight: 训练完成后PLA模型的权重矩阵，由模型训练函数得出
        :param model_bias:  训练完成后PLA模型的偏置项, 由模型训练函数得出
        :return: 模型在校验集合上的表现
        """
        valid_num = np.int(X_Valid.shape[0])
        err_num, precision = 0, 0
        for x, y in zip(X_Valid, Y_Valid):
            y_predict = float(x * model_weight.T) + model_bias
            if y * y_predict < 0:
                err_num += 1
            else: continue
        precision = err_num / valid_num * 100.0
        print("Test model precision: %s%s" % (str(precision), '%'))

def loadDataSet(file_route):
    """
    加载数据集合, 并生成矩阵形式
    :param file_route: 数据集的路径
    :return: 生成的训练集合以及标签集合
    """
    if not os.path.exists(file_route):
        raise Exception("error! the input file: {} is not found!".format(file_route))
    filein = open(file_route, "r", encoding="utf-8")
    feature_num = len(open(file_route, "r", encoding="utf-8").readline().strip().split(",")) - 1
    print("check the feature number: {}".format(feature_num))
    XArr, YArr = list(), list()
    for line in filein:
        temp = list()
        line_ext = line.strip().split(",")
        for idx in range(feature_num): temp.append(float(line_ext[idx]))
        XArr.append(temp)
        if line_ext[-1] == "Iris-setosa": YArr.append(1)
        else: YArr.append(-1)
    filein.close()
    print("X_DataSet Matrix Shape: %s" % (str(np.array(XArr).shape)))
    print("Y_DataSet Matrix Shape: %s" % (str(np.array(YArr).shape)))
    return XArr, YArr

def Split_DataSet(XArr, YArr, split_ratio=0.8):
    split_idx = np.int(np.array(XArr).shape[0] * split_ratio)
    X_Train = np.array(XArr[:split_idx])
    Y_Train = np.array(YArr[:split_idx])
    X_Valid = np.array(XArr[split_idx:])
    Y_Valid = np.array(YArr[split_idx:])
    return X_Train, Y_Train, X_Valid, Y_Valid

XArr, YArr = loadDataSet("./IRIS-data")
X_Train, Y_Train, X_Valid, Y_Valid = Split_DataSet(XArr, YArr)
PLA = PLA_Classifier()
PLA.model_training(X_Train, Y_Train, X_Valid, Y_Valid)
