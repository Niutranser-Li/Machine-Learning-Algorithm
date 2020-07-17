import numpy as np
import os, sys
import re
"""
使用朴素贝叶斯方式完成垃圾邮件分类器, 使用学习完成后的模型能够自动将模型归类为垃圾邮件或非垃圾邮件
数据集介绍: 数据集中ham文件夹中存放的是25个正常的邮件(程序中用非垃圾邮件表示), spam中存放的是25个垃圾邮件(用垃圾邮件标签表示)
"""

def str2list(str_content):
    """
    接收一个字符串, 使用特殊的切分符号进行字符串切分"/W"-按照非字母/非字符进行字符串分割(产生的空字符会在后续步骤中被过滤掉)
    对于切分出来的字符串列表, 将长度大于等于2的单词全部进行小写化(能够减少词汇的多样性, 减小词汇表大小)
    :param str_content: 待切分的长字符串
    :return: 切分后的列表
    """
    token_list = re.split(r'\W+', str_content)
    return [token.lower() for token in token_list if len(token) > 2]

def generate_Vocab(email_content):
    """
    根据邮件数据集中存在的内容创建词汇表, 词汇表中的单词均唯一
    :param email_content: 全部邮件数据集的内容, 包括垃圾邮件及非垃圾邮件
    :return: 返回词汇表, 类型是list类型
    """
    vocab = list()
    for document in email_content:
        vocab.extend(document)
    vocab = set(vocab)
    return list(vocab)

def word2Vec(vocab, input):
    """
    句子向量化, 将邮件内容表示为One-hot向量的形式, 方便处理
    :param vocab: 词汇表
    :param input: 待转换列表
    :return: 词向量化后的结果
    """
    final_vec = [0] * len(vocab)
    for word in input:
        if word in vocab: final_vec[vocab.index(word)] = 1
        else: continue
    return final_vec

def split_train_and_test(ratio = 0.8):
    """
    切分训练数据及测试数据, 按照索引分别生成训练数据集以及测试数据集
    :param ratio: 切分比例(在训练数据集中的保留比例)
    :return: 返回带有索引值的训练数据集以及测试数据集
    """
    train_set, test_set = list(range(50)), list()
    split_num = round(50 * (1.0 - ratio))
    for idx in range(split_num):
        random_idx = int(np.random.uniform(0, len(train_set)))
        test_set.append(train_set[random_idx])
        del(train_set[random_idx])
    return train_set, test_set

def load_DataSet():
    doc_list, class_list, full_list = list(), list(), list()
    for idx in range(1, 26):
        # 使用str2list函数直接将整个邮件中的内容转换为长字符串后又转换成为列表的形式, 此处读取垃圾邮件
        word_list = str2list(open("./dataset/email/spam/%d" % idx, "r").read())
        doc_list.append(word_list)
        full_list.append(word_list)
        class_list.append(1)
        # 读取非垃圾邮件信息, 同样使用str2list函数对其进行转换
        word_list = str2list(open("./dataset/email/ham/%d" % idx, "r").read())
        doc_list.append(word_list)
        full_list.append(word_list)
        class_list.append(0)
    print("email information loading finish! sample number is: %d" % (len(doc_list)))
    # 根据读取得到的邮件内容, 生成对应的词汇表
    vocab = generate_Vocab(doc_list)
    print("generate vocab finish, vocab size is {}".format(len(vocab)))
    train_set, test_set = split_train_and_test(0.8)
    return doc_list, full_list, class_list, vocab, train_set, test_set

def native_bayes_classifier_train(train_feature_Matrix, train_label_Matrix):
    """
    朴素贝叶斯训练方法, 对于模型参数估计部分使用了拉普拉斯平滑方法
    :param train_feature_Matrix: 训练特征向量矩阵
    :param train_label_Matrix: 训练标签矩阵
    :return: 垃圾邮件的条件概率, 非垃圾邮件的条件概率, 先验概率
    """
    # 计算训练文本的总数目, 用于计算文档属于侮辱类的先验概率
    train_sample_num = len(train_feature_Matrix)
    # 计算文档属于侮辱类的先验概率
    pre_ratio = sum(train_label_Matrix) / float(train_sample_num)
    word_number = len(train_feature_Matrix[0])
    # 创建两个不同类别之间的单词出现个数, 使用拉普拉斯平滑, 各个单词初始化频率为1
    class1_ratio, class0_ratio = np.ones(word_number), np.ones(word_number)
    class1_sum, class0_sum = 2.0, 2.0
    for idx in range(train_sample_num):
        if train_label_Matrix[idx] == 1:
            class1_ratio += train_feature_Matrix[idx]
            class1_sum += sum(train_feature_Matrix[idx])
        else:
            class0_ratio += train_feature_Matrix[idx]
            class0_sum += sum(train_feature_Matrix[idx])
    # 对计算得到的概率结果取对数, 防止概率值下溢
    class1_vec = np.log(class1_ratio / class1_sum)
    class0_vec = np.log(class0_ratio / class0_sum)
    return class0_vec, class1_vec, pre_ratio

def native_bayes_classifier_test(str_vec, class1_vec, class0_vec, pre_ratio):
    """
    根据待预测样本, 使用训练完成后的贝叶斯模型参数进行预测
    :param str_vec: 转换成向量的结果
    :param class1_vec: 垃圾邮件的条件概率
    :param class0_vec: 非垃圾邮件的条件概率
    :param pre_ratio: 先验概率
    :return: 分类结果 1-垃圾邮件  0-非垃圾邮件
    """
    # 计算待测样本属于垃圾邮件的概率, 由于log A * B * C * ... = log A + log B + log C + ..., 因此后面+np.log(.)
    class1_pre = sum(str_vec * class1_vec) + np.log(pre_ratio)
    class0_pre = sum(str_vec * class0_vec) + np.log(1.0 - pre_ratio)
    if class1_pre > class0_pre: return 1
    else: return 0

def native_Bayes_email_classify():
    """
    使用朴素贝叶斯方法进行训练以及测试
    :return:None
    """
    doc_list, full_list, class_list, vocab, train_set, test_set = load_DataSet()
    train_feature_Matrix, train_label_Matrix = list(), list()
    for doc_idx in train_set:
        train_feature_Matrix.append(word2Vec(vocab, doc_list[doc_idx]))
        train_label_Matrix.append(class_list[doc_idx])
    print("generate train feature and label matrix finish! feature:{} label:{}".format(len(train_feature_Matrix), len(train_label_Matrix)))
    test_feature_Matrix, test_label_Matrix = list(), list()
    for idx in test_set:
        test_feature_Matrix.append(word2Vec(vocab, doc_list[idx]))
        test_label_Matrix.append(class_list[idx])
    print("generate test feature and label matrix finish! feature:{} label:{}".format(len(test_feature_Matrix), len(test_label_Matrix)))

    # 使用朴素贝叶斯模型进行训练, 分别返回分类为0/1的条件概率以及先验概率
    class0_vec, class1_vec, pre_ratio = native_bayes_classifier_train(train_feature_Matrix, train_label_Matrix)

    for idx in range(len(test_feature_Matrix)):
        class_res = native_bayes_classifier_test(test_feature_Matrix[idx], class1_vec, class0_vec, pre_ratio)
        print("pre value: {}  true value: {}".format(class_res, test_label_Matrix[idx]))

native_Bayes_email_classify()