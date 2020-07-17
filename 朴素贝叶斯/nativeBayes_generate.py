import numpy as np
import os, sys
from functools import reduce

def loadDataType():
    posting_list = [
        ['my', 'dog', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'tale', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVector = [0, 1, 0, 1, 0, 1]
    return posting_list, classVector

def Generate_Vocab(dataSet):
    word_vocab = set()
    for document in dataSet:
        word_vocab = word_vocab | set(document)
    return list(word_vocab)

def word2Vec(vocab_list, inputSet):
    returnVec = [0] * len(vocab_list)
    for word in inputSet:
        if word in vocab_list:
            returnVec[vocab_list.index(word)] = 1
        else: print("the word: {} is not in my vocabulary!".format(word))
    return returnVec

def train_NativeBayes(trainMatrix, trainCategory):
    num_traindocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory) / float(num_traindocs)
    p0Num, p1Num = np.zeros(numWords), np.zeros(numWords)
    p0Denom, p1Denom = 0.0, 0.0
    for idx in range(num_traindocs):
        if trainCategory[idx] == 1:
            p1Num += trainMatrix[idx]
            p1Denom += sum(trainMatrix[idx])
        else:
            p0Num += trainMatrix[idx]
            p0Denom += sum(trainMatrix[idx])
    print(p0Num)
    print(p1Num)
    p1Vec = p1Num / p1Denom
    p0Vec = p0Num / p0Denom
    print("p0Vec: " + str(p0Vec))
    print("p1Vec: " + str(p1Vec))
    return p0Vec, p1Vec, pAbusive

def classify(vec2Classify, p0Vec, p1Vec, pClass1):
    print(vec2Classify * p0Vec)
    print(vec2Classify)
    print(p0Vec)
    p0_res, p1_res = 0.0, 0.0
    for node in vec2Classify * p0Vec:
        if node > 0.0 and p0_res == 0.0: p0_res = node
        elif node > 0.0: p0_res *= node
        else: continue
    for node in vec2Classify * p1Vec:
        if node > 0.0 and p1_res == 0.0: p1_res = node
        elif node > 0.0: p1_res *= node
        else: continue
    print(vec2Classify * p1Vec)

    print("p0: " + str(p0_res))
    print("p1: " + str(p1_res))
    if p1_res >= p0_res: return True
    else: return False

def run_model():
    Posts, Classes = loadDataType()
    Vocab = Generate_Vocab(Posts)
    trainMatrix = list()
    print(Posts)
    print(str(Vocab) + " \n" + str(len(Vocab)))
    for postinDoc in Posts:
        trainMatrix.append(word2Vec(Vocab, postinDoc))
    print(trainMatrix)

    p0V, p1V, pAb = train_NativeBayes(np.array(trainMatrix), np.array(Classes))
    print(type(p0V))
    test = ['love', 'my', 'dalmation']
    test_vector = np.array(word2Vec(Vocab, test), dtype=float)
    print(test_vector)
    print(test_vector.shape)
    if classify(test_vector, p0V, p1V, Classes):
        print("it is False!")
    else: print("it is a True!")

run_model()
