#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from math import sqrt
from collections import Counter
from metrics_score import accuracy_score

class kNNClassifier:
    def __init__(self,k):
        self.k =k
        self._X_train = None
        self._y_train = None

    def fit(self,X_train,y_train):
        assert X_train.shape[0] == y_train.shape[0],"添加 assert 断言是为了确保输入正常的数据集和k值，如果不添加一旦输入不正常的值，难找到出错原因"
        assert self.k <= X_train.shape[0]
        self._X_train = X_train
        self._y_train = y_train
        return self
        
    # def predict(self,X_predict):
    #     assert self._X_train is not None,"要求predict 之前要先运行 fit 这样self._X_train 就不会为空"
    #     assert self._y_train is not None
    #     assert X_predict.shape[1] == self._X_train.shape[1],"要求测试集和预测集的特征数量一致"
    #
    #     distances = [sqrt(np.sum((x_train - X_predict)**2)) for x_train in self._X_train]
    #     sort = np.argsort(distances)
    #     topK = [self._y_train[i] for i in sort[:self.k]]
    #     votes = Counter(topK)
    #     y_predict = votes.most_common(1)[0][0]
    #     return y_predict

    # predict 用列表生成式来存储多个预测分类值，预测值从哪里来呢，
    # 就是利用 _predict 函数计算，_predict 前面的下划线同样表明它是封装的私有函数，只在内部使用，外界不能调用，因为不需要。
    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]  # 列表生成是把分类结果都存储到list 中然后返回
        return np.array(y_predict)

    def _predict(self, x):  # _predict私有函数
        assert self._X_train is not None
        assert self._y_train is not None

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        sort = np.argsort(distances)
        topK = [self._y_train[i] for i in sort[:self.k]]
        votes = Counter(topK)
        y_predict = votes.most_common(1)[0][0]
        return y_predict

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test,y_predict)