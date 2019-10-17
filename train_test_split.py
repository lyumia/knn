#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
from math import ceil

def train_test_split(X, y, test_ratio=0.3, seed= None):
    assert X.shape[0] == y.shape[0], 'X y 的行数要一样'
    if seed:
        np.random.seed(seed)
    # 观察到原始数据集的标签值是从 0 到 2 有序排列的，所以不能直接划分，要先把数据集打乱保证随机抽样。
    # 打乱可以用 numpy 的 permutation 函数，它会返回打乱后的数据集的索引，
    # 这个函数的妙处就在于根据索引就能同时匹配到 X 和 y。二者是一一对应的。
    shuffle_index = np.random.permutation(len(X))
    test_size = int(ceil(len(X) * test_ratio))
    test_index = shuffle_index[:test_size]
    train_index = shuffle_index[test_size:]
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    return X_train, X_test, y_train, y_test






