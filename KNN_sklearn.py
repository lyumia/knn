import numpy as np
from math import sqrt
from collections import Counter

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
        
    def predict(self,X_predict):
        assert self._X_train is not None,"要求predict 之前要先运行 fit 这样self._X_train 就不会为空"
        assert self._y_train is not None
        assert X_predict.shape[1] == self._X_train.shape[1],"要求测试集和预测集的特征数量一致"

        distances = [sqrt(np.sum((x_train - X_predict)**2)) for x_train in self._X_train]
        sort = np.argsort(distances)
        topK = [self._y_train[i] for i in sort[:self.k]]
        votes = Counter(topK)
        y_predict = votes.most_common(1)[0][0]
        return y_predict
