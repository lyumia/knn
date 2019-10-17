#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn import datasets

# 加载葡萄酒数据集
wine = datasets.load_wine()
#查看wine能调用的方法
# print wine.keys()

X = wine.data
# print X[:, 3]
# print X.shape
y = wine.target
# print y
# print y.shape

# 调用 Sklearn 数据集划分函数 train_test_split
# test_size 是测试集比例，random_state 是一个随机种子，保证每次运行都能得到一样的随机结果
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=321)
# print X_train.shape
# print X_test.shape
# print y_train.shape
# print y_test.shape

# 实现train_test_split()
from train_test_split import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,seed=321)
# print X_train.shape
# print X_test.shape
# print y_train.shape
# print y_test.shape

# 调用 Skearn 中的 kNN 模型计算模型得分
# from sklearn.neighbors import KNeighborsClassifier
# kNN_clf = KNeighborsClassifier(n_neighbors=3)
# kNN_clf.fit(X_train,y_train)
# y_predict = kNN_clf.predict(X_test);
# from sklearn.metrics import accuracy_score
# # accuracy_score 函数利用 y_test 和 y_predict 计算出得分
# print accuracy_score(y_test,y_predict)

from KNN_sklearn import kNNClassifier
knn_clf = kNNClassifier(k=3)
knn_clf.fit(X_train,y_train)
y_predict = knn_clf.predict(X_test)
# # print y_predict
# # print ''
# # print  y_test
# accuracy_score = sum(y_predict==y_test)/float(len(y_test))
# print 'accuracy score: '+ str(accuracy_score)

# from metrics_score import accuracy_score
# score = accuracy_score(y_test,y_predict)

score = knn_clf.score(X_test,y_test)
print 'accuracy score: '+ str(score)




