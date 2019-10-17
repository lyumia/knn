from KNN_sklearn import KNNClassifier

# 样本集
X_raw = [[13.23,  5.64],
       [13.2 ,  4.38],
       [13.16,  4.68],
       [13.37,  4.8 ],
       [13.24,  4.32],
       [12.07,  2.76],
       [12.43,  3.94],
       [11.79,  3.  ],
       [12.37,  2.12],
       [12.04,  2.6 ]]
X_train = np.array(X_raw)

# 特征值
y_raw = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_train = np.array(y_raw)

# 待预测值
# 当预测变量只有一个时，一定要 reshape(1,-1) 成二维数组不然会报错。
x_test= np.array([12.08,  3.3])
X_predict = x_test.reshape(1,-1)

# 调用程序中的 kNNClassifier 类，赋予 k 参数为 3，命名为一个实例 kNN_classify 。
KNN_clssify = KNNClassifier(3)
# 把样本集 X_train，y_train 传给实例 fit 
kNN_classify.fit(X_train,y_train)
# fit 好后再传入待预测样本 X_predict 进行预测就可以得到分类结果了
y_predict = kNN_classify.predict(X_predict)
print y_predict
