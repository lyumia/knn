# knn
https://www.dazhuanlan.com/2019/10/05/5d9846c0d167b/

kNN超参数
knn_clf = KNeighborsClassifier(algorithm='brute', n_jobs=-1, n_neighbors=k, weights='distance', p=p)

1 algorithm
  algorithm 即算法，意思就是建立 kNN 模型时采用什么算法去搜索最近的 k 个点，有四个选项：
    brute（暴力搜索）
      计算预测样本和全部训练集样本的距离，最后筛选出前 K 个最近的样本，这种方法很蛮力，所以叫暴力搜索。
      当样本和特征数量很大的时候，每计算一个样本距离就要遍历一遍训练集样本，很费时间效率低下。
    kd_tree（K-dimension tree）
      简单来说 KD 树是一种「二叉树」结构
      采用这种方法不用遍历全部样本就可以快速找到最近的 K 个点，速度比暴力搜索快很多
      假设数据集样本数为 m，特征数为 n，则当样本数量 m 大于 2 的 n 次方时，用 KD 树算法搜索效果会比较好。
      一旦特征过多，KD 树的搜索效率就会大幅下降，最终变得和暴力搜索差不多。
      通常来说 KD 树适用维数不超过 20 维的数据集，超过这个维数可以用球树这种算法。
    ball_tree（球树）
    auto（默认值，自动选择上面三种速度最快的）
    
  当数据集样本数量 m > 2 的 n 次方时，kd_tree 和 ball_tree 速度比 brute 暴力搜索快了一个量级，auto 采用其中最快的算法。
  当数据集样本数量 m 小于 2 的 n 次方 时，KD 树和球树的搜索速度大幅降低，暴力搜索算法相差无几。
  
2 n_neighbors
  n_neighbors 即要选择最近几个点，默认值是 5（等效 k )。

3 距离权重
  在 Sklearn 的 kNN 模型中有专门考虑权重的超参数：weights，它有两个选项：
    uniform 不考虑距离权重，默认值
    distance 考虑距离权重
  
4 p
  欧拉距离 p=2
  曼哈顿距离 p=1
  明可夫斯基距离（Minkowski Distance）
  
