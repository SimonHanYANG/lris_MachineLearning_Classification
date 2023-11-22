import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def IRISDataset():
    # 加载数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("X_train Shape: {}.".format(X_train.shape))
    print("X_test Shape: {}.".format(X_test.shape))
    print("y_train Shape: {}.".format(y_train.shape))
    print("y_test Shape: {}".format(y_test.shape))
    
    return  X_train, X_test, y_train, y_test


# test dataset func
# X_train, X_test, y_train, y_test = IRISDataset()

'''
    IRIS(鸢尾花) Dataset:

    这是一个用于多分类问题的数据集，包含了150个样本，每个样本有4个特征和一个标签

    - 特征（Features）：

        1. 花萼长度（sepal length in cm）
        2. 花萼宽度（sepal width in cm）
        3. 花瓣长度（petal length in cm）
        4. 花瓣宽度（petal width in cm）
        p.s. 这些特征都是连续型的数值。

    - 标签（Targets）：数据集包含三种不同类型的鸢尾花，每种各50个样本。这些类型是：

        1. Iris Setosa
        2. Iris Versicolour
        3. Iris Virginica
'''
