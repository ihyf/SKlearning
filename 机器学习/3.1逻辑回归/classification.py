# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 10:09
# @Author  : ihyf
# @File    : classification.py
# @Software: PyCharm
# @ Desc :
import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class Classifocatison(object):
    @staticmethod
    def data_preprocessing_template():
        data_set = pd.read_csv("Social_Network_Ads.csv")
        x = data_set.iloc[:, 2: 4]
        y = data_set.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

        # 特征缩放 标准化数据 我们可以将特征中的值进行标准差标准化，即转换为均值为0，方差为1的正态分布
        """
        StandardScaler类是一个用来讲数据进行归一化和标准化的类。
        所谓归一化和标准化，即应用下列公式：
        X=(x-\mu)/\sigma
        使得新的X数据集方差为1，均值为0
        fit_transform方法是fit和transform的结合，fit_transform(X_train) 意思是找出X_train的和，并应用在X_train上。
        这时对于X_test，我们就可以直接使用transform方法。因为此时StandardScaler已经保存了X_train的和。
        """
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        return x_train, x_test, y_train, y_test

    def classifocatison(self):
        x_train, x_test, y_train, y_test = self.data_preprocessing_template()
        # 拟合逻辑回归器
        logistic_reg = LogisticRegression(random_state=0)
        logistic_reg.fit(x_train, y_train)
        # 预测测试集
        y_pred = logistic_reg.predict(x_test)
        # 用混淆矩阵 评估结果
        cm = confusion_matrix(y_test, y_pred)
        # 图像展示结果分布
        from matplotlib.colors import ListedColormap
        x_set, y_set = x_train, y_train
        # 生成所有像素点
        x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min()-1, stop=x_set[:, 0].max()+1, step=0.01),
                             np.arange(start=x_set[:, 1].min()-1, stop=x_set[:, 1].max()+1, step=0.01))
        # 逻辑回归器拟合所有像素点
        plt.contourf(x1, x2, logistic_reg.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())

        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('orange', 'blue'))(i), label=j)
        plt.title('train set')
        plt.xlabel('age')
        plt.ylabel('salary')
        plt.legend()
        plt.show()


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cf = Classifocatison()

    def test_data_preprocessing_template(self):
        self.cf.data_preprocessing_template()

    def test_classifocatison(self):
        self.cf.classifocatison()