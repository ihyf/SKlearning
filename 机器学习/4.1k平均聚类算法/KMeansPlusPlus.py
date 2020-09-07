# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 10:08
# @Author  : ihyf
# @File    : K-means++.py
# @Software: PyCharm
# @ Desc :
import unittest
import matplotlib.pyplot as plt
import pandas as pd


class KMeansPlusPlusAAAA(object):
    @staticmethod
    def data_preprocessing_template():
        data_set = pd.read_csv("Mall_Customers.csv")
        x = data_set.iloc[:, 3: 5]
        return x

    def k_means_plus_plus_function(self):
        x = self.data_preprocessing_template()
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1, 11):
            k_means = KMeans(n_clusters=i, max_iter=300, n_init=10, init="k-means++", random_state=0)
            # 拟合 k_means 聚类器
            k_means.fit(x)
            # 组内平方和 加进列表
            wcss.append(k_means.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title("手肘法判断分几类")
        plt.xlabel("num")
        plt.ylabel("wcss")
        plt.show()
        # 手肘法判断k = 5
        k = 5
        k_means = KMeans(n_clusters=5, random_state=0, init="k-means++")
        y_pred = k_means.fit_predict(x)
        # 展示群组图像
        # plt.scatter(x[y_pred[0] == 0, 0], x[y_pred == 0, 1], s=100, c='red', label="0")
        # plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s=100, c='yellow', label="1")
        # plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s=100, c='blue', label="2")
        # plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s=100, c='green', label="3")
        # plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s=100, c='magenta', label="4")
        # plt.legend()
        # plt.show()


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.km = KMeansPlusPlusAAAA()

    def test_data_preprocessing_template(self):
        self.km.data_preprocessing_template()

    def test_k_means_plus_plus_function(self):
        self.km.k_means_plus_plus_function()

