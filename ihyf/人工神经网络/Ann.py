# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 14:16
# @Author  : ihyf
# @File    : Ann.py
# @Software: PyCharm
# @ Desc :

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


class Ann(object):
    @staticmethod
    def data_preprocessing_template():
        # 导入csv数据
        data_set = pd.read_csv("Churn_Modelling.csv")
        x = data_set.iloc[:, 3:13].values
        y = data_set.iloc[:, 13].values
        # 类别数据处理
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder_x_1 = LabelEncoder()
        labelencoder_x_2 = LabelEncoder()
        x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
        x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

        one_hot_encoder = OneHotEncoder()
        x = one_hot_encoder.fit_transform(x).toarray()
        x = x[:, 1:]

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test

    def ann_function(self):
        x_train, x_test, y_train, y_test = self.data_preprocessing_template()
        # 初始化人工神经网络
        classifier = Sequential()

        # 添加隐藏层
        classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
        classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        # optimizer随机梯度下降所用的算法 adam 损失函数用crossentropy 由于结果为二元 所以用binary_crossentropy metrics 为模型评判指标 accuracy 准确度
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(x_train, y_train, batch_size=10, epochs=100)


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ann = Ann()

    def test_data_preprocessing_template(self):
        x_train, x_test, y_train, y_test = self.ann.data_preprocessing_template()

    def test_ann_function(self):
        self.ann.ann_function()
