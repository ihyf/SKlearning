# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 15:28
# @Author  : ihyf
# @File    : PreProcess.py
# @Software: PyCharm
# @ Desc :
import numpy as np
import matplotlib as plt
import pandas as pd
import unittest


class PreProcess(object):
    @staticmethod
    def preprocess_func_one():
        """
        :return:x_train, y_train, x_test, y_test
        """
        # 导入数据集
        data_set = pd.read_csv("Data.csv")
        x = data_set.iloc[:, 0:-1].values
        y = data_set.iloc[:, 3:].values
        # 填充缺失数据
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
        # 处理分类数据
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        label_encoder_x = LabelEncoder()
        x[:, 0] = label_encoder_x.fit_transform(x[:, 0])
        # 虚拟编码 categories=[0] 为处理数据集里的第0列
        # title 独热编码 第0列
        ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')
        x = ct.fit_transform(x)
        # 处理因变量 因变量不需要OneHotEncoder
        label_encoder_y = LabelEncoder()
        y = label_encoder_y.fit_transform(y)
        # 将数据分成数据集 测试集
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        # 特征缩放(归一化)
        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        # y 因变量 如果是分类问题就不用特征缩放， 如果是回归问题就需要
        return x_train, y_train, x_test, y_test


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocess = PreProcess()

    def test_preprocess_func_one(self):
        self.preprocess.preprocess_func_one()
