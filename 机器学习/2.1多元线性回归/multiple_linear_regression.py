# -*- coding: utf-8 -*-
# @Time    : 2020/8/17 15:47
# @Author  : ihyf
# @File    : multiple_linear_regression.py
# @Software: PyCharm
# @ Desc :
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


class MultipleLinearRegression(object):
    @staticmethod
    def data_preprocessing_template():
        # Importing the dataset
        dataset = pd.read_csv('50_Startups.csv')
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 4].values

        label_encoder = LabelEncoder()
        x[:, 3] = label_encoder.fit_transform(x[:, 3])
        ct = ColumnTransformer([("Country", OneHotEncoder(), [3])], remainder='passthrough')
        x = ct.fit_transform(x)
        # 虚拟变量陷阱
        x = x[:, 1:]

        # Splitting the dataset into the Training set and Test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test

    def multiple_linear_regression(self):
        x_train, x_test, y_train, y_test = self.data_preprocessing_template()
        from sklearn.linear_model import LinearRegression
        # 定义回归器
        regressor = LinearRegression()
        # 拟合回归器
        regressor.fit(x_train, y_train)
        # 用回归器预测 x_test
        y_pred = regressor.predict(x_test)
        # 反向淘汰自变量
        import statsmodels.api as sm
        x_train = np.append(arr=np.ones((40, 1)), values=x_train, axis=1)
        # 最佳自变量选择
        x_opt = x_train[:, [0, 1, 2, 3, 4, 5]]
        x_opt = x_opt.astype(np.float64)
        regressor_ols = sm.OLS(endog=y_train, exog=x_opt).fit()
        print(regressor_ols.summary())
        # 根据P值 大于0.1 去除相应自变量 1，2，4
        x_opt = x_train[:, [0, 3, 5]]
        x_opt = x_opt.astype(np.float64)
        regressor_ols = sm.OLS(endog=y_train, exog=x_opt).fit()
        print(regressor_ols.summary())


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mr = MultipleLinearRegression()

    def test_data_preprocessing_template(self):
        x_train, x_test, y_train, y_test = self.mr.data_preprocessing_template()

    def test_multiple_linear_regression(self):
        self.mr.multiple_linear_regression()
