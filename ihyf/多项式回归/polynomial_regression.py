# -*- coding: utf-8 -*-
# @Time    : 2020/8/17 17:53
# @Author  : ihyf
# @File    : polynomial_regression.py
# @Software: PyCharm
# @ Desc : 多项式回归
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


class PolynomialRegression(object):
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