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
        dataset = pd.read_csv('Position_Salaries.csv')
        x = dataset.iloc[:, 1:2].values
        y = dataset.iloc[:, 2].values
        return x, y

    def polynomial_regression(self):
        # 用线性回归模型拟合
        from sklearn.linear_model import LinearRegression
        x, y = self.data_preprocessing_template()
        lin_regress = LinearRegression()
        lin_regress.fit(x, y)
        # 用多项式回归拟合
        from sklearn.preprocessing import PolynomialFeatures
        poly_regress = PolynomialFeatures(degree=4)
        x_poly = poly_regress.fit_transform(x)
        lin_regress_2 = LinearRegression()
        lin_regress_2.fit(x_poly, y)

        # 线性回归拟合结果展示
        plt.scatter(x, y, c='r')
        plt.plot(x, lin_regress.predict(x), c='b')
        plt.title('Truth or Bluff (Linear Regression)')
        plt.xlabel('Position Level')
        plt.ylabel('Salary')
        plt.show()
        # 多项式回归拟合结果展示
        x_grid = np.arange(min(x), max(x), 0.1)
        x_grid = x_grid.reshape(len(x_grid), 1)
        plt.scatter(x, y, c='r')
        plt.plot(x, lin_regress.predict(x), c='b')
        plt.plot(x_grid, lin_regress_2.predict(poly_regress.fit_transform(x_grid)), c='b')
        plt.title('Truth or Bluff (Polynomial Regression)')
        plt.xlabel('Position Level')
        plt.ylabel('Salary')
        plt.show()
        # 预测新人的薪水
        new_person_level = 6.5
        new_person_level = np.array(new_person_level)
        new_person_level = new_person_level.reshape(1, 1)
        # 两种模型预测的新人薪水
        print(lin_regress.predict(new_person_level)[-1])
        new_person_salary = lin_regress_2.predict(poly_regress.fit_transform(new_person_level))
        print(new_person_salary[-1])


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.pr = PolynomialRegression()

    def test_data_preprocessing_template(self):
        self.pr.data_preprocessing_template()

    def test_polynomial_regression(self):
        self.pr.polynomial_regression()

