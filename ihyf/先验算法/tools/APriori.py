# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 15:13
# @Author  : ihyf
# @File    : APriori.py
# @Software: PyCharm
# @ Desc : 先验算法
import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .apyori import apriori


class APriori(object):
    @staticmethod
    def data_preprocessing_template():
        data_set = pd.read_csv("./Market_Basket_Optimisation.csv", header=None)
        transactions = []
        for i in range(0, 7501):
            transactions.append([str(data_set.values[i, j]) for j in range(0, 20)])
        return transactions

    def APriori_func(self):
        transactions = self.data_preprocessing_template()
        rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
        results = list(rules)
        myResults = [list(x) for x in results]
        print(myResults)


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ap = APriori()

    def test_APriori_func(self):
        self.ap.APriori_func()
