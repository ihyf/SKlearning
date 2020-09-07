# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 17:22
# @Author  : ihyf
# @File    : three.py
# @Software: PyCharm
# @ Desc : 使用多流
from numba import cuda


def main():
    stream = cuda.stream()

if __name__ == "__main__":
    main()