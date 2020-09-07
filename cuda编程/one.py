# -*- coding: utf-8 -*-
# @Time    : 2020/9/4 16:22
# @Author  : ihyf
# @File    : one.py
# @Software: PyCharm
# @ Desc :
from numba import cuda


def cpu_print():
    print("print by cpu.")


@cuda.jit
def gpu_print():
    # gpu 核函数
    print(cuda.threadIdx.x)
    print(cuda.blockDim.x)
    print(cuda.gridDim.x)
    print("print by gpu")


def main():
    gpu_print[1, 2]()
    cuda.synchronize()
    cpu_print()


if __name__ == "__main__":
    main()
