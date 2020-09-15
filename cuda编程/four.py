# -*- coding: utf-8 -*-
# @Time    : 2020/9/11 9:49
# @Author  : ihyf
# @File    : four.py
# @Software: PyCharm
# @ Desc :  未内存优化版本

from numba import cuda
import numpy as np
import math
from time import time


@cuda.jit
def matmul(a, b, c):
    """矩阵乘法 C = A*B"""
    row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    if row < c.shape[0] and col < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]):
            tmp += a[row, k] * b[k, col]
        c[row, col] = tmp


def main():
    # 矩阵初始化
    m = 6000
    n = 4800
    p = 4000
    a = np.random.random((m, n))
    b = np.random.random((n, p))

    start = time()
    a = cuda.to_device(a)
    b = cuda.to_device(b)
    c_gpu = cuda.device_array((m, p))

    threads_per_block = (16, 16)
    block_per_grid_x = int(math.ceil(a.shape[0]/threads_per_block[0]))
    block_per_grid_y = int(math.ceil(b.shape[1]/threads_per_block[1]))

    block_per_grid = (block_per_grid_x, block_per_grid_y)

    # 启动核函数
    matmul[block_per_grid, threads_per_block](a, b, c_gpu)

    # copy to host
    c = c_gpu.copy_to_host()
    cuda.synchronize()

    print("gpu matmul time :" + str(time() - start))

    # 使用cpu 计算
    start = time()
    c_cpu = np.empty((m, p), np.float)
    np.matmul(a, b, c_cpu)
    print("cpu matmul time :" + str(time() - start))

    # 对比结果

    if np.allclose(c_cpu, c):
        print("gpu result correct")


if __name__ == '__main__':
    main()


