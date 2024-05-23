from __future__ import annotations

from numpy import ndarray

from ._base import BaseKernel


class LinearKernel(BaseKernel):

    def __init__(self):
        super().__init__()

    def __call__(self, xs: ndarray, ys: ndarray, *args, **kwargs):
        return self.__kernel(xs, ys, self.__kernel_func) #改成self._BaseKernel__kernel()

    def __kernel_func(self, x: ndarray, y: ndarray):
        return x.dot(y.T) #dot()矩阵乘法运算 一维的时候就是两个数字的乘积 y.T表示y的转置
