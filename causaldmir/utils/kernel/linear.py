from __future__ import annotations

from numpy import ndarray

from ._base import BaseKernel


class LinearKernel(BaseKernel):

    def __init__(self):
        super().__init__()

    def __call__(self, xs: ndarray, ys: ndarray, *args, **kwargs):
        return self.__kernel(xs, ys, self.__kernel_func)

    def __kernel_func(self, x: ndarray, y: ndarray):
        return x.dot(y.T)
