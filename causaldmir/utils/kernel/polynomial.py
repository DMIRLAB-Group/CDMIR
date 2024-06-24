from __future__ import annotations

from numpy import ndarray

from ._base import BaseKernel


class PolynomialKernel(BaseKernel):

    def __init__(self, degree: int = 2, const: float = 1.0):
        super().__init__()
        self.degree = degree
        self.const = const

    def __kernel_func(self, x: ndarray, y: ndarray):
        return pow(self.const + x.dot(y.T), self.degree)#const是加减上的调整，degree是控制多项式的次数

    def __kernel(self, xs: ndarray, ys: ndarray, *args, **kwargs):
        dict_key = hash(str((xs, ys, self.degree, self.const)))  # add 'degree' and 'const' to cache
        if self.cache_dict.__contains__(dict_key):
            res = self.cache_dict[dict_key]
        else:
            res = self.__kernel(xs, ys) #Plan B:改成 res = self.__kernel_func(xs,ys)
            self.cache_dict[dict_key] = res

        return res

    def __call__(self, xs: ndarray, ys: ndarray, *args, **kwargs):
        return self._BaseKernel__kernel(xs, ys, self.__kernel_func) #Plan A：把self.__kernel改成return self._BaseKernel__kernel(xs, ys, self.__kernel_func)
