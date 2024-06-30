from __future__ import annotations

from enum import Enum

from numpy import exp, median, ndarray, shape, sqrt
from numpy.random import permutation
from pandas import DataFrame
from scipy.spatial.distance import cdist, pdist, squareform

from ._base import BaseKernel


class GaussianKernel(BaseKernel):
    class WidthStrategyEnum(Enum):
        manual = 1,
        median = 2,
        empirical_kci = 3,
        empirical_hsic = 4

    def __init__(self, width: float = 1.0, width_strategy: WidthStrategyEnum = WidthStrategyEnum.manual):
        super().__init__()

        self.width = width
        self.width_strategy = width_strategy

    def __call__(self, xs: ndarray, ys: ndarray, *args, **kwargs):
        return self.__kernel(xs, ys, self.__kernel_func)

    def __kernel(self, xs: ndarray, ys: ndarray, kernel_func, *args, **kwargs):
        if self.width_strategy != self.WidthStrategyEnum.manual:
            self.__update_kernel_width_by_width_strategy(self.width_strategy, xs)

        dict_key = hash(str((xs, ys, self.width)))  # add width to cache
        if self.cache_dict.__contains__(dict_key):
            res = self.cache_dict[dict_key]
        else:
            res = kernel_func(xs, ys)
            self.cache_dict[dict_key] = res

        return res

    def __kernel_func(self, x: ndarray, y: ndarray):
        if y is None:
            sq_dists = squareform(pdist(x, 'sqeuclidean')) #计算矩阵每行与其他行之间的距离，然后把距离转化成方阵
        else:
            assert (shape(x)[1] == shape(y)[1]) #如果x和y的列数一样
            sq_dists = cdist(x, y, 'sqeuclidean')#计算两个集合向量之间的距离
        k = exp(-0.5 * sq_dists * self.width)
        return k

    def __update_kernel_width_by_width_strategy(self, strategy, data):
        if type(data) == ndarray:
            pass
        elif type(data) == DataFrame:
            data = data.values
        else:
            raise Exception("'data' must be ndarray or DataFrame!")

        if strategy == self.width_strategy.median:
            width = self.__cal_kernel_width_by_median(data)
        elif strategy == self.width_strategy.empirical_kci:
            width = self.__cal_kernel_width_by_empirical_kci(data)
        elif strategy == self.width_strategy.empirical_hsic:
            width = self.__cal_kernel_width_by_empirical_hsic(data)
        else:
            raise NotImplementedError("width_strategy '{}' is not implemented!".format(strategy))

        self.width = width

    @staticmethod
    def __cal_kernel_width_by_median(data_x):
        n = shape(data_x)[0]
        if n > 1000:
            data_x = data_x[permutation(n)[:1000], :]
        dists = squareform(pdist(data_x, 'euclidean'))
        median_dist = median(dists[dists > 0])
        width = sqrt(2.) * median_dist
        theta = 1.0 / (width ** 2)
        return theta

    @staticmethod
    def __cal_kernel_width_by_empirical_kci(data_x):
        n = shape(data_x)[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        theta = 1.0 / (width ** 2)
        return theta / shape(data_x)[1]

    @staticmethod
    def __cal_kernel_width_by_empirical_hsic(data_x):
        n = shape(data_x)[0]
        if n < 200:
            width = 0.8
        elif n < 1200:
            width = 0.5
        else:
            width = 0.3
        theta = 1.0 / (width ** 2)
        return theta * data_x.shape[1]
