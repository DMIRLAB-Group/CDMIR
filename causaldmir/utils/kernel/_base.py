from __future__ import annotations

from numpy import eye, ndarray, shape
from numpy.linalg import pinv


class BaseKernel(object):

    def __init__(self, *args, **kwargs):
        self.cache_dict = dict()

    def __call__(self, xs: ndarray, ys: ndarray, *args, **kwargs):
        return self.__kernel(xs, ys, self.__kernel_func)

    def __kernel(self, xs: ndarray, ys: ndarray, kernel_func, *args, **kwargs):
        dict_key = hash(str((xs, ys)))
        if self.cache_dict.__contains__(dict_key):
            res = self.cache_dict[dict_key]
        else:
            res = kernel_func(xs, ys)
            self.cache_dict[dict_key] = res

        return res

    def __kernel_func(self, data_x: ndarray, data_y: ndarray):
        raise NotImplementedError()

    @staticmethod
    def center_kernel_matrix(K: ndarray):
        n = shape(K)[0]
        K_colsums = K.sum(axis=0)
        K_allsum = K_colsums.sum()
        return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n ** 2)

    @staticmethod
    def center_kernel_matrix_regression(K: ndarray, Kz: ndarray, epsilon: float):
        """
        Centers the kernel matrix via a centering matrix R=I-Kz(Kz+\epsilonI)^{-1} and returns RKR
        """
        n = shape(K)[0]
        Rz = epsilon * pinv(Kz + epsilon * eye(n))
        return Rz.dot(K.dot(Rz)), Rz
