from __future__ import annotations

from abc import ABC
from typing import List, Tuple

import numpy as np
from numpy import isnan, ndarray, shape, sqrt, trace
from numpy.linalg import eigh, eigvalsh
from pandas import DataFrame
from scipy.stats import gamma, zscore
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from ..kernel import BaseKernel, GaussianKernel

from ._base import BaseConditionalIndependenceTest


class KCI(BaseConditionalIndependenceTest):
    """
    K. Zhang, J. Peters, D. Janzing, and B. Schoelkopf,  "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    """

    def __init__(self, data: ndarray | DataFrame, kernel_x: BaseKernel = GaussianKernel(),
                 kernel_y: BaseKernel = GaussianKernel(), kernel_z: BaseKernel | None = None,
                 approximate_mode: bool = True, *args, **kwargs):
        super().__init__(data)

        self.__ukci = self.__UnconditionalKCI(data, kernel_x, kernel_y, approximate_mode, **kwargs)
        self.__ckci = self.__ConditionalKCI(data, kernel_x, kernel_y, kernel_z, approximate_mode, **kwargs)

    def __call__(self, xs: int | str | List[int | str] | ndarray,
                        ys: int | str | List[int | str] | ndarray,
                        zs: int | str | List[int | str] | ndarray | None = None, *args, **kwargs) -> Tuple[float, float | ndarray | None]:
        return self._compute_p_value(xs, ys, zs, self.__compute_p_value_without_condition, self.__compute_p_value_with_condition)

    def __compute_p_value_with_condition(self, x_ids: List[int], y_ids: List[int], z_ids: List[int]) -> Tuple[float, float | ndarray | None]:
        return self.__ckci(x_ids, y_ids, z_ids)

    def __compute_p_value_without_condition(self, x_ids: List[int], y_ids: List[int]) -> Tuple[float, float | ndarray | None]:
        return self.__ukci(x_ids, y_ids)

    class __UnconditionalKCI(BaseConditionalIndependenceTest, ABC):

        def __init__(self, data: ndarray | DataFrame, kernel_x: BaseKernel, kernel_y: BaseKernel,
                     approximate_mode: bool = True, **kwargs):
            super().__init__(data)
            self.kernel_x = kernel_x
            self.kernel_y = kernel_y
            self.approximate_mode = approximate_mode
            if kwargs.__contains__("null_distribution_sample_size"):
                self.null_distribution_sample_size = kwargs["null_distribution_sample_size"]
            else:
                self.null_distribution_sample_size = 1000  # default value
            if kwargs.__contains__("lambda_product_threshold"):
                self.lambda_product_threshold = kwargs["lambda_product_threshold"]
            else:
                self.lambda_product_threshold = 1e-6  # default value

        def __call__(self, xs: int | str | List[int | str] | ndarray,
                     ys: int | str | List[int | str] | ndarray,
                     zs: int | str | List[int | str] | ndarray | None = None, *args, **kwargs) -> Tuple[
            float, float | ndarray | None]:
            return self._compute_p_value(xs, ys, zs, self.__compute_p_value_without_condition, None)

        def __compute_p_value_without_condition(self, x_ids: List[int], y_ids: List[int]) \
                -> Tuple[float, float | ndarray | None]:
            Kx, Ky = self.kernel_matrix(self._data, x_ids, y_ids)
            test_stat = self.HSIC_V_test_statistic_from_kernel_matrix(Kx, Ky)

            if self.approximate_mode:
                k_appr, theta_appr = self.get_kappa_from_kernel_matrix(Kx, Ky)
                pvalue = 1 - gamma.cdf(test_stat, k_appr, 0, theta_appr)
            else:
                null_dstr = self.null_sample_spectral_from_centralized_kernel_matrix(Kx, Ky)
                pvalue = sum(null_dstr.squeeze() > test_stat) / float(self.null_distribution_sample_size)
            return pvalue, test_stat

        def kernel_matrix(self, data: ndarray, x_ids: List[int], y_ids: List[int]) -> Tuple[ndarray, ndarray]:

            # We set 'ddof=1' to conform to the normalization way in the original Matlab implementation in
            # http://people.tuebingen.mpg.de/kzhang/KCI-test.zip
            data_x = zscore(data[:, x_ids], ddof=1, axis=0)
            data_x[isnan(data_x)] = 0.  # in case some dim of data_x is constant
            data_y = zscore(data[:, y_ids], ddof=1, axis=0)
            data_y[isnan(data_y)] = 0.

            Kx = self.kernel_x(data_x, data_x)
            Ky = self.kernel_y(data_y, data_y)
            Kx = self.kernel_x.center_kernel_matrix(Kx)
            Ky = self.kernel_y.center_kernel_matrix(Ky)

            return Kx, Ky

        @staticmethod
        def HSIC_V_test_statistic_from_kernel_matrix(Kx, Ky):
            return np.sum(Kx * Ky)

        @staticmethod
        def get_kappa_from_kernel_matrix(Kx, Ky):
            T = shape(Kx)[0]
            mean_appr = trace(Kx) * trace(Ky) / T
            var_appr = 2 * np.sum(Kx ** 2) * np.sum(
                Ky ** 2) / T / T  # same as np.sum(Kx * Kx.T) ..., here Kx is symmetric
            k_appr = mean_appr ** 2 / var_appr
            theta_appr = var_appr / mean_appr
            return k_appr, theta_appr

        def null_sample_spectral_from_centralized_kernel_matrix(self, Kxc, Kyc):
            T = Kxc.shape[0]
            if T > 1000:
                num_eig = np.int(np.floor(T / 2))
            else:
                num_eig = T
            lambdax = eigvalsh(Kxc)
            lambday = eigvalsh(Kyc)
            lambdax = -np.sort(-lambdax)
            lambday = -np.sort(-lambday)
            lambdax = lambdax[0:num_eig]
            lambday = lambday[0:num_eig]
            lambda_prod = np.dot(lambdax.reshape(num_eig, 1), lambday.reshape(1, num_eig)).reshape((num_eig ** 2, 1))
            lambda_prod = lambda_prod[lambda_prod > lambda_prod.max() * self.lambda_product_threshold]
            f_rand = np.random.chisquare(1, (lambda_prod.shape[0], self.null_distribution_sample_size))
            null_dstr = lambda_prod.T.dot(f_rand) / T
            return null_dstr

    class __ConditionalKCI(BaseConditionalIndependenceTest, ABC):

        def __init__(self, data: ndarray | DataFrame, kernel_x: BaseKernel, kernel_y: BaseKernel,
                     kernel_z: BaseKernel | None = None, approximate_mode: bool = True, **kwargs):
            super().__init__(data)

            self.kernel_x = kernel_x
            self.kernel_y = kernel_y
            self.kernel_z = kernel_z
            self.use_gaussian_process = True if self.kernel_z is None else False
            self.approximate_mode = approximate_mode
            if kwargs.__contains__("null_distribution_sample_size"):
                self.null_distribution_sample_size = kwargs["null_distribution_sample_size"]
            else:
                self.null_distribution_sample_size = 5000  # default value
            if kwargs.__contains__("lambda_product_threshold"):
                self.lambda_product_threshold = kwargs["lambda_product_threshold"]
            else:
                self.lambda_product_threshold = 1e-5  # default value
            if kwargs.__contains__("epsilon_x"):
                self.epsilon_x = kwargs["epsilon_x"]
            else:
                self.epsilon_x = 1e-3  # default value
            if kwargs.__contains__("epsilon_y"):
                self.epsilon_y = kwargs["epsilon_y"]
            else:
                self.epsilon_y = 1e-3  # default value

        def __call__(self, xs: int | str | List[int | str] | ndarray,
                     ys: int | str | List[int | str] | ndarray,
                     zs: int | str | List[int | str] | ndarray | None = None, *args, **kwargs) -> Tuple[
            float, float | ndarray | None]:
            return self._compute_p_value(xs, ys, zs, None, self.__compute_p_value_with_condition)

        def __compute_p_value_with_condition(self, x_ids: List[int], y_ids: List[int], z_ids: List[int]) \
                -> Tuple[float, float | ndarray | None]:
            Kx, Ky, Kzx, Kzy = self.kernel_matrix(self._data, x_ids, y_ids, z_ids)
            test_stat, KxR, KyR = self.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
            uu_prod, size_u = self.get_uuprod(KxR, KyR)
            if self.approximate_mode:
                k_appr, theta_appr = self.get_kappa(uu_prod)
                pvalue = 1 - gamma.cdf(test_stat, k_appr, 0, theta_appr)
            else:
                null_samples = self.null_sample_spectral(uu_prod, size_u, Kx.shape[0])
                pvalue = sum(null_samples > test_stat) / float(self.null_distribution_sample_size)
            return pvalue, test_stat

        def kernel_matrix(self, data: ndarray, x_ids: List[int], y_ids: List[int], z_ids: List[int]):

            def kernel_matrix_for_z(Kx, data_z):
                # learning the kernel width of Kz using Gaussian process
                n, Dz = data_z.shape
                if type(self.kernel_x) == GaussianKernel:
                    width_z = sqrt(1.0 / (self.kernel_x.width * shape(data_x)[1]))
                else:
                    width_z = 1.0
                # Instantiate a Gaussian Process model for x
                wx, vx = eigh(0.5 * (Kx + Kx.T))
                topkx = int(np.min((400, np.floor(n / 4))))
                idx = np.argsort(-wx)
                wx = wx[idx]
                vx = vx[:, idx]
                wx = wx[0:topkx]
                vx = vx[:, 0:topkx]
                vx = vx[:, wx > wx.max() * self.lambda_product_threshold]
                wx = wx[wx > wx.max() * self.lambda_product_threshold]
                vx = 2 * sqrt(n) * vx.dot(np.diag(np.sqrt(wx))) / sqrt(wx[0])
                kernel_x = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(width_z * np.ones(Dz), (1e-2, 1e2)) + \
                          WhiteKernel(0.1, (1e-10, 1e+1))
                gpx = GaussianProcessRegressor(kernel=kernel_x)
                # fit Gaussian process, including hyperparameter optimization
                gpx.fit(data_z, vx)

                # construct Gaussian kernels according to learned hyperparameters
                Kzx = gpx.kernel_.k1(data_z, data_z)
                epsilon = np.exp(gpx.kernel_.theta[-1])
                return Kzx, epsilon

            # normalize the data
            data_x = zscore(data[:, x_ids], ddof=1, axis=0)
            data_x[isnan(data_x)] = 0.

            data_y = zscore(data[:, y_ids], ddof=1, axis=0)
            data_y[isnan(data_y)] = 0.

            data_z = zscore(data[:, z_ids], ddof=1, axis=0)
            data_z[isnan(data_z)] = 0.
            # We set 'ddof=1' to conform to the normalization way in the original Matlab implementation in
            # http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

            # concatenate x and z
            data_x = np.concatenate((data_x, 0.5 * data_z), axis=1)

            Kx = self.kernel_x(data_x, data_x)
            Ky = self.kernel_y(data_y, data_y)

            # centering kernel matrix
            Kx = self.kernel_x.center_kernel_matrix(Kx)
            Ky = self.kernel_y.center_kernel_matrix(Ky)

            if self.use_gaussian_process:
                Kzx, epsilon_x = kernel_matrix_for_z(Kx, data_z)
                Kzy, epsilon_y = kernel_matrix_for_z(Ky, data_z)
                self.epsilon_x = epsilon_x
                self.epsilon_y = epsilon_y
            else:
                Kzx = self.kernel_z(data_z, data_z)
                Kzx = self.kernel_z.center_kernel_matrix(Kzx)
                Kzy = Kzx

            return Kx, Ky, Kzx, Kzy

        def KCI_V_statistic(self, Kx, Ky, Kzx, Kzy):
            KxR, Rzx = self.kernel_x.center_kernel_matrix_regression(Kx, Kzx, self.epsilon_x)
            if self.epsilon_x != self.epsilon_y or self.use_gaussian_process:
                KyR, _ = self.kernel_y.center_kernel_matrix_regression(Ky, Kzy, self.epsilon_y)
            else:
                # assert np.all(Kzx == Kzy), 'Kzx and Kzy are the same'
                KyR = Rzx.dot(Ky.dot(Rzx))
            Vstat = np.sum(KxR * KyR)
            return Vstat, KxR, KyR

        def get_uuprod(self, Kx, Ky):

            wx, vx = eigh(0.5 * (Kx + Kx.T))
            wy, vy = eigh(0.5 * (Ky + Ky.T))
            idx = np.argsort(-wx)
            idy = np.argsort(-wy)
            wx = wx[idx]
            vx = vx[:, idx]
            wy = wy[idy]
            vy = vy[:, idy]
            vx = vx[:, wx > np.max(wx) * self.lambda_product_threshold]
            wx = wx[wx > np.max(wx) * self.lambda_product_threshold]
            vy = vy[:, wy > np.max(wy) * self.lambda_product_threshold]
            wy = wy[wy > np.max(wy) * self.lambda_product_threshold]
            vx = vx.dot(np.diag(np.sqrt(wx)))
            vy = vy.dot(np.diag(np.sqrt(wy)))

            # calculate their product
            T = Kx.shape[0]
            num_eigx = vx.shape[1]
            num_eigy = vy.shape[1]
            size_u = num_eigx * num_eigy
            uu = np.zeros((T, size_u))
            for i in range(0, num_eigx):
                for j in range(0, num_eigy):
                    uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

            if size_u > T:
                uu_prod = uu.dot(uu.T)
            else:
                uu_prod = uu.T.dot(uu)

            return uu_prod, size_u

        def get_kappa(self, uu_prod):

            mean_appr = np.trace(uu_prod)
            var_appr = 2 * np.trace(uu_prod.dot(uu_prod))
            k_appr = mean_appr ** 2 / var_appr
            theta_appr = var_appr / mean_appr
            return k_appr, theta_appr

        def null_sample_spectral(self, uu_prod, size_u, T):

            eig_uu = eigvalsh(uu_prod)
            eig_uu = -np.sort(-eig_uu)
            eig_uu = eig_uu[0:np.min((T, size_u))]
            eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.lambda_product_threshold]

            f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.null_distribution_sample_size))
            null_dstr = eig_uu.T.dot(f_rand)
            return null_dstr