from __future__ import annotations

from math import floor
from typing import Iterable

import numpy as np
from numpy import ndarray, shape, asmatrix
from numpy.linalg import inv
from pandas import DataFrame

from ._base import BaseLocalScoreFunction
from .score_utils import kernel, pdinv


class GeneralCVScore(BaseLocalScoreFunction):

    def __init__(self, data: ndarray | DataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.data = asmatrix(data)
        if not kwargs.__contains__('lambda_value'):
            self.lambda_value = 0.01
        else:
            self.lambda_value = kwargs["lambda_value"]
        if not kwargs.__contains__('k_fold'):
            self.k_fold = 10
        else:
            self.k_fold = kwargs["k_fold"]

    def _score_function(self, i: int, parent_i: Iterable[int]):
        # calculate the local score
        # using k-fold cross-validated log likelihood as the score
        # based on a regression model in RKHS
        #
        # INPUT:
        # Data: (sample, features)
        # i: current index
        # PAi: parent indexes
        # OUTPUT:
        # local score

        parent_i = list(parent_i)

        T = shape(self.data)[0]
        X = self.data[:, i]
        n0 = floor(T / self.k_fold)
        gamma = 0.01

        if len(parent_i):
            PA = self.data[:, parent_i]

            # set the kernel for X
            GX = np.sum(np.multiply(X, X), axis=1)
            Q = np.tile(GX, (1, T))
            R = np.tile(GX.T, (T, 1))
            dists = Q + R - 2 * X * X.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
            width = width * 2
            theta = 1 / (width ** 2)

            Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
            H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
            Kx = H0 * Kx * H0  # kernel matrix for X

            # eig_Kx, eix = eigdec((Kx + Kx.T)/2, np.min([400, math.floor(T/2)]), evals_only=False)   # /2
            # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
            # eig_Kx = eig_Kx[IIx]
            # eix = eix[:, IIx]
            # mx = len(IIx)

            # set the kernel for PA
            Kpa = np.matlib.ones((T, T))

            for m in range(PA.shape[1]):
                G = np.sum((np.multiply(PA[:, m], PA[:, m])), axis=1)
                Q = np.tile(G, (1, T))
                R = np.tile(G.T, (T, 1))
                dists = Q + R - 2 * PA[:, m] * PA[:, m].T
                dists = dists - np.tril(dists)
                dists = np.reshape(dists, (T ** 2, 1))
                width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
                width = width * 2
                theta = 1 / (width ** 2)
                Kpa = np.multiply(Kpa, kernel(PA[:, m], PA[:, m], (theta, 1))[0])

            H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
            Kpa = H0 * Kpa * H0  # kernel matrix for PA

            CV = 0
            for kk in range(self.k_fold):
                if (kk == 0):
                    Kx_te = Kx[0:n0, 0:n0]
                    Kx_tr = Kx[n0: T, n0: T]
                    Kx_tr_te = Kx[n0: T, 0: n0]
                    # Kpa_te = Kpa[0:n0, 0: n0]
                    Kpa_tr = Kpa[n0: T, n0: T]
                    Kpa_tr_te = Kpa[n0: T, 0: n0]
                    nv = n0  # sample size of validated data
                if (kk == self.k_fold - 1):
                    Kx_te = Kx[kk * n0:T, kk * n0: T]
                    Kx_tr = Kx[0:kk * n0, 0: kk * n0]
                    Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                    # Kpa_te = Kpa[kk * n0:T, kk * n0: T]
                    Kpa_tr = Kpa[0: kk * n0, 0: kk * n0]
                    Kpa_tr_te = Kpa[0:kk * n0, kk * n0: T]
                    nv = T - kk * n0
                if (kk < self.k_fold - 1 and kk > 0):
                    Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                    Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                      np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                    Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                         np.arange(kk * n0, (kk + 1) * n0))]
                    Kpa_te = Kpa[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                    Kpa_tr = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                        np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                    Kpa_tr_te = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                           np.arange(kk * n0, (kk + 1) * n0))]
                    nv = n0

                n1 = T - nv
                tmp1 = pdinv(Kpa_tr + n1 * self.lambda_value * np.matlib.eye(n1))
                tmp2 = tmp1 * Kx_tr * tmp1
                tmp3 = tmp1 * pdinv(np.matlib.eye(n1) + n1 * self.lambda_value ** 2 / gamma * tmp2) * tmp1
                A = (Kx_te + Kpa_tr_te.T * tmp2 * Kpa_tr_te - 2 * Kx_tr_te.T * tmp1 * Kpa_tr_te
                     - n1 * self.lambda_value ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr_te
                     - n1 * self.lambda_value ** 2 / gamma * Kpa_tr_te.T * tmp1 * Kx_tr * tmp3 * Kx_tr * tmp1 * Kpa_tr_te
                     + 2 * n1 * self.lambda_value ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr * tmp1 * Kpa_tr_te) / gamma

                B = n1 * self.lambda_value ** 2 / gamma * tmp2 + np.matlib.eye(n1)
                L = np.linalg.cholesky(B)
                C = np.sum(np.log(np.diag(L)))
                #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
                CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

            CV = CV / self.k_fold
        else:
            # set the kernel for X
            GX = np.sum(np.multiply(X, X), axis=1)
            Q = np.tile(GX, (1, T))
            R = np.tile(GX.T, (T, 1))
            dists = Q + R - 2 * X * X.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
            width = width * 2
            theta = 1 / (width ** 2)

            Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
            H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
            Kx = H0 * Kx * H0  # kernel matrix for X

            # eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 2)]), evals_only=False)  # /2
            # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
            # mx = len(IIx)

            CV = 0
            for kk in range(self.k_fold):
                if kk == 0:
                    Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                    Kx_tr = Kx[(kk + 1) * n0:T, (kk + 1) * n0: T]
                    Kx_tr_te = Kx[(kk + 1) * n0:T, kk * n0: (kk + 1) * n0]
                    nv = n0
                if (kk == self.k_fold - 1):
                    Kx_te = Kx[kk * n0: T, kk * n0: T]
                    Kx_tr = Kx[0: kk * n0, 0: kk * n0]
                    Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                    nv = T - kk * n0
                if (kk < self.k_fold - 1 and kk > 0):
                    Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                    Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                      np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                    Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                         np.arange(kk * n0, (kk + 1) * n0))]
                    nv = n0

                n1 = T - nv
                A = (Kx_te - 1 / (gamma * n1) * Kx_tr_te.T * pdinv(
                    np.matlib.eye(n1) + 1 / (gamma * n1) * Kx_tr) * Kx_tr_te) / gamma
                B = 1 / (gamma * n1) * Kx_tr + np.matlib.eye(n1)
                L = np.linalg.cholesky(B)
                C = np.sum(np.log(np.diag(L)))

                # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
                CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

            CV = CV / self.k_fold

        score = -CV  # cross-validated likelihood
        return score

    def __call__(self, i: int, parent_i: Iterable[int], *args, **kwargs):
        return self._score(i, parent_i, self._score_function)


class MultiCVScore(BaseLocalScoreFunction):

    def __init__(self, data: ndarray | DataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.data = asmatrix(data)
        if not kwargs.__contains__('lambda_value'):
            self.lambda_value = 0.01
        else:
            self.lambda_value = kwargs["lambda_value"]
        if not kwargs.__contains__('k_fold'):
            self.k_fold = 10
        else:
            self.k_fold = kwargs["k_fold"]
        if not kwargs.__contains__('d_label'):
            self.d_label = {}
            for i in range(self.data.shape[1]):
                self.d_label[i] = i
        else:
            self.d_label = kwargs["d_label"]

    def _score_function(self, i: int, parent_i: Iterable[int]):
        # calculate the local score
        # using negative k-fold cross-validated log likelihood as the score
        # based on a regression model in RKHS
        # for variables with multi-variate dimensions
        #
        # INPUT:
        # Data: (sample, features)
        # i: current index
        # PAi: parent indexes
        # parameters:
        #               kfold: k-fold cross validation
        #               lambda: regularization parameter
        #               dlabel: for variables with multi-dimensions,
        #                                indicate which dimensions belong to the i-th variable.
        #
        # OUTPUT:
        # local score
        parent_i = list(parent_i)
        T = self.data.shape[0]
        X = self.data[:, self.d_label[i]]
        n0 = floor(T / self.k_fold)
        gamma = 0.01

        if len(parent_i):
            # set the kernel for X
            GX = np.sum(np.multiply(X, X), axis=1)
            Q = np.tile(GX, (1, T))
            R = np.tile(GX.T, (T, 1))
            dists = Q + R - 2 * X * X.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
            width = width * 3  ###
            theta = 1 / (width ** 2 * X.shape[1])  #

            Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
            H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
            Kx = H0 * Kx * H0  # kernel matrix for X

            # set the kernel for PA
            Kpa = np.matlib.ones((T, T))

            for m in range(len(parent_i)):
                PA = self.data[:, self.d_label[parent_i[m]]]
                G = np.sum((np.multiply(PA, PA)), axis=1)
                Q = np.tile(G, (1, T))
                R = np.tile(G.T, (T, 1))
                dists = Q + R - 2 * PA * PA.T
                dists = dists - np.tril(dists)
                dists = np.reshape(dists, (T ** 2, 1))
                width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
                width = width * 3  ###
                theta = 1 / (width ** 2 * PA.shape[1])
                Kpa = np.multiply(Kpa, kernel(PA, PA, (theta, 1))[0])

            H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
            Kpa = H0 * Kpa * H0  # kernel matrix for PA

            CV = 0
            for kk in range(self.k_fold):
                if (kk == 0):
                    Kx_te = Kx[0:n0, 0:n0]
                    Kx_tr = Kx[n0: T, n0: T]
                    Kx_tr_te = Kx[n0: T, 0: n0]
                    Kpa_te = Kpa[0:n0, 0: n0]
                    Kpa_tr = Kpa[n0: T, n0: T]
                    Kpa_tr_te = Kpa[n0: T, 0: n0]
                    nv = n0  # sample size of validated data
                if (kk == self.k_fold - 1):
                    Kx_te = Kx[kk * n0:T, kk * n0: T]
                    Kx_tr = Kx[0:kk * n0, 0: kk * n0]
                    Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                    Kpa_te = Kpa[kk * n0:T, kk * n0: T]
                    Kpa_tr = Kpa[0: kk * n0, 0: kk * n0]
                    Kpa_tr_te = Kpa[0:kk * n0, kk * n0: T]
                    nv = T - kk * n0
                if (kk < self.k_fold - 1 and kk > 0):
                    Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                    Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                      np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                    Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                         np.arange(kk * n0, (kk + 1) * n0))]
                    Kpa_te = Kpa[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                    Kpa_tr = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                        np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                    Kpa_tr_te = Kpa[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                           np.arange(kk * n0, (kk + 1) * n0))]
                    nv = n0

                n1 = T - nv
                tmp1 = pdinv(Kpa_tr + n1 * self.lambda_value * np.matlib.eye(n1))
                tmp2 = tmp1 * Kx_tr * tmp1
                tmp3 = tmp1 * pdinv(np.matlib.eye(n1) + n1 * self.lambda_value ** 2 / gamma * tmp2) * tmp1
                A = (Kx_te + Kpa_tr_te.T * tmp2 * Kpa_tr_te - 2 * Kx_tr_te.T * tmp1 * Kpa_tr_te
                     - n1 * self.lambda_value ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr_te
                     - n1 * self.lambda_value ** 2 / gamma * Kpa_tr_te.T * tmp1 * Kx_tr * tmp3 * Kx_tr * tmp1 * Kpa_tr_te
                     + 2 * n1 * self.lambda_value ** 2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr * tmp1 * Kpa_tr_te) / gamma

                B = n1 * self.lambda_value ** 2 / gamma * tmp2 + np.matlib.eye(n1)
                L = np.linalg.cholesky(B)
                C = np.sum(np.log(np.diag(L)))
                #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
                CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

            CV = CV / self.k_fold
        else:
            # set the kernel for X
            GX = np.sum(np.multiply(X, X), axis=1)
            Q = np.tile(GX, (1, T))
            R = np.tile(GX.T, (T, 1))
            dists = Q + R - 2 * X * X.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T ** 2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])  # median value
            width = width * 3  ###
            theta = 1 / (width ** 2 * X.shape[1])  #

            Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
            H0 = np.matlib.eye(T) - np.matlib.ones((T, T)) / (T)  # for centering of the data in feature space
            Kx = H0 * Kx * H0  # kernel matrix for X

            CV = 0
            for kk in range(self.k_fold):
                if (kk == 0):
                    Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                    Kx_tr = Kx[(kk + 1) * n0:T, (kk + 1) * n0: T]
                    Kx_tr_te = Kx[(kk + 1) * n0:T, kk * n0: (kk + 1) * n0]
                    nv = n0
                if (kk == self.k_fold - 1):
                    Kx_te = Kx[kk * n0: T, kk * n0: T]
                    Kx_tr = Kx[0: kk * n0, 0: kk * n0]
                    Kx_tr_te = Kx[0:kk * n0, kk * n0: T]
                    nv = T - kk * n0
                if (kk < self.k_fold - 1 and kk > 0):
                    Kx_te = Kx[kk * n0: (kk + 1) * n0, kk * n0: (kk + 1) * n0]
                    Kx_tr = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                      np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]))]
                    Kx_tr_te = Kx[np.ix_(np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]),
                                         np.arange(kk * n0, (kk + 1) * n0))]
                    nv = n0

                n1 = T - nv
                A = (Kx_te - 1 / (gamma * n1) * Kx_tr_te.T * pdinv(
                    np.matlib.eye(n1) + 1 / (gamma * n1) * Kx_tr) * Kx_tr_te) / gamma
                B = 1 / (gamma * n1) * Kx_tr + np.matlib.eye(n1)
                L = np.linalg.cholesky(B)
                C = np.sum(np.log(np.diag(L)))

                # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
                CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

            CV = CV / self.k_fold

        score = -CV  # cross-validated likelihood
        return score

    def __call__(self, i: int, parent_i: Iterable[int], *args, **kwargs):
        return self._score(i, parent_i, self._score_function)

