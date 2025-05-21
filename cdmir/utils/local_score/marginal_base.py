from __future__ import annotations

from math import floor
from typing import Iterable

import numpy as np
from numpy import asmatrix, corrcoef, ix_, log, mat, ndarray, shape
from numpy.linalg import inv
from pandas import DataFrame

from ._base import BaseLocalScoreFunction
from .score_utils import eigdec, gpr_multi_new, kernel, minimize


class GeneralMarginalScore(BaseLocalScoreFunction):

    def __init__(self, data: ndarray | DataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.data = asmatrix(data)

    def _score_function(self, i: int, parent_i: Iterable[int]):
        parent_i = list(parent_i)
        T = self.data.shape[0]
        X = self.data[:, parent_i]

        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
        width = width * 2.5  # kernel width
        theta = 1 / (width ** 2)
        H = np.eye(T) - np.ones((T, T)) / T
        Kx, _ = kernel(X, X, (theta, 1))
        Kx = H * Kx * H

        Thresh = 1E-5
        eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, floor(T / 4)]), evals_only=False)  # /2
        IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        eig_Kx = eig_Kx[IIx]
        eix = eix[:, IIx]

        if len(parent_i):
            PA = self.data[:, parent_i]

            widthPA = np.empty((PA.shape[1], 1))
            # set the kernel for PA
            for m in range(PA.shape[1]):
                G = np.sum((np.multiply(PA[:, m], PA[:, m])), axis=1)
                Q = np.tile(G, (1, T))
                R = np.tile(G.T, (T, 1))
                dists = Q + R - 2 * PA[:, m] * PA[:, m].T
                dists = dists - np.tril(dists)
                dists = np.reshape(dists, (T ** 2, 1))
                widthPA[m] = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
            widthPA = widthPA * 2.5  # kernel width

            covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']], dtype=object)
            logtheta0 = np.vstack([np.log(widthPA), 0, np.log(np.sqrt(0.1))])
            logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                             2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

            nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                        2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                        nargout=2)
        else:
            covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']], dtype=object)
            PA = np.zeros((T, 1))
            logtheta0 = np.asmatrix([100, 0, np.log(np.sqrt(0.1))]).T
            logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                             2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

            nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                        2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                        nargout=2)
        score = -nlml  # log-likelihood
        return score

    def __call__(self, i: int, parent_i: Iterable[int], *args, **kwargs):
        return self._score(i, parent_i, self._score_function)


class MultiMarginalScore(BaseLocalScoreFunction):

    def __init__(self, data: ndarray | DataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.data = asmatrix(data)
        if not kwargs.__contains__('d_label'):
            self.d_label = {}
            for i in range(self.data.shape[1]):
                self.d_label[i] = i
        else:
            self.d_label = kwargs["d_label"]

    def _score_function(self, i: int, parent_i: Iterable[int]):
        # calculate the local score by negative marginal likelihood
        # based on a regression model in RKHS
        # for variables with multi-variate dimensions
        #
        # INPUT:
        # Data: (sample, features)
        # i: current index
        # PAi: parent indexes
        #
        # OUTPUT:
        # local score
        parent_i = list(parent_i)
        T = self.data.shape[0]
        X = self.data[:, self.d_label[i]]
        dX = X.shape[1]

        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T ** 2, 1))
        widthX = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
        widthX = widthX * 2.5  # kernel width
        theta = 1 / (widthX ** 2)
        H = np.eye(T) - np.ones((T, T)) / T
        Kx, _ = kernel(X, X, (theta, 1))
        Kx = H * Kx * H

        Thresh = 1E-5
        eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, floor(T / 4)]), evals_only=False)  # /2
        IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        eig_Kx = eig_Kx[IIx]
        eix = eix[:, IIx]

        if len(parent_i):
            widthPA_all = np.empty((1, 0))
            # set the kernel for PA
            PA_all = np.empty((self.data.shape[0], 0))
            for m in range(len(parent_i)):
                PA = self.data[:, self.d_label[parent_i[m]]]
                PA_all = np.hstack([PA_all, PA])
                G = np.sum((np.multiply(PA, PA)), axis=1)
                Q = np.tile(G, (1, T))
                R = np.tile(G.T, (T, 1))
                dists = Q + R - 2 * PA * PA.T
                dists = dists - np.tril(dists)
                dists = np.reshape(dists, (T ** 2, 1))
                widthPA = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
                widthPA_all = np.hstack(
                    [widthPA_all, widthPA * np.ones((1, np.size(self.d_label[parent_i[m]])))])
            widthPA_all = widthPA_all * 2.5  # kernel width
            covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']], dtype=object)
            logtheta0 = np.vstack([np.log(widthPA_all.T), 0, np.log(np.sqrt(0.1))])
            logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA_all,
                                             2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

            nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA_all,
                                        2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                        nargout=2)
        else:
            covfunc = np.asarray(['covSum', ['covSEard', 'covNoise']], dtype=object)
            PA = np.zeros((T, 1))
            logtheta0 = np.asmatrix([100, 0, np.log(np.sqrt(0.1))]).T
            logtheta, fvals, iter = minimize(logtheta0, 'gpr_multi_new', -300, covfunc, PA,
                                             2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]))

            nlml, dnlml = gpr_multi_new(logtheta, covfunc, PA,
                                        2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
                                        nargout=2)
        score = -nlml  # log-likelihood
        return score

    def __call__(self, i: int, parent_i: Iterable[int], *args, **kwargs):
        return self._score(i, parent_i, self._score_function)