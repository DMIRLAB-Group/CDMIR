from __future__ import annotations

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.decomposition import FastICA
import itertools
import scipy.stats
from sklearn.linear_model import LinearRegression


def find_all_P(n):
    temp = np.zeros(n, dtype=np.int8)
    temp[0] = 1
    temp = set(itertools.permutations(temp, len(temp)))
    return np.asarray(list(set(itertools.permutations(temp, len(temp)))))


def find_W_wave(W):
    P_list = find_all_P(W.shape[0])
    score = [sum(sum(np.eye(W.shape[0]) * abs(1 / P.dot(W)))) for P in P_list]
    return P_list[np.argmin(score)].dot(W)


def find_B_wave(B_hat):
    P_list = find_all_P(B_hat.shape[0])
    score = [sum(np.diag((P.dot(B_hat) * P.dot(B_hat)).dot(np.tri(B_hat.shape[0])))) for P in P_list]
    return P_list[np.argmin(score)].dot(B_hat)


def g(x):
    return np.tanh(x)


def fk(Z, W, k):
    dim = Z.shape[0]
    sample = Z.shape[1]

    y = W.T.dot(Z[:, k])
    ygy = y.dot(g(y).T)

    fkout = y.dot(y.T) - np.eye(dim) + ygy - ygy.T

    fkout = fkout.flatten('F')
    return fkout


def diff_g(x):
    return 1 - np.tanh(x) ** 2


def computeQblock(i, j, Z, W):
    dim = Z.shape[0]
    S = W.T.dot(Z)

    Qblockij = np.zeros((dim, dim))

    if i == j:
        for k in range(dim):
            if k == i:
                Qblockij[k, :] = 2 * W[:, i].T
            else:
                Qblockij[k, :] = (1 - np.mean(S[k, :] * g(S[k, :])) + np.mean(diff_g(S[i, :]))) * W[:, k].T
    else:
        for k in range(dim):
            if k == j:
                Qblockij[k, :] = (1 - np.mean(diff_g(S[j, :])) + np.mean(S[i, :] * g(S[i, :]))) * W[:, i].T

    return Qblockij


def computeQ(Z, W):
    dim = Z.shape[0]
    Q = np.zeros((dim ** 2, dim ** 2))

    for i in range(dim):
        for j in range(dim):
            Q[dim * i: dim + dim * i, dim * j: dim + dim * j] = computeQblock(i, j, Z, W)

    return Q


def acovW(Z, W):
    dim = Z.shape[0]
    sample = Z.shape[1]

    # Compute sample covariance matrix of f, covf
    covf = np.zeros((dim ** 2, dim ** 2, sample))

    for i in range(sample):
        fi = fk(Z, W, i)
        covf[:, :, i] = np.outer(fi, fi)

    covf = np.mean(covf, axis=2)

    # Compute sample Q
    Q = computeQ(Z, W)

    # Compute Acov
    invQ = np.linalg.pinv(Q)
    Acov = invQ.dot(covf).dot(invQ.T) / sample

    return Acov


def acovB(X, B):
    dim = X.shape[0]
    sample = X.shape[1]

    # Find a whitening matrix in the same manner as FastICA
    C = np.cov(X.T, rowvar=0, bias=0)
    E, D, E_t = np.linalg.svd(C, full_matrices=True)
    V = E.dot(np.diag(D ** (-1 / 2))).dot(E_t)  # whitening matrix

    # Compute Asymptotic variance-covariance matrix of vec(W)
    Z = V.dot(X)
    Acovtilde = acovW(Z, np.linalg.pinv(V.T).dot(B))

    # Compute Asymptotic variance-covariance matrix of vec(B)
    G = np.kron(np.eye(dim), V.T)
    Acov = G.dot(Acovtilde).dot(G.T)

    return Acov


def calcwald(X, W):
    dims = X.shape[0]
    avar = np.diag(acovB(X, W.T))  # asymptotic variance
    avar = avar[0:dims ** 2]  # reshape to correspond W
    avar = avar.reshape((dims, dims)).T
    wald = (W ** 2) / avar
    P = np.ones(dims) - scipy.stats.chi2.cdf(wald, 1)

    return P


def wald_prune(X, W, B, alpha):
    P = calcwald(X, W)
    Bpruned = B
    Bpruned[P > alpha] = 0

    return Bpruned


class ICA_LINGAM(object):

    def __init__(self, wald_alpha: float=0.05):
        self.__coef = None
        self.__causal_graph = None
        self.wald_alpha = wald_alpha

    def get_coef(self):
        if self.__coef is None:
            raise Exception("please fit some data with this algorithm before get coef!")
        return self.__coef

    def get_causal_graph(self):
        if self.__causal_graph is None:
            raise Exception("please fit some data with this algorithm before get causal graph!")
        return self.__causal_graph

    def fit(self, data: ndarray | DataFrame):
        if type(data) == DataFrame:
            data = data.values

        # ICA
        A = FastICA(n_components=data.shape[1]).fit(data).mixing_
        W = np.linalg.pinv(A)

        # find W_wave
        W_wave = find_W_wave(W)

        # find_W_wave_slash
        W_wave_slash = np.linalg.pinv(np.diag(np.diag(W_wave))).dot(W_wave)

        # find B_hat
        B_hat = np.eye(W_wave_slash.shape[0]) - W_wave_slash

        # find B_wave
        B_wave = find_B_wave(B_hat)

        # prune upper triangle
        for x in range(B_wave.shape[0]):
            for y in range(x, B_wave.shape[1]):
                B_wave[x, y] = 0

        # wald test
        B_Prune = wald_prune(data.T, W_wave, B_wave, 0.5)

        # Linear Regression
        reg_list = {i: B_wave[i, :] != 0 for i in range(B_Prune.shape[0])}
        for i in range(B_Prune.shape[0]):
            if np.sum(reg_list[i]) != 0:
                y_reg = data[:, i]
                X_reg = data.T[reg_list[i]].T
                clf = LinearRegression()
                clf.fit(y=y_reg.reshape(data.shape[0], -1), X=X_reg.reshape(data.shape[0], -1))
                B_Prune[i, reg_list[i]] = clf.coef_

        # save parameters
        self.__coef = B_Prune
        self.__causal_graph = (B_Prune != 0) * 1


