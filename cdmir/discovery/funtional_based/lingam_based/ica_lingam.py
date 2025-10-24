from __future__ import annotations

import itertools

import numpy as np
import scipy.stats
from numpy import ndarray
from pandas import DataFrame
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression


def find_all_P(n):
    """Find all possible permutation matrices of a matrix.

    Parameters
    ----------
    n : int
        The dimension of the matrix to be permuted.

    Returns
    -------
    p_matrix : array, shape [n!, n, n]
        The set of all permutation matrices.
    """
    temp = np.zeros(n, dtype=np.int8)
    temp[0] = 1
    temp = set(itertools.permutations(temp, len(temp)))
    return np.asarray(list(set(itertools.permutations(temp, len(temp)))))


def find_W_wave(W):
    """Find the optimal W_wave matrix.\n
    .. math:: \\mathrm{by \\quad minimize} \\quad \\sum_i \\quad \\frac{1}{\\mathbf{\\tilde{W}}_{ii} \\quad}

    Parameters
    ----------
    W : array, shape (n_features, n_features)
        Unmixing matrix.

    Returns
    -------
    W_wave : array, shape [n_features, n_features]
        New unmixing matrix obtained by permuting to minimize diagonal elements.
    """
    P_list = find_all_P(W.shape[0])
    score = [sum(sum(np.eye(W.shape[0]) * abs(1 / P.dot(W)))) for P in P_list]
    return P_list[np.argmin(score)].dot(W)


def find_B_wave(B_hat):
    """Find the optimal B_wave matrix.\n
    .. math:: \\mathrm{by \\quad minimize} \\quad \\sum_{i\\leq j} \\quad \\tilde{\\mathbf{B}}_{ij} \\quad ^2.

    Parameters
    ----------
    B_hat : array, shape (n_features, n_features)
        The causality effect matrix is obtained by calculating B = I - W.

    Returns
    -------
    B_wave : array, shape [n_features, n_features]
        The new causality effect matrix is obtained by minimizing the upper triangular elements.
    """
    P_list = find_all_P(B_hat.shape[0])
    score = [sum(np.diag((P.dot(B_hat) * P.dot(B_hat)).dot(np.tri(B_hat.shape[0])))) for P in P_list]
    return P_list[np.argmin(score)].dot(B_hat)


def g(x):
    """The hyperbolic tangent function is often abbreviated as "tanh"(A nonlinear function).

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    tanh(x) : array-like
        Output data.
    """
    return np.tanh(x)


def fk(Z, W, k):
    """Compute the estimation function F.\n
    .. math :: \\mathbf{F(X,Q)}=yy^{T}-\\mathbf{\\mathit{I}}+yg^{T}(y)-g(y)y^{T}

    Parameters
    ----------
    Z : array, shape (n_features, n_samples)
        The whitened data matrix.
    W : array, shape (n_features, n_features)
        An orthogonal matrix in the whitened space is used to obtain the independent components in the whitened space.
    k : int
        Sample indices in the data matrix.

    Returns
    -------
    fk : array, shape [n_features^2,]
        The estimated function F values for the k-th sample transformed into a vector.
    """
    dim = Z.shape[0]
    sample = Z.shape[1]

    y = W.T.dot(Z[:, k])
    ygy = y.dot(g(y).T)

    fkout = y.dot(y.T) - np.eye(dim) + ygy - ygy.T

    # Vectorize
    fkout = fkout.flatten('F')
    return fkout


def diff_g(x):
    """The derivative of the hyperbolic tangent function.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    derivative_tanh(x) : array-like
        Output data.
    """
    return 1 - np.tanh(x) ** 2


def computeQblock(i, j, Z, W):
    """Compute the block of the Jacobian matrix.

    Parameters
    ----------
    i : int
        The row indices of the Jacobian matrix block.
    j : int
        The column indices of the Jacobian matrix block.
    Z : array, shape (n_features, n_samples)
        The whitened data matrix.
    W : array, shape (n_features, n_features)
        An orthogonal matrix in the whitened space is used to obtain the independent components in the whitened space.

    Returns
    -------
    Qij : array shape [n_features, n_features]
        The block of the Jacobian matrix at the i-th row and j-th column.
    """
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
    """Compute the Jacobian matrix.

    Parameters
    ----------
    Z : array, shape (n_features, n_samples)
        The whitened data matrix.
    W : array, shape (n_features, n_features)
        An orthogonal matrix in the whitened space is used to obtain the independent components in the whitened space.

    Returns
    -------
    Q : array, shape [n_features^2, n_features^2]
        The Jacobian matrix of F with respect to the transpose of Q.
    """
    dim = Z.shape[0]
    Q = np.zeros((dim ** 2, dim ** 2))

    for i in range(dim):
        for j in range(dim):
            Q[dim * i: dim + dim * i, dim * j: dim + dim * j] = computeQblock(i, j, Z, W)

    return Q


def acovW(Z, W):
    """Compute the asymptotic variance of the vectorized orthogonal matrix Q.\n
    .. math :: \\mathrm{acov\\{vec(\\mathbf{Q})\\}}=\\frac1n \\quad \\mathbf{J}^{-1} E \\quad \\mathrm{[vec\{\\mathbf{F(X,Q)}\} vec\{\\mathbf{F(X,Q)}\}^{T}]} \\mathbf{J}^{-T}
    Parameters
    ----------
    Z : array, shape (n_features, n_samples)
        The whitened data matrix.
    W : array, shape (n_features, n_features)
        An orthogonal matrix in the whitened space is used to obtain the independent components in the whitened space.

    Returns
    -------
    acov(vec(Q)) : array, shape [n_features^2, n_features^2]
        The asymptotic covariance matrix of the vectorized orthogonal matrix Q.
    """
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
    """Compute the asymptotic variance of the vectorized unmixing matrix W.\n
    .. math:: \\mathrm{acov\\{vec(\\mathbf{W})\\}}=\\mathbf{(\\mathit{V^T \\otimes I \\quad})} \\mathrm{acov\\{vec(\\mathbf{Q}^T)\\}} \\mathbf{(\\mathit{V^T \\otimes I \\quad})}\\mathrm{^T}

    Parameters
    ----------
    X : array, shape (n_features, n_samples)
        The data matrix.
    B : array, shape (n_features, n_features)
        The unmixing matrix.

    Returns
    -------
    acov(vec(W)): array, shape [n_features^2, n_features^2]
        The asymptotic covariance matrix of the vectorized unmixing matrix W.
    """
    dim = X.shape[0]
    sample = X.shape[1]

    # Find a whitening matrix in the same manner as FastICA
    C = np.cov(X.T, rowvar=0, bias=0)
    E, D, E_t = np.linalg.svd(C, full_matrices=True)
    V = E.dot(np.diag(D ** (-1 / 2))).dot(E_t)  # whitening matrix

    # Compute Asymptotic variance-covariance matrix of vec(W)
    # Map to the whitened space and compute the asymptotic variance.
    Z = V.dot(X)
    Acovtilde = acovW(Z, np.linalg.pinv(V.T).dot(B))

    # Compute Asymptotic variance-covariance matrix of vec(B)
    # Map back to the original space.
    G = np.kron(np.eye(dim), V.T)
    Acov = G.dot(Acovtilde).dot(G.T)

    return Acov


def calcwald(X, W):
    """Compute the Wald statistic and its corresponding p-value.\n
    .. math :: \\mathrm{Wald}=\\frac{\\tilde{w}_{ij} \\quad ^2 \\quad}{\\mathrm{avar}(\\tilde{w}_{ij} \\quad)}

    Parameters
    ----------
    X : array, shape (n_features, n_samples)
        The data matrix.
    W : array, shape (n_features, n_features)
        The unmixing matrix.

    Returns
    -------
    P : array, shape [n_features, n_features]
        The probability of having a value of the Wald statistic larger than or equal to the empirical one computed from data.
    """
    dims = X.shape[0]
    avar = np.diag(acovB(X, W.T))  # asymptotic variance
    # Obtain the variances on the diagonal.
    avar = avar[0:dims ** 2]  # reshape to correspond W
    avar = avar.reshape((dims, dims)).T
    wald = (W ** 2) / avar
    # The Wald statistic asymptotically approximates to a chi-square variate with one degree of freedom.
    P = np.ones(dims) - scipy.stats.chi2.cdf(wald, 1)

    return P


def wald_prune(X, W, B, alpha):
    """Prune based on the p-value of the Wald statistic.

    Parameters
    ----------
    X : array, shape (n_features, n_samples)
        The data matrix.
    W : array, shape (n_features, n_features)
        The unmixing matrix.
    B : array, shape (n_features, n_features)
        The causality effect matrix.
    alpha : float
        Significance level threshold.

    Returns
    -------
    Bpruned : array, shape (n_features, n_features)
        The causality effect matrix after pruning.
    """
    P = calcwald(X, W)
    Bpruned = B
    Bpruned[P > alpha] = 0

    return Bpruned


class ICA_LINGAM(object):
    """Implementation of ICA-based LiNGAM Algorithm [1]_

    References
    ----------
    .. [1] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. J. Kerminen.
       A linear non-gaussian acyclic model for causal discovery.
       Journal of Machine Learning Research, 7:2003-2030, 2006.
    """

    def __init__(self, wald_alpha: float=0.05):
        """Initialize the parameters of ICA-LiNGAM.

        Parameters
        ----------
        wald_alpha : float
            The significance level threshold for the Wald statistic.
        """
        self.__coef = None
        self.__causal_graph = None
        self.wald_alpha = wald_alpha

    def get_coef(self):
        """Return the connection coefficients of the causal graph.

        Returns
        -------
        coef : array, shape (n_features, n_features)
            The connection coefficients of the causal graph.
        """
        if self.__coef is None:
            raise Exception("please fit some data with this algorithm before get coef!")
        return self.__coef

    def get_causal_graph(self):
        """Return the causal graph.

        Returns
        -------
        causal_graph : array, shape (n_features, n_features)
            The causal graph among features.
        """
        if self.__causal_graph is None:
            raise Exception("please fit some data with this algorithm before get causal graph!")
        return self.__causal_graph

    def fit(self, data: ndarray | DataFrame):
        """Fit the ICA-LiNGAM model.

        Parameters
        ----------
        data : array | DataFrame shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        coef : array shape (n_features, n_features)
            The connection coefficients of the causal graph fitted from the data.
        causal_graph : array, shape (n_features, n_features)
            The causal graph fitted from the data.
        """
        if type(data) == DataFrame:
            data = data.values

        # ICA
        A = FastICA(n_components=data.shape[1]).fit(data).mixing_
        W = np.linalg.pinv(A)

        # find W_wave
        W_wave = find_W_wave(W)

        # find_W_wave_slash
        # To make the diagonal of a matrix all ones
        W_wave_slash = np.linalg.pinv(np.diag(np.diag(W_wave))).dot(W_wave)

        # find B_hat
        B_hat = np.eye(W_wave_slash.shape[0]) - W_wave_slash

        # find B_wave
        B_wave = find_B_wave(B_hat)

        # prune upper triangle
        # Allow only nodes with a smaller causal order to have causal effects on nodes with a larger causal order.
        for x in range(B_wave.shape[0]):
            for y in range(x, B_wave.shape[1]):
                B_wave[x, y] = 0

        # wald test
        # prune
        B_Prune = wald_prune(data.T, W_wave, B_wave, 0.5)

        # Linear Regression
        # Fit the linear causal effects among the variables.
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