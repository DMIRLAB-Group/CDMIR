"""
Additive Noise Model.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import scale


def rbf_dot2(p1, p2, deg):
    if p1.ndim == 1:
        p1 = p1[:, np.newaxis]
        p2 = p2[:, np.newaxis]

    size1 = p1.shape
    size2 = p2.shape

    G = np.sum(p1 * p1, axis=1)[:, np.newaxis]
    H = np.sum(p2 * p2, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))
    H = Q + R - 2.0 * np.dot(p1, p2.T)
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def rbf_dot(X, deg):
    '''
    Calculate the rbf kernel matrix.
    Set kernel size to median distance between points, if no kernel specified.

    Parameters
    ----------
    X: input data (n_sample,n_features)
    deg: kernel parameters for X (-1 means set kernel size to median distance between points)

    Returns
    ---------
    H: rbf kernel matrix
    '''
    
    if X.ndim == 1:
        X = X[:, np.newaxis]
    m = X.shape[0]
    G = np.sum(X * X, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def FastHsicTestGamma(X, Y, sig=[-1, -1], maxpnt=200):
    '''
    Calculate the HSIC statistics between two variables using the rbf kernel.
    This is a fast implementation of the HSIC statistic using the rbf kernel.

    Parameters
    ----------
    X: input data (n,1)or(n,)
    Y: output data (n,1)or(n,)
    sig: kernel parameters for X and Y,default = [-1,1]
    maxpnt: maximum number of points to use,default = 200

    Returns
    ---------
    testStat: HSIC statistic
    '''

    m = X.shape[0]
    if m > maxpnt:
        indx = np.floor(np.r_[0:m:float(m - 1) / (maxpnt - 1)]).astype(int)
        #       indx = np.r_[0:maxpnt]
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
        m = Xm.shape[0]
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    H = np.eye(m) - 1.0 / m * np.ones((m, m))

    K = rbf_dot(Xm, sig[0])
    L = rbf_dot(Ym, sig[1])

    Kc = np.dot(H, np.dot(K, H))
    Lc = np.dot(H, np.dot(L, H))

    testStat = (1.0 / m) * (Kc.T * Lc).sum()
    if ~np.isfinite(testStat):
        testStat = 0

    return testStat


def normalized_hsic(x, y):
    '''
    Calculate the standardized HSIC statistics

    Parameters
    ----------
    x: input data (n,1)or(n,)
    y: output data (n,1)or(n,)

    Returns
    ---------
    h: normalized HSIC statistic
    '''

    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    h = FastHsicTestGamma(x, y, maxpnt=2000)

    return h


class ANM(object):
    '''
    Python implementation of additive noise model-based causal discovery.
    References
    ----------
    [1] Hoyer, Patrik O., et al. "Nonlinear causal discovery with additive noise models." NIPS. Vol. 21. 2008.
    '''
    def __init__(self):
        '''
        Construct the ANM model.
        '''
        super(ANM, self).__init__()

    def cause_or_effect(self, x, y, **kwargs):
        '''
        Fit a GP model in two directions and test the independence between the input and estimated noise

        Parameters
        ---------
        x: input data (n,)or(n,1)
        y: output data (n,)or(n,1)

        Returns
        ---------
        nonindepscore_forward: HSIC statistic in the x->y direction
        nonindepscore_backward: HSIC statistic in the y->x direction
        '''
        # Standardize the input data        
        x = scale(x).reshape((-1, 1))
        y = scale(y).reshape((-1, 1))
        # calculate the x->y score
        nonindepscore_forward = self.anm_score(x, y)
        # calculate the y->x score
        nonindepscore_backward = self.anm_score(y, x)

        return nonindepscore_forward, nonindepscore_backward

    def anm_score(self, x, y):
        '''
        Calculate the causal direction score of the ANM model

        Parameters
        ---------
        x: input data (n,1)
        y: output data (n,1)

        Returns
        ---------
        nonindepscore: HSIC statistic in the x->y direction
        '''
        # fit Gaussian process, including hyperparameter optimization
        gp = GaussianProcessRegressor().fit(x, y)
        y_predict = gp.predict(x)
        # calculate the normalized HSIC statistic between the estimated noise and the input data
        nonindepscore = normalized_hsic(y_predict - y, x)

        return nonindepscore
