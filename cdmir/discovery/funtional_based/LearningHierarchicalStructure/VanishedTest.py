import math
import random

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from sklearn.utils import resample


# test the Tetrad vanished by wishart method
# input:
#   x:
#   y:
#   z:
#   w:
#   alpha:
# output:
#   True(Satisfying constraints) of False
def vanishes(x, y, z, w, alpha=0.01):
    p1 = tetradPValue(x, y, z, w)
    if p1 > alpha:
        return True
    else:
        return False


def vanishes_Pval(x, y, z, w):
    p1 = tetradPValue(x, y, z, w)
    return p1


def tetradPValue(i, j, k, l):
    pval = wishartEvalTetradDifference(i, j, k, l)
    return pval


def wishartEvalTetradDifference(i, j, k, l):
    pval = 0
    TAUijkl = 0

    a = np.cov(i, k)[0][1]
    b = np.cov(j, l)[0][1]
    c = np.cov(i, l)[0][1]
    d = np.cov(j, k)[0][1]

    TAUijkl = a * b - c * d
    SD = wishartTestTetradDifference(i, j, k, l)

    ratio = TAUijkl / SD
    if (ratio > 0.0):
        ratio = -ratio

    pval = 2.0 * norm.cdf(ratio)
    return pval


def wishartTestTetradDifference(a0, a1, a2, a3):
    data = pd.DataFrame(np.array([a0, a1, a2, a3]).T, columns=['a0', 'a1', 'a2', 'a3'])
    M_cov = data.cov()
    p1 = np.cov(a0, a0)[0][1] * np.cov(a3, a3)[0][1] - np.cov(a0, a3)[0][1] * np.cov(a0, a3)[0][1]
    p2 = np.cov(a1, a1)[0][1] * np.cov(a2, a2)[0][1] - np.cov(a1, a2)[0][1] * np.cov(a1, a2)[0][1]
    n = len(a0)
    product3 = (n + 1) / ((n - 1) * (n - 2)) * p1 * p2
    determinant = np.linalg.det(M_cov)
    var = (product3 - determinant / (n - 2))

    return math.sqrt(abs(var))
