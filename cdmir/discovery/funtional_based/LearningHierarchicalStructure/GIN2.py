import numpy as np
import pandas as pd
import lingam.hsic as hsic

import cdmir.discovery.funtional_based.LearningHierarchicalStructure.indTest.independence as ID


# GIN by fast HSIC
# X=['X1','X2']
# Z=['X3']
# data.type=Pandas.DataFrame
def GIN(X, Z, data, alpha=0.05):
    omega = getomega(data, X, Z)
    tdata = data[X]
    result = np.dot(omega, tdata.T)
    for i in Z:
        temp = np.array(data[i])

        pvalue = hsic.hsic_test_gamma(result.T, temp, alpha)

        if pvalue > alpha:
            flag = True
        else:
            flag = False

        if not flag:
            return False

    return True


# mthod 1: estimating mutual information by k nearest neighbors (density estimation)
# mthod 2: estimating mutual information by sklearn package
def GIN_MI(X, Z, data, method='1'):
    omega = getomega(data, X, Z)
    tdata = data[X]
    result = np.dot(omega, tdata.T)
    MIS = 0
    for i in Z:

        temp = np.array(data[i])
        if method == '1':
            mi = ID.independent(result.T, temp)
        else:
            mi = ID.independent11(result.T, temp)
        MIS += mi
    MIS = MIS / len(Z)

    return MIS


def getomega(data, X, Z):
    cov_m = np.cov(data, rowvar=False)
    col = list(data.columns)
    Xlist = []
    Zlist = []
    for i in X:
        t = col.index(i)
        Xlist.append(t)
    for i in Z:
        t = col.index(i)
        Zlist.append(t)
    B = cov_m[Xlist]
    B = B[:, Zlist]
    A = B.T
    u, s, v = np.linalg.svd(A)
    lens = len(X)
    omega = v.T[:, lens - 1]
    omegalen = len(omega)
    omega = omega.reshape(1, omegalen)

    return omega