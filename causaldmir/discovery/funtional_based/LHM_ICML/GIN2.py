import numpy as np
import pandas as pd
import HSIC2 as hsic
import FisherTest

#GIN with Fisher'Method
def FisherGIN(X,Z,data,alph=0.01):
    omega = getomega(data,X,Z)
    tdata= data[X]
    result = np.dot(omega, tdata.T)
    pvals=[]

    for i in Z:
        temp = np.array(data[i])
        pval=hsic.INtest(result.T,temp)
        pvals.append(pval)
    flag,fisher_pval=FisherTest.FisherTest(pvals,alph)

    return flag


#GIN condition test
#Input:
#X and Z are index, such as X=['x1','x2'],Z=['x3']
#data: DataFrame type
#alph: independent test alph
#output:
#Boolean: True(independent) or False(dependent)
def GIN(X,Z,data,alph=0.01):
    omega = getomega(data,X,Z)
    tdata= data[X]
    result = np.dot(omega, tdata.T)


    for i in Z:

        temp = np.array(data[i])
        flag =hsic.test(result.T,temp,alph)
        if not flag:
            return False
    return True


#esimate omega w: w satisfy w.Cov(X,Z)=0
def getomega(data,X,Z):
    cov_m =np.cov(data,rowvar=False)
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
    B = B[:,Zlist]
    A = B.T
    u,s,v = np.linalg.svd(A)
    lens = len(X)
    omega =v.T[:,lens-1]
    omegalen=len(omega)
    omega=omega.reshape(1,omegalen)
    return omega
