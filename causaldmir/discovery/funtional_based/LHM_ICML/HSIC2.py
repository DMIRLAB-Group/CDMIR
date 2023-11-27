#API for HSIC test
from kerpy.GaussianKernel import GaussianKernel
from HSICTestObject import HSICTestObject
from numpy import shape,savetxt,loadtxt,transpose,shape,reshape,concatenate
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
from independence_testing.HSICBlockTestObject import HSICBlockTestObject
import numpy as np
import pandas as pd

#method 1:HSIC test and return boolean
#x and y:
#    data type: numpy.array()
#    dim: samples * 1
#alph: test alph
def test(x,y,alph=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
##    kernelX=GaussianKernel()
##    kernelY=GaussianKernel()

    kernelY = GaussianKernel(float(0.1))
    kernelX=GaussianKernel(float(0.1))
    num_samples = lens
    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=False, kernelY_use_median=False,
                                          rff=True, num_rfx=30, num_rfy=30, num_nullsims=1000)
    pvalue = myspectralobject.compute_pvalue(x, y)

    #print(pvalue)
    if pvalue >alph:
        return True
    else:
        return False

#method 2
def test2(x,y,alph=0.08):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
##    kernelX=GaussianKernel()
##    kernelY=GaussianKernel()
    kernelY = GaussianKernel(float(0.45))
    kernelX=GaussianKernel(float(0.45))
    num_samples = lens
    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                    kernelX_use_median=False, kernelY_use_median=False,
                                    blocksize=80, nullvarmethod='permutation')

    pvalue = myblockobject.compute_pvalue(x, y)

    return pvalue
    #print(pvalue)
##    if pvalue >alph:
##        return True
##    else:
##        return False

# HSIC test by return hsic pval
def INtest(x,y,alph=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()
##    kernelY = GaussianKernel(float(0.4))
##    kernelX=GaussianKernel(float(0.4))
    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=True, kernelY_use_median=True,
                                          rff=True, num_rfx=30, num_rfy=30, num_nullsims=1000)
    pvalue = myspectralobject.compute_pvalue(x, y)

    return pvalue

def INtest2(x,y,alph=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()
    num_samples = lens

    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                    kernelX_use_median=True, kernelY_use_median=True,
                                    blocksize=200, nullvarmethod='permutation')

    pvalue = myblockobject.compute_pvalue(x, y)

    return pvalue
