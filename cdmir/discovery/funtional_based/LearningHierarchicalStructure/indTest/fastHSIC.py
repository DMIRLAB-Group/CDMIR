import sys

sys.path.append("./indTest")
import numpy as np
import pandas as pd

from .HSICTestObject import HSICTestObject
from .HSICBlockTestObject import HSICBlockTestObject
from .HSICSpectralTestObject import HSICSpectralTestObject
from numpy import concatenate, loadtxt, reshape, savetxt, shape, shape, transpose
from kerpy.kerpy.GaussianKernel import GaussianKernel


def test(alph=0.05):
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    lens = len(x)
    x = x.reshape(lens, 1)
    y = y.reshape(lens, 1)

    kernelY = GaussianKernel(float(0.15))
    kernelX = GaussianKernel(float(0.15))

    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                              kernelX_use_median=False, kernelY_use_median=False,
                                              rff=True, num_rfx=20, num_rfy=20, num_nullsims=1000)

    pvalue = myspectralobject.compute_pvalue(x, y)

    if pvalue > alph:
        return True
    else:
        return False


def test2(alph=0.08):
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    lens = len(x)
    x = x.reshape(lens, 1)
    y = y.reshape(lens, 1)
    kernelX = GaussianKernel()
    kernelY = GaussianKernel()

    num_samples = lens

    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                        kernelX_use_median=False, kernelY_use_median=False,
                                        blocksize=80, nullvarmethod='permutation')

    pvalue = myblockobject.compute_pvalue(x, y)

    if pvalue > alph:
        return True
    else:
        return False


def INtest(x, y, alph=0.01):
    lens = len(x)
    x = x.reshape(lens, 1)
    y = y.reshape(lens, 1)
    kernelX = GaussianKernel()
    kernelY = GaussianKernel()

    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                              kernelX_use_median=True, kernelY_use_median=True,
                                              rff=True, num_rfx=20, num_rfy=20, num_nullsims=1000)

    pvalue = myspectralobject.compute_pvalue(x, y)

    return pvalue


def INtest2(x, y, alph=0.01):
    lens = len(x)
    x = x.reshape(lens, 1)
    y = y.reshape(lens, 1)
    kernelX = GaussianKernel()
    kernelY = GaussianKernel()
    num_samples = lens

    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                        kernelX_use_median=True, kernelY_use_median=True,
                                        blocksize=200, nullvarmethod='permutation')

    pvalue = myblockobject.compute_pvalue(x, y)

    return pvalue
