import numpy as np

from causaldmir.utils.kernel._base import BaseKernel
from causaldmir.utils.kernel.linear import LinearKernel
def test_linear():

    arr1=np.array([1,2,3])
    arr2=np.array([1,1,1])

    l=LinearKernel()
    l(arr1,arr2)
    #l.__call__(arr1,arr2)
    print(arr1.dot(arr2.T))
    print(l.__call__(arr1,arr2))

test_linear()