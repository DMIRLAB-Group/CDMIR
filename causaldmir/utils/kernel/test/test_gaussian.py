import numpy as np

from causaldmir.utils.kernel.gaussian import GaussianKernel

#arr1 = np.array([[1, 2, 3, 4], [1, 3, 4, 5]])
#arr2 = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])

def test_gaussian_case1(): #shape(x)[1]==shape(y)[1]
    np.random.seed(10)
    xs=np.random.randn(20,20)
    ys=np.random.randn(30,20)

    gk=GaussianKernel()
    gk(xs,ys)
    print(gk.__call__(xs,ys))
    # print(gk(xs,ys))


def test_gaussian_case2():#y is none
    np.random.seed(10)
    xs = np.random.randn(20, 20)
    #ys = np.array([])

    gk = GaussianKernel()
    gk(xs,ys=None)
    print(gk.__call__(xs,ys=None))
    # print(gk(xs, ys=None))

test_gaussian_case1()
# test_gaussian_case2()
