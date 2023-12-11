from causaldmir.utils.kernel.polynomial import PolynomialKernel
import numpy as np

def test_polynomial():#多项式核函数 (const+x.dot(y.T))^2
    # arr1=np.array([[1,2,3]])
    # arr2=np.array([[1,1,1]])

    np.random.seed(1)
    xs=np.random.randn(1,3)
    ys=np.random.randn(1,3)

    pk=PolynomialKernel()
    #print(pk.__call__(arr1,arr2))
    print(pk.__call__(xs,ys))

test_polynomial()
#修改polynomial.py的call方法将return self.__kernel(xs, ys, self.__kernel_func)改为return self._BaseKernel__kernel(xs, ys, self.__kernel_func)