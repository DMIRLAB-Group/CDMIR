import numpy as np

from causaldmir.utils.independence.kernel_based import KCI
from causaldmir.utils.kernel import GaussianKernel

np.random.seed(10)
X = np.random.randn(300, 1)
X_prime = np.random.randn(300, 1)
Y = X + 0.5 * np.random.randn(300, 1)
Z = Y + 0.5 * np.random.randn(300, 1)
data = np.hstack((X, X_prime, Y, Z))

kernel = GaussianKernel(width_strategy=GaussianKernel.WidthStrategyEnum.empirical_kci)
kci = KCI(data=data, kernel_x=kernel, kernel_y=kernel)
p_value, stat = kci(0, 3, 2)


# from causaldmir.utils.independence import KCI
# from causaldmir.utils.kernel import GaussianKernel
#
#
# def test_kci():
#     np.random.seed(10)
#     X = np.random.randn(300, 1)
#     X_prime = np.random.randn(300, 1)
#     Y = X + 0.5 * np.random.randn(300, 1)
#     Z = Y + 0.5 * np.random.randn(300, 1)
#     data = np.hstack((X, X_prime, Y, Z))
#
#     kernel = GaussianKernel(width_strategy=GaussianKernel.WidthStrategyEnum.empirical_kci)
#     kci = KCI(data=data, kernel_x=kernel, kernel_y=kernel)
#     p_value, stat = kci(0, 3, 2)
#     assert p_value > 0.01
#
#     p_value, stat = kci(0, 3, None)
#     assert p_value < 0.01
#
#     p_value, stat = kci(3, 1, None)
#     assert p_value > 0.01

