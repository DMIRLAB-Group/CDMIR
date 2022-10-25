import numpy as np

from ..kernel import GaussianKernel
from ..kernel_based import KCI

np.random.seed(10)
X = np.random.randn(300, 1)
X_prime = np.random.randn(300, 1)
Y = X + 0.5 * np.random.randn(300, 1)
Z = Y + 0.5 * np.random.randn(300, 1)
data = np.hstack((X, X_prime, Y, Z))

kernel = GaussianKernel(width_strategy=GaussianKernel.WidthStrategyEnum.empirical_kci)
kci = KCI(data=data, kernel_x=kernel, kernel_y=kernel)
p_value, stat = kci(0, 3, 2)
