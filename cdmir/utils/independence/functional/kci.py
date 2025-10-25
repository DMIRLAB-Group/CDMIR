import numpy as np
from numpy import exp
from scipy import stats
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist, squareform


class KCI:
    """
    Simplified Kernel-based Conditional Independence test (unconditional version)
    Uses only Gaussian kernels for independence testing
    """

    def __init__(self, null_samples=1000, use_gamma_approx=True):
        """
        Initialize KCI test

        Parameters:
        null_samples: number of samples for null distribution (if not using gamma approximation)
        use_gamma_approx: whether to use gamma approximation for p-value (faster)
        """
        self.null_samples = null_samples
        self.use_gamma_approx = use_gamma_approx

    def test(self, X, Y):
        """
        Perform independence test between X and Y

        Parameters:
        X: input data matrix (n_samples x n_features_x)
        Y: input data matrix (n_samples x n_features_y)

        Returns:
        p_value: p-value for independence test
        test_stat: test statistic value
        """
        n = X.shape[0]

        # Standardize data
        X = stats.zscore(X, ddof=1, axis=0)
        Y = stats.zscore(Y, ddof=1, axis=0)
        X[np.isnan(X)] = 0
        Y[np.isnan(Y)] = 0

        # Compute Gaussian kernel matrices
        Kx = self._gaussian_kernel(X)
        Ky = self._gaussian_kernel(Y)

        # Center kernel matrices
        H = np.eye(n) - np.ones((n, n)) / n
        Kx = H @ Kx @ H
        Ky = H @ Ky @ H

        # Compute test statistic (HSIC)
        test_stat = np.sum(Kx * Ky)

        # Compute p-value
        p_value = self._gamma_approx_pvalue(Kx, Ky, test_stat)

        return p_value, test_stat

    def _gaussian_kernel(self, data, width=1.):
        """
        Compute Gaussian kernel matrix

        Parameters:
        data: input data matrix

        Returns:
        kernel_matrix: Gaussian kernel matrix
        """
        n, d = data.shape

        sq_dists = squareform(pdist(data, 'sqeuclidean'))
        K = exp(-0.5 * sq_dists)

        return K

    def _gamma_approx_pvalue(self, Kx, Ky, test_stat):
        """
        Compute p-value using Gamma distribution approximation

        Parameters:
        Kx: centered kernel matrix for X
        Ky: centered kernel matrix for Y
        test_stat: test statistic value

        Returns:
        p_value: approximated p-value
        """
        n = Kx.shape[0]

        # Compute moments for Gamma approximation
        mean_approx = np.trace(Kx) * np.trace(Ky) / n
        var_approx = 2 * np.sum(Kx ** 2) * np.sum(Ky ** 2) / n ** 2

        # Gamma distribution parameters
        k = mean_approx ** 2 / var_approx
        theta = var_approx / mean_approx

        return 1 - stats.gamma.cdf(test_stat, k, scale=theta)
