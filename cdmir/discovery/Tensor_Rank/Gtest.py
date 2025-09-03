import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly import kruskal_to_tensor
from scipy import stats
from tensorly.decomposition import parafac,non_negative_parafac
# Ensure Tensorly uses the NumPy backend explicitly
tl.set_backend('numpy')

def compute_cp_decomposition(tensor, rank):
    """Perform CP decomposition on a given tensor with a specified rank."""
    factors = non_negative_parafac(tensor, rank=rank, init='random', tol=1e-50, verbose=False, n_iter_max=800,cvg_criterion="rec_error",normalize_factors=True)
    reconstructed_tensor = kruskal_to_tensor(factors)
    return reconstructed_tensor, factors

def frobenius_norm_error(tensor, reconstructed_tensor):
    """Calculate the relative Frobenius norm error between observed and reconstructed tensors."""
    # Ensure tensors are in float format for tensorly calculations
    tensor = tensor.astype(np.float64)
    reconstructed_tensor = reconstructed_tensor.astype(np.float64)

    # Calculate the residual tensor
    residual_tensor = tensor - reconstructed_tensor
    residual_frobenius_norm = np.linalg.norm(residual_tensor.ravel(), ord=2)
    tensor_frobenius_norm = np.linalg.norm(tensor.ravel(), ord=2)
    relative_error = residual_frobenius_norm / tensor_frobenius_norm
    return relative_error

def normalize_expected(observed, expected):
    """Normalize the expected tensor to have the same sum as the observed tensor."""
    observed_sum = np.sum(observed)
    expected_sum = np.sum(expected)

    if expected_sum != 0:
        expected = expected * (observed_sum / expected_sum)

    return expected

def chi_square_goodness_of_fit(observed, expected):
    """Perform the Chi-square goodness of fit test between observed and expected counts."""
    observed = observed.astype(np.float64)
    expected = expected.astype(np.float64)

    # Normalize expected counts
    expected = normalize_expected(observed, expected)

    chi_square_statistic, p_value = stats.chisquare(f_obs=observed.ravel(), f_exp=expected.ravel())
    return chi_square_statistic, p_value

def t_test_goodness_of_fit(observed, expected):
    """Perform a one-sample T-test between observed and expected counts."""
    observed = observed.astype(np.float64)
    expected = expected.astype(np.float64)
    residuals = observed - expected
    t_statistic, p_value = stats.ttest_1samp(residuals.ravel(), 0)
    #t_statistic, p_value = stats.ttest_rel(observed.ravel(), expected.ravel())
    return t_statistic, p_value


def Get_Reconstructed_Error(tensor, rank):
    reconstructed_tensor, _ = compute_cp_decomposition(tensor, rank)
    relative_error = frobenius_norm_error(tensor, reconstructed_tensor)

    return relative_error

def test_goodness_of_fit(tensor, rank):
    """
    Test the goodness of fit for a tensor decomposition using Frobenius norm error, Chi-square test, and T-test.

    Parameters:
        tensor (ndarray): The original four-way tensor.
        rank (int): The rank of the CP decomposition.

    Returns:
        dict: A dictionary containing the Frobenius norm error, Chi-square statistic, T-test statistic, and p-value.
    """
    reconstructed_tensor, _ = compute_cp_decomposition(tensor, rank)
    relative_error = frobenius_norm_error(tensor, reconstructed_tensor)

    # Use the original tensor data as the observed counts
    observed = tensor
    expected = reconstructed_tensor

    chi_square_statistic, chi_square_p_value = chi_square_goodness_of_fit(observed, expected)
    t_statistic, t_p_value = t_test_goodness_of_fit(observed, expected)

    return chi_square_p_value

    return {
        'Frobenius Norm Error': relative_error,
        'Chi-square Statistic': chi_square_statistic,
        'Chi-square P-value': chi_square_p_value,
        'T-test Statistic': t_statistic,
        'T-test P-value': t_p_value
    }
