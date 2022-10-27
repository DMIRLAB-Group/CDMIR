from numpy import array, concatenate, corrcoef, ix_, log1p, reshape, sqrt
from numpy.linalg import inv
from scipy.stats import norm


def fisherz(x, y, S=None):
    if S is None:
        data = array([x, y]).T
    else:
        x = reshape(x, (-1, 1)) if x.ndim == 1 else x
        y = reshape(y, (-1, 1)) if y.ndim == 1 else y
        S = reshape(S, (-1, 1)) if S.ndim == 1 else S
        data = concatenate((x, y, S), axis=1)
    corr = corrcoef(data, rowvar=False)
    num_records = data.shape[0]
    if S is None:
        return fisherz_via_corr(corr, num_records, 0, 1)
    else:
        return fisherz_via_corr(corr, num_records, 0, 1, list(range(2, 2 + S.shape[1])))


def fisherz_via_corr(corr, num_records, x, y, S=None):
    S = [] if S is None else S
    var = [x, y] + S
    sub_corr = corr[ix_(var, var)]
    inv_mat = inv(sub_corr)
    stats = -inv_mat[0, 1] / sqrt(inv_mat[0, 0] * inv_mat[1, 1])
    abs_stats = max(0.9999999, abs(stats))
    z = 1/2 * log1p(2 * abs_stats / (1 - abs_stats))
    X = sqrt(num_records - len(S) - 3) * abs(z)
    pval = 2 * (1 - norm.cdf(abs(X)))
    return stats, pval