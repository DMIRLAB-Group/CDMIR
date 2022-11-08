# from typing import Iterable
#
# from numpy import (array, concatenate, corrcoef, ix_, log1p, ndarray, reshape,
#                    sqrt)
# from numpy.linalg import inv
# from scipy.stats import norm
#
#
# <<<<<<< HEAD
# def fisherz(x, y, z=None):
#     if z is None:
# =======
# def fisherz(x, y, S=None):
#     if S is None:
# >>>>>>> origin/wayne
#         data = array([x, y]).T
#     else:
#         x = reshape(x, (-1, 1)) if x.ndim == 1 else x
#         y = reshape(y, (-1, 1)) if y.ndim == 1 else y
# <<<<<<< HEAD
#         S = reshape(z, (-1, 1)) if z.ndim == 1 else z
#         data = concatenate((x, y, S), axis=1)
#     corr = corrcoef(data, rowvar=False)
#     num_records = data.shape[0]
#     if z is None:
#         return fisherz_from_corr(corr, num_records, 0, 1)
#     else:
#         return fisherz_from_corr(corr, num_records, 0, 1, list(range(2, 2 + z.shape[1])))
#
#
# def fisherz_from_corr(corr: ndarray, num_records: int, x_id: int, y_id: int, z_ids: Iterable[int] = None):
#     z_ids = [] if z_ids is None else z_ids
#     var = [x_id, y_id] + z_ids
#     sub_corr = corr[ix_(var, var)]
#     inv_mat = inv(sub_corr)
#     stats = -inv_mat[0, 1] / sqrt(abs(inv_mat[0, 0] * inv_mat[1, 1]))
#     abs_stats = min(0.9999999, abs(stats))
#     z = 1 / 2 * log1p(2 * abs_stats / (1 - abs_stats))
#     X = sqrt(num_records - len(z_ids) - 3) * abs(z)
#     pval = 2 * (1 - norm.cdf(abs(X)))
#     return pval, stats
# =======
#         S = reshape(S, (-1, 1)) if S.ndim == 1 else S
#         data = concatenate((x, y, S), axis=1)
#     corr = corrcoef(data, rowvar=False)
#     num_records = data.shape[0]
#     if S is None:
#         return fisherz_via_corr(corr, num_records, 0, 1)
#     else:
#         return fisherz_via_corr(corr, num_records, 0, 1, list(range(2, 2 + S.shape[1])))
#
#
# def fisherz_via_corr(corr, num_records, x, y, S=None):
#     S = [] if S is None else S
#     var = [x, y] + S
#     sub_corr = corr[ix_(var, var)]
#     inv_mat = inv(sub_corr)
#     stats = -inv_mat[0, 1] / sqrt(inv_mat[0, 0] * inv_mat[1, 1])
#     abs_stats = min(0.9999999, abs(stats))
#     z = 1/2 * log1p(2 * abs_stats / (1 - abs_stats))
#     X = sqrt(num_records - len(S) - 3) * abs(z)
#     pval = 2 * (1 - norm.cdf(abs(X)))
#     return stats, pval
# >>>>>>> origin/wayne


# 一个用S 一个用z 统一一下