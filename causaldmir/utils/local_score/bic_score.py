from __future__ import annotations

from typing import Iterable

from numpy import ndarray, corrcoef, shape, log, mat, ix_
from numpy.linalg import inv
from pandas import DataFrame

from ._base import BaseLocalScoreFunction


class BICScore(BaseLocalScoreFunction):

    def __init__(self, data: ndarray | DataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.cov = corrcoef(data.T)
        self.sample_count = shape(data)[0]
        if not kwargs.__contains__('lambda_value'):
            self.lambda_value = 1
        else:
            self.lambda_value = kwargs["lambda_value"]

    def _score_function(self, i: int, parent_i: Iterable[int]):
        parent_i = list(parent_i)

        if len(parent_i) == 0:
            return self.sample_count * log(self.cov[i, i])

        yX = mat(self.cov[ix_([i], parent_i)])
        XX = mat(self.cov[ix_(parent_i, parent_i)])
        H = log(self.cov[i, i] - yX * inv(XX) * yX.T)

        return -(self.sample_count * H + log(self.sample_count) * len(parent_i) * self.lambda_value).item()

    def __call__(self, i: int, parent_i: Iterable[int], *args, **kwargs):
        return self._score(i, parent_i, self._score_function)

