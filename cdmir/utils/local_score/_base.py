from __future__ import annotations

from typing import Iterable

from numpy import ndarray
from pandas import DataFrame


class BaseLocalScoreFunction(object):

    def __init__(self, data: ndarray | DataFrame, *args, **kwargs):
        if type(data) == ndarray:
            self.data = data
        else:
            self.data = data.values
        self.cache_dict = dict()

    def _score(self, i: int, parent_i: Iterable[int], score_function):
        dict_key = hash(str((i, parent_i)))
        if self.cache_dict.__contains__(dict_key):
            res = self.cache_dict[dict_key]
        else:
            res = score_function(i, parent_i)
            self.cache_dict[dict_key] = res

        return res

    def _score_function(self, i: int, parent_i: Iterable[int]):
        raise NotImplementedError()

    def __call__(self, i: int, parent_i: Iterable[int], *args, **kwargs):
        return self._score(i, parent_i, self._score_function)
