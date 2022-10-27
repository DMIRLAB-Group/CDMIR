from typing import Iterable

import numpy as np


def _stringize_list(S: Iterable[int] = None):
    if S is None:
        return ''
    return ', '.join([str(s) for s in S])


def _get_cache_key(x: int, y: int, S: Iterable[int] = None):
    if x > y:
        x, y = y, x
    if S is None:
        return f'I({x}, {y})'
    else:
        sSlist = sorted(list(S))
        return f'I({x}, {y} | {_stringize_list(sSlist)})'


class Independence(object):
    def __init__(self, dataset: np.ndarray):
        self._dataset = np.copy(dataset)
        self.cache = dict()

    def test(self, x: int, y: int, S: Iterable[int] = None):
        key = _get_cache_key(x, y, S)
        if self.cache.get(key):
            value = self.cache[key]
        else:
            value = self.cal_stats(x, y, S)
            self.cache[key] = value

        return value

    def cal_stats(self, x: int, y: int, S: Iterable[int] = None):
        '''
        Calculate a Statistic with associated p-value.
        Parameters
        ----------
        x : int
            variable index
        y : int
            variable index
        S : tuple
            variables index
        Returns
        -------
        '''
        raise NotImplementedError
