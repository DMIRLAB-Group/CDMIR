from typing import Iterable

from .basic_independence import ConditionalIndependentTest
from .functional import kci


class KCI(ConditionalIndependentTest):
    def __init__(self, data, var_names=None):
        super().__init__(data, var_names=var_names)
        self._num_records = data.shape[0]

    def cal_stats(self, x: int, y: int, z: Iterable[int] = None):
        return kci(self._data[:, x], self._data[:, y], self._data[:, z])
