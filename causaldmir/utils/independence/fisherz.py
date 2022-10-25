from typing import Iterable

from numpy import corrcoef

from .basic_independence import ConditionalIndependentTest
from .functional import fisherz_from_corr


class FisherZ(ConditionalIndependentTest):
    def __init__(self, data, var_names=None):
        super().__init__(data, var_names=var_names)
        self._num_records = data.shape[0]
        self._corr = corrcoef(self._data, rowvar=False)

    def cal_stats(self, x_id: int, y_id: int, z_ids: Iterable[int] = None):
        return fisherz_from_corr(corr=self._corr, num_records=self._num_records, x_id=x_id, y_id=y_id, z_ids=z_ids)
