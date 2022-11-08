from __future__ import annotations
from typing import Iterable, List, Tuple

from numpy import corrcoef

from ._base import BaseConditionalIndependenceTest
from .basic_independence import ConditionalIndependentTest
from .functional import fisherz_from_corr


class FisherZ(ConditionalIndependentTest):
    def __init__(self, data, var_names=None):
        super().__init__(data, var_names=var_names)
        self._num_records = data.shape[0]
        self._corr = corrcoef(self._data, rowvar=False)

    def cal_stats(self, x_id: int, y_id: int, z_ids: Iterable[int] = None):
        return fisherz_from_corr(corr=self._corr, num_records=self._num_records, x_id=x_id, y_id=y_id, z_ids=z_ids)



class FisherZ(BaseConditionalIndependenceTest):
    def __init__(self, data):
        super().__init__(data)
        self._num_records = data.shape[0]
        self._corr = corrcoef(self._data, rowvar=False)

    def __compute_p_value_with_condition(self, x_ids: List[int], y_ids: List[int], z_ids: List[int]) -> Tuple[float, float | ndarray | None]:
        stat, p_value = fisherz_via_corr(corr=self._corr, num_records=self._num_records, x=x_ids[0], y=y_ids[0], S=z_ids)
        return p_value, stat

    def __compute_p_value_without_condition(self, x_ids: List[int], y_ids: List[int]) -> Tuple[float, float | ndarray | None]:
        stat, p_value = fisherz_via_corr(corr=self._corr, num_records=self._num_records, x=x_ids[0], y=y_ids[0], S=None)
        return p_value, stat

    def __call__(self, xs: int | str | List[int | str] | ndarray,
                        ys: int | str | List[int | str] | ndarray,
                        zs: int | str | List[int | str] | ndarray | None = None, *args, **kwargs) -> Tuple[float, float | ndarray | None]:
        return self._compute_p_value(xs, ys, zs, self.__compute_p_value_without_condition, self.__compute_p_value_with_condition)

