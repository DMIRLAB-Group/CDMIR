from __future__ import annotations

from typing import List, Tuple

from numpy import ndarray
from pandas import DataFrame


class BaseConditionalIndependenceTest(object):
    """
    Base model for conditional independence test (CIT).
    """

    @staticmethod
    def __init_input_data(data: ndarray | DataFrame):
        if type(data) == ndarray:
            var_dim = data.shape[1]
            var_names = ["x%d" % i for i in range(var_dim)]
            var_values = data
        else:
            var_names = data.columns
            var_values = data.values
        name_dict = {var_names[index]: index for index in range(len(var_names))}
        name_to_index = lambda name: name_dict[name]
        index_to_name = lambda index: var_names[index]
        return var_values, var_names, name_to_index, index_to_name

    def __init__(self, data: ndarray | DataFrame, *args, **kwargs):
        self._data, self._names, self._name_to_index, self._index_to_name = self.__init_input_data(data)
        self.cache_dict = dict()

    def __input_to_list(self, input: int | str | List[int | str] | ndarray | None) -> List[int] | None:
        if input is None:
            final_res = None
        elif type(input) == int:
            final_res = [input]
        elif type(input) == str:
            final_res = [self._name_to_index(input)]
        elif type(input) == list:
            if len(input) == 0:
                final_res = None
            elif type(input[0]) == int:
                final_res = input
            elif type(input[0]) == str:
                final_res = [self._name_to_index(name) for name in input]
            else:
                raise Exception("data type should be int or str!")
        else:
            if len(input) == 0:
                final_res = None
            else:
                final_res = input.tolist()
        return final_res

    def _compute_p_value(self, xs: int | str | List[int | str] | ndarray,
                         ys: int | str | List[int | str] | ndarray,
                         zs: int | str | List[int | str] | ndarray | None = None,
                         compute_p_value_without_condition_func=None,
                         compute_p_value_with_condition_func=None) -> Tuple[float, float | ndarray | None]:
        x_ids = self.__input_to_list(xs)
        y_ids = self.__input_to_list(ys)
        z_ids = self.__input_to_list(zs)
        hash_key = hash(str((x_ids, y_ids, z_ids)))
        if self.cache_dict.__contains__(hash_key):
            p_value, stat = self.cache_dict[hash_key]
        else:
            if z_ids is None:
                if compute_p_value_without_condition_func is None:
                    raise Exception("'compute_p_value_without_condition_func' has not been given!")
                p_value, stat = compute_p_value_without_condition_func(x_ids, y_ids)
            else:
                if compute_p_value_with_condition_func is None:
                    raise Exception("'compute_p_value_with_condition_func' has not been given!")
                p_value, stat = compute_p_value_with_condition_func(x_ids, y_ids, z_ids)
            self.cache_dict[hash_key] = (p_value, stat)

        return p_value, stat

    def __call__(self, xs: int | str | List[int | str] | ndarray,
                 ys: int | str | List[int | str] | ndarray,
                 zs: int | str | List[int | str] | ndarray | None = None, *args, **kwargs) -> Tuple[
        float, float | ndarray | None]:
        return self._compute_p_value(xs, ys, zs, self.__compute_p_value_without_condition,
                                     self.__compute_p_value_with_condition)

    def __compute_p_value_with_condition(self, x_ids: List[int], y_ids: List[int], z_ids: List[int]) -> Tuple[
        float, float | ndarray | None]:
        raise NotImplementedError()

    def __compute_p_value_without_condition(self, x_ids: List[int], y_ids: List[int]) -> Tuple[
        float, float | ndarray | None]:
                        ys: int | str | List[int | str] | ndarray,
                        zs: int | str | List[int | str] | ndarray | None = None, *args, **kwargs) -> Tuple[float, float | ndarray | None]:
        return self._compute_p_value(xs, ys, zs, self.__compute_p_value_without_condition, self.__compute_p_value_with_condition)
