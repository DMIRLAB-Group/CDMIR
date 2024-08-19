from typing import Iterable

from cdmir.utils.adapters import data_form_converter_for_class_method


def _stringize_list(z: Iterable[int] = None):
    if z is None:
        return ''
    return ', '.join([str(s) for s in z])


def _get_cache_key(x: int, y: int, z: Iterable[int] = None):
    if x > y:
        x, y = y, x
    if z is None:
        return f'I({x}, {y})'
    else:
        sSlist = sorted(list(z))
        return f'I({x}, {y} | {_stringize_list(sSlist)})'


class ConditionalIndependentTest(object):

    @data_form_converter_for_class_method
    def __init__(self, data, var_names=None):
        self._data, self.var_names = data, var_names
        self.name_id = {name: i for i, name in enumerate(self.var_names)}
        self.cache = dict()

    def _get_name_id(self, name):
        return self.name_id[name]

    def test(self, x, y, z: Iterable = None):
        x_id, y_id = map(self._get_name_id, (x, y))
        z_ids = list(map(self._get_name_id, z)) if z is not None else None
        return self.itest(x_id, y_id, z_ids)

    def itest(self, x_id: int, y_id: int, z_ids: Iterable[int] = None):
        key = _get_cache_key(x_id, y_id, z_ids)
        if self.cache.get(key):
            value = self.cache[key]
        else:
            value = self.cal_stats(x_id, y_id, z_ids)
            self.cache[key] = value
        # print(f'{key}: {value}')
        return value

    def cal_stats(self, x: int, y: int, z: Iterable[int] = None):
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
