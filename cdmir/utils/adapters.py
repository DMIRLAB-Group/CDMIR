from numpy import ndarray
from pandas import DataFrame


def data_form_converter_for_class_method(func):
    def wrapper(self, data, var_names=None, *args, **kwargs):
        assert type(data) in [DataFrame, ndarray]
        n = data.shape[1]
        if var_names is None:
            if type(data) == DataFrame:
                var_names = list(data.columns)
                data = data.to_numpy()
            else:
                var_names = [i for i in range(0, n)]
        else:
            assert len(var_names) == n
        return func(self=self, data=data, var_names=var_names, *args, **kwargs)

    return wrapper
