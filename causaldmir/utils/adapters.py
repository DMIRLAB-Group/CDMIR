from numpy import ndarray
from pandas import DataFrame


def data_form_converter(data, var_names):
    assert type(data) in [DataFrame, ndarray]
    n = data.shape[1]
    if var_names is None:
        if type(data) == DataFrame:
            var_names = list(data.columns)
        else:
            var_names = [i for i in range(0, n)]
    else:
        assert len(var_names) == n
    if type(data) == DataFrame:
        data = data.to_numpy()

    return data, var_names

# def data_form_converter(func):
#     def wrapper(data, var_names=None, *args, **kwargs):
#         assert type(data) in [DataFrame, ndarray]
#         n = data.shape[1]
#         if var_names is None:
#             if type(data) == DataFrame:
#                 var_names = list(data.columns)
#                 data = data.to_numpy()
#             else:
#                 var_names = [i for i in range(0, n)]
#         else:
#             assert len(var_names) == n
#         return func(data, var_names, *args, **kwargs)
#
#     return wrapper
