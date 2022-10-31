class BaseDiscoveryModel(object):
    def __init__(self):
        raise NotImplementedError(f'Function __init__ of {type(self).__name__} is not implemented.')

    def fit(self, data, var_names):
        raise NotImplementedError(f'Function fit of {type(self).__name__} is not implemented.')

class BasePairCausalModel(object):
    pass
