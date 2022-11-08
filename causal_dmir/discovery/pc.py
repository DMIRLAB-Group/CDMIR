from ..utils import Independence


class PC(object):
    def __init__(self,
                 indep_test: Independence=None,
                 alpha: float = 0.05
                 ):

        self.causal_graph = None
        self.indep_test = indep_test
        self.alpha = alpha
        self.independence = None


    def fit(self, data):

        pass