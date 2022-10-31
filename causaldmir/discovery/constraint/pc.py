import copy

from causaldmir.utils import data_form_converter_for_class_method
from causaldmir.utils.independence import ConditionalIndependentTest

from .adjacency_search import adjacency_search


class PC(object):
    def __init__(self,
                 alpha: float = 0.05,
                 adjacency_search_method=adjacency_search,
                 verbose: bool = False
                 ):
        self.skeleton = None
        self.causal_graph = None
        self.sep_set = None
        self.alpha = alpha
        self.indep_test = None
        self.adjacency_search_method = adjacency_search_method
        self.verbose = verbose

    @data_form_converter_for_class_method
    def fit(self, data, var_names, indep_cls, *args, **kwargs):
        assert issubclass(indep_cls, ConditionalIndependentTest)
        self.indep_test = indep_cls(data, var_names, *args, **kwargs)
        self.causal_graph, self.sep_set = self.adjacency_search_method(self.indep_test, self.indep_test.var_names,
                                                                       self.alpha, verbose=self.verbose)
        self.skeleton = copy.deepcopy(self.causal_graph)
        self.causal_graph.rule0(self.sep_set, self.verbose)
        self.causal_graph.orient_by_meek_rules(self.verbose)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def get_alpha(self):
        return self.alpha

    def set_verbose(self, verbose):
        self.verbose = verbose

    def get_verbose(self):
        return self.verbose
