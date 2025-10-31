import copy
from cdmir.utils import data_form_converter_for_class_method
from cdmir.utils.independence import ConditionalIndependentTest

from cdmir.discovery.constraint.adjacency_search import adjacency_search


class PC(object):
    """
	    A Python implementation of the PC algorithm for causal discovery.

        :param skeleton : Undirected graph representing causal relationships
        :param causal_graph : Partially directed acyclic graph (PDAG)
        :param sep_set : Separation sets for node pairs
        :param alpha : Significance level for independence tests
        :param indep_test : Example of Conditional Independence Test
        :param verbose : Whether to print algorithm progress
       """

    def __init__(self,
                 alpha: float = 0.05,
                 adjacency_search_method=adjacency_search,
                 verbose: bool = False
                 ):
        """

                Args:
                    alpha: Significance level for conditional independence tests (default: 0.05)
                    adjacency_search_method: Function for adjacency search phase (default: adjacency_search)
                    verbose: Whether to print algorithm progress (default: False)
        """
        self.skeleton = None
        self.causal_graph = None
        self.sep_set = None
        self.alpha = alpha
        self.indep_test = None
        self.adjacency_search_method = adjacency_search_method
        self.verbose = verbose

    @data_form_converter_for_class_method
    def fit(self, data, var_names, indep_cls, *args, **kwargs):
        """
                :param data: Input dataset
                :param var_names: List of variable names
                :param indep_cls: Conditional independence test class
                :param *args: Positional arguments for independence test
                :param **kwargs: Keyword arguments for independence test
        """
        assert issubclass(indep_cls, ConditionalIndependentTest)
        self.indep_test = indep_cls(data, var_names, *args, **kwargs)
        self.causal_graph, self.sep_set = self.adjacency_search_method(self.indep_test, self.indep_test.var_names,
                                                                       self.alpha, verbose=self.verbose)
        self.skeleton = copy.deepcopy(self.causal_graph)
        self.causal_graph.rule0(self.sep_set, self.verbose)
        self.causal_graph.orient_by_meek_rules(self.verbose)

    def set_alpha(self, alpha):
        # Set significance level for conditional independence tests.
        self.alpha = alpha

    def get_alpha(self):
        # Get current significance level.
        return self.alpha

    def set_verbose(self, verbose):
        # Toggle verbose output during algorithm execution.
        self.verbose = verbose

    def get_verbose(self):
        # Get current verbosity setting.
        return self.verbose
