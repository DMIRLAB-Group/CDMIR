from unittest import TestCase
import numpy as np
import pandas as pd
from numpy.random import normal
from scipy import stats
import random
# from causaldmir.discovery.score_based import GES
from causaldmir.discovery.score_based.greedy_equivalence_search import Score_G
from causaldmir.utils.local_score import BICScore
from causaldmir.discovery.score_based import dist2
from causaldmir.graph.pdag import PDAG

class TestGES(TestCase):
    def test_dist2(self):
        x = [[0, 0, 0, 0, 0, ], [1, 1, 1, 1, 1, ], [2, 2, 2, 2, 2, ], [3, 3, 3, 3, 3, ], [0, 0, 1, 1, 0]]
        c = [[0, 0, 0, 0, 0, ], [1, 1, 1, 1, 1, ], [2, 2, 2, 2, 2, ], ]
        x = np.array(x)
        c = np.array(c)
        n2 = dist2(x, c)

        assert n2.shape == (5, 3)
        assert (n2 == np.array([[0., 5., 20.],
                               [5., 0., 5.],
                               [20., 5., 0.],
                               [45., 20., 5.],
                               [2., 3., 14.]])).all()

    def test_Score_G(self):
        random.seed(3407)
        np.random.seed(3407)
        sample_size = 100000
        X1 = normal(size=(sample_size, 1))
        X2 = X1 + normal(size=(sample_size, 1))
        X3 = X1 + normal(size=(sample_size, 1))
        X4 = X2 + X3 + normal(size=(sample_size, 1))
        X = np.hstack((X1, X2, X3, X4))
        score_function = BICScore(data=X, lambda_value=2)
        var_count = X.shape[1]
        node_names = [("x%d" % i) for i in range(var_count)]
        G = PDAG(node_names)
        score = Score_G(G, score_func=score_function)
        print(score)
# # X = stats.zscore(X, ddof=1, axis=0)
#
#
# score_function = BICScore(data=X, lambda_value=2)
# s = score_function(0, [1])
#
# ges = GES(score_function)
# ges.fit()
# ges.get_causal_graph()
