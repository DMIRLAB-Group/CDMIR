import random
from unittest import TestCase
from numpy.random import normal
import numpy as np
from scipy import stats

from ..pc import PC
from causaldmir.utils.independence import FisherZ

import pprint


class TestPC(TestCase):
    def test_case1(self):
        random.seed(3407)
        np.random.seed(3407)
        sample_size = 100000
        X1 = normal(size=(sample_size,))
        X2 = X1 + normal(size=(sample_size,))
        X3 = X1 + normal(size=(sample_size,))
        X4 = X2 + X3 + normal(size=(sample_size,))
        X = np.array([X1, X2, X3, X4]).T
        print(X.shape)
        X = stats.zscore(X, ddof=1, axis=0)
        print(np.mean(X, axis=0))
        pc = PC(verbose=True)
        pc.fit(FisherZ(X))
        print(pc.causal_graph)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(pc.causal_graph._adj)

    def test_case2(self):
        random.seed(3407)
        np.random.seed(3407)
        sample_size = 1000000
        X1 = normal(size=(sample_size,))
        X2 = normal(size=(sample_size,))
        X3 = X1 + X2 + normal(size=(sample_size,))
        X4 = X1 + X3 + normal(size=(sample_size,))
        X5 = X1 + X2 + X3 + X4 + normal(size=(sample_size,))

        X = np.array([X1, X2, X3, X4, X5]).T
        print(X.shape)
        X = stats.zscore(X, ddof=1, axis=0)
        print(np.mean(X, axis=0))
        pc = PC(verbose=True)
        pc.fit(FisherZ(X))
        print(pc.causal_graph)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(pc.causal_graph._adj)