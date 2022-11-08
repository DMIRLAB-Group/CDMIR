import logging
import random
from itertools import permutations
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.random import normal
from scipy import stats

from causaldmir.discovery.constraint import PC
from causaldmir.graph import DiGraph, Edge, dag2cpdag
from causaldmir.utils.independence import Dsep, FisherZ
from causaldmir.utils.metrics.graph_evaluation import graph_equal

logging.basicConfig(level=logging.DEBUG,
                    format=' %(levelname)s :: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


class TestPC(TestCase):
    def test_pc_numpy_dataset(self):
        X = self.gen_numpy_dataset()
        pc = PC()
        pc.fit(X, indep_cls=FisherZ)
        print(pc.causal_graph)
        assert pc.causal_graph.node_list == list(range(X.shape[1]))
        assert pc.causal_graph.number_of_edges() == 4

    def test_pc_pandas_dataset(self):
        pd = self.gen_pandas_dataset()
        pc = PC(verbose=True)
        pc.fit(pd, indep_cls=FisherZ)
        print(pc.causal_graph)
        assert (pc.causal_graph.node_list == pd.columns).all()
        assert pc.causal_graph.number_of_edges() == 7

    def test_pc_with_random_graph(self):
        random.seed(3407)
        np.random.seed(3407)
        node_dim = 10
        for i in range(100):
            graph = np.random.choice([0, 1], size=(node_dim, node_dim), p=[0.7, 0.3])
            dag = np.tril(graph, k=-1)
            perm_mat = np.random.permutation(np.eye(node_dim))
            dag = perm_mat.T @ dag @ perm_mat
            true_graph = DiGraph(range(node_dim))
            data = np.empty(shape=(0, node_dim))
            for node_u, node_v in permutations(range(node_dim), 2):
                if dag[node_u, node_v] > 0.5:
                    true_graph.add_edge(Edge(node_u, node_v))
            pc = PC(verbose=True)
            pc.fit(data, indep_cls=Dsep, true_graph=true_graph)

            cpdag = dag2cpdag(true_graph)
            assert graph_equal(pc.causal_graph, cpdag)

    def gen_numpy_dataset(self):
        random.seed(3407)
        np.random.seed(3407)
        sample_size = 100000
        X1 = normal(size=(sample_size, 1))
        X2 = X1 + normal(size=(sample_size, 1))
        X3 = X1 + normal(size=(sample_size, 1))
        X4 = X2 + X3 + normal(size=(sample_size, 1))
        X = np.hstack((X1, X2, X3, X4))
        X = stats.zscore(X, ddof=1, axis=0)
        return X

    def gen_pandas_dataset(self):
        random.seed(3407)
        np.random.seed(3407)
        sample_size = 100000
        X1 = normal(size=(sample_size, 1))
        X2 = normal(size=(sample_size, 1))
        X3 = X1 + X2 + 0.3 * normal(size=(sample_size, 1))
        X4 = X1 + X3 + 0.3 * normal(size=(sample_size, 1))
        X5 = 0.5 * X1 + 0.5 * X2 + X3 + X4 + 0.3 * normal(size=(sample_size, 1))

        X = np.hstack((X1, X2, X3, X4, X5))
        X = stats.zscore(X, ddof=1, axis=0)
        return pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E'])
