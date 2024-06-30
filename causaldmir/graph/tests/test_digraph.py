import logging
from copy import deepcopy
from unittest import TestCase

from causaldmir.graph import DiGraph, Edge, Mark

logging.basicConfig(level=logging.DEBUG,
                    format=' %(levelname)s :: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

sample_dag = DiGraph(range(5))
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
for edge in edges:
    sample_dag.add_edge(Edge(*edge))

class TestDiGraph(TestCase):
    def test_case1(self):
        assert 3 not in sample_dag.get_reachable_nodes(0, [1, 2])
        assert 2 not in sample_dag.get_reachable_nodes(1, [0])
        assert 2 in sample_dag.get_reachable_nodes(1, [0, 3])
        assert 2 in sample_dag.get_reachable_nodes(1, [0, 4])

    def test_case2(self):
        parents = dict()
        children = dict()
        for i in range(5) :
            parents[i] = []
            children[i] = []
        for u, v in edges :
            parents[v].append(u)
            children[u].append(v)

        for u in range(5):
            assert list(sample_dag.get_parents(u)) == parents[u]
            assert list(sample_dag.get_children(u)) == children[u]

    def test_is_d_separate(self):
        dag = DiGraph(range(5))
        edges = [(0, 1), (2, 1), (1, 3), (3, 4), (2, 3)]
        for e in edges:
            dag.add_edge(Edge(*e))
        assert dag.is_d_separate(0, 2, []) == True
        assert dag.is_d_separate(0, 2, [1]) == False
        assert dag.is_d_separate(0, 2, [3]) == False
        assert dag.is_d_separate(0, 2, [1, 3]) == False
        assert dag.is_d_separate(2, 4, []) == False
        assert dag.is_d_separate(2, 4, [1, 3]) == True

    def test_degree(self):
        ind = [0 for i in range(5)]
        outd =[0 for i in range(5)]
        for u, v in edges:
            ind[v] = ind[v] + 1
            outd[u] = outd[u] + 1

        for u in range(5):
            assert ind[u] == sample_dag.in_degree(u)
            assert outd[u] == sample_dag.out_degree(u)

    def test_topo_sort(self):
        topo_list = sample_dag.topo_sort()
        for i in range(5):
            for j in range(i+1, 5):
                u, v = topo_list[i], topo_list[j]
                assert v not in sample_dag._adj[u] or sample_dag._adj[v][u] != Mark.Arrow

    def test_is_acyclic(self):
        dag = deepcopy(sample_dag)

        assert dag.is_acyclic() == True
        dag.add_edge(Edge(4, 1))
        assert dag.is_acyclic() == False
        dag.remove_edge(4, 1)
        dag.add_edge(Edge(4, 0))
        assert dag.is_acyclic() == False