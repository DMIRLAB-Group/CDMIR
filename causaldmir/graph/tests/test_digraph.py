from unittest import TestCase

from .. import DiGraph, Edge


class TestDiGraph(TestCase):
    def test_case1(self):
        dag = DiGraph(range(4))
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for edge in edges:
            dag.add_edge(Edge(*edge))

        assert 3 not in dag.get_reachable_nodes(0, [1, 2])
        assert 2 not in dag.get_reachable_nodes(1, [0])
        assert 2 in dag.get_reachable_nodes(1, [0, 3])