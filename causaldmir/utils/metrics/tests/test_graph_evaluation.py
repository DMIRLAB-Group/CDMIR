from unittest import TestCase

import numpy as np

from causaldmir.graph import DiGraph, Edge, Graph, Mark, PDAG

from causaldmir.utils.metrics import (
    arrow_evaluation,
    directed_edge_evaluation,
    graph_equal,
    shd,
    skeleton_evaluation,
)


class TestGraphEvaluation(TestCase):
    def test_graph_equal(self):
        g1 = PDAG([1, 2, 3])
        g2 = PDAG([1, 2, 3])

        assert graph_equal(g1, g2)

    def test_case1(self):
        g1 = PDAG([1, 2, 3])
        g1.create_complete_undirected_graph()
        g2 = PDAG([1, 2, 3])
        g2.create_complete_undirected_graph()

        skeleton_eval = skeleton_evaluation(g1, g2)
        assert np.isclose(skeleton_eval['precision'], 1.0)
        assert np.isclose(skeleton_eval['recall'], 1.0)
        assert np.isclose(skeleton_eval['f1'], 1.0)

        shd_eval = shd(g1, g2)
        assert shd_eval == 0



    def test_case2(self):
        g1 = PDAG([1, 2, 3])
        g1.add_edge(Edge(1, 2, Mark.Tail, Mark.Arrow))
        g2 = PDAG([1, 2, 3])
        g2.add_edge(Edge(1, 2, Mark.Tail, Mark.Arrow))


        arrow_eval = arrow_evaluation(g1, g2)
        assert np.isclose(arrow_eval['precision'], 1.0)
        assert np.isclose(arrow_eval['recall'], 1.0)
        assert np.isclose(arrow_eval['f1'], 1.0)

        directed_edge_eval = directed_edge_evaluation(g1, g2)
        assert np.isclose(directed_edge_eval['precision'], 1.0)
        assert np.isclose(directed_edge_eval['recall'], 1.0)
        assert np.isclose(directed_edge_eval['f1'], 1.0)

        shd_eval = shd(g1, g2)
        assert shd_eval == 0

    def test_case3(self):
        g1 = PDAG([1, 2, 3])
        g1.add_edge(Edge(1, 2, Mark.Tail, Mark.Arrow))
        g2 = PDAG([1, 2, 3])
        g2.add_edge(Edge(1, 2, Mark.Tail, Mark.Arrow))
        g2.add_edge(Edge(1, 3, Mark.Tail, Mark.Arrow))

        arrow_eval = arrow_evaluation(g1, g2)
        assert np.isclose(arrow_eval['precision'], 0.5)
        assert np.isclose(arrow_eval['recall'], 1.0)
        assert np.isclose(arrow_eval['f1'], 2/3)

        directed_edge_eval = directed_edge_evaluation(g1, g2)
        assert np.isclose(directed_edge_eval['precision'], 0.5)
        assert np.isclose(directed_edge_eval['recall'], 1.0)
        assert np.isclose(directed_edge_eval['f1'], 2/3)

        shd_eval = shd(g1, g2)
        assert shd_eval == 1


class TestAssertion(TestCase):
    def test_graph_type(self):
        g1 = DiGraph([1, 2, 3])
        g2 = PDAG([1, 2, 3])
        with self.assertRaises(AssertionError):
            _ = graph_equal(g1, g2)

    def test_graph_node(self):
        g1 = PDAG([1, 2, 3])
        g2 = PDAG([3, 2, 1])

        with self.assertRaises(AssertionError):
            _ = graph_equal(g1, g2)

