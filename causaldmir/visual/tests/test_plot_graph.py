from unittest import TestCase

from ...graph import Graph
from .. import circular_layout, plot_graph


class TestPlotGraph(TestCase):
    def test_case1(self):
        g = Graph(range(2))
        plot_graph(g, circular_layout)

    def test_case2(self):
        g = Graph()
        g.add_node('dmir')
        g.add_node(1)
        g.add_node('2')
        g.add_node('X3')
        g.add_node('?')
        g.add_node('L3')
        plot_graph(g, circular_layout)

    def test_case3(self):
        g = Graph(range(20))
        plot_graph(g, circular_layout)
