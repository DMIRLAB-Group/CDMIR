from unittest import TestCase

from ...graph import Graph, Edge, Mark
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
        g = Graph([f'X{i+1}' for i in range(14)])
        g.add_edge(Edge('X1', 'X3', Mark.Tail, Mark.Arrow))
        g.add_edge(Edge('X1', 'X11', Mark.Circle, Mark.Arrow))
        g.add_edge(Edge('X1', 'X14', Mark.Arrow, Mark.Arrow))

        # g.add_edge(Edge('X1', 'X20', Mark.Arrow, Mark.Arrow))
        plot_graph(g, circular_layout, is_latent=['X1', 'X2'])
