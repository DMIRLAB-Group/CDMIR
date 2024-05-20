from unittest import TestCase

from causaldmir.graph import Graph


class TestGraph(TestCase):
    def test_add_node_case1(self):
        g = Graph(range(5))
        g.add_node(6)
        assert g.node_list == [0, 1, 2, 3, 4, 6] and g.node_set == set(g.node_list)
