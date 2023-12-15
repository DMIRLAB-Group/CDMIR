from unittest import TestCase

from .. import Graph
from .. import Edge
from .. import Mark

from copy import deepcopy
import logging

logging.basicConfig(level=logging.DEBUG,
                    format=' %(levelname)s :: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

sample_graph = Graph(range(1, 6))
sample_graph._adj[1][3], sample_graph._adj[3][1] = Mark.Arrow, Mark.Tail
sample_graph._adj[2][5], sample_graph._adj[5][2] = Mark.Arrow, Mark.Tail
sample_graph._adj[4][2], sample_graph._adj[2][4] = Mark.Arrow, Mark.Tail
sample_graph._adj[3][4], sample_graph._adj[4][3] = Mark.Arrow, Mark.Tail


class TestGraph(TestCase):
    def test_init(self):
        g = Graph([1, 2, 'x', '2'])
        assert g.node_set == set([1, 2, 'x', '2'])
        g = Graph(x for x in range(10))
        assert g.node_set == set(x for x in range(10))
        def gen():
            for i in range(10): yield i
        g = Graph(gen())
        assert g.node_set == set(gen())


    def test_add_node(self):
        g = Graph(range(5))
        g.add_node(6)
        assert g.node_list == [0, 1, 2, 3, 4, 6] and g.node_set == set(g.node_list)
        assert {0, 1, 2, 3, 4, 6} == g._adj.keys()

    def test_add_nodes(self):
        g = Graph(range(1))
        g.add_nodes(range(1, 5))
        assert g.node_list == list(range(0, 5)) and g.node_set == set(g.node_list)

    def test_nodes(self):
        g = Graph(range(1, 6))
        assert list(g.nodes) == list(range(1, 6))

    def test_nodes_num(self):
        g = Graph(range(1, 6))
        assert g.number_of_nodes() == 5 and g.number_of_nodes() == len(g.node_list) \
            and g.number_of_nodes() == len(g.node_set)

    def test_add_edge(self):
        g = Graph(range(1, 6))
        g.add_edge(Edge(node_u=1, node_v=4))
        assert g._adj[1][4] == Mark.Arrow and g._adj[4][1] == Mark.Tail
        assert all(j not in g._adj[i].keys() for i in range(1, 6) for j in range(1, 6) if i != j \
                   and (i, j) not in {(1, 4), (4, 1)})
        g.add_edge(Edge(node_u=4, node_v=1), True)
        assert g._adj[1][4] == Mark.Tail and g._adj[4][1] == Mark.Arrow
        assert all(j not in g._adj[i].keys() for i in range(1, 6) for j in range(1, 6) if i != j \
                   and (i, j) not in {(1, 4), (4, 1)})

    def test_add_edges(self):
        g = Graph(range(1, 6))
        edges = [
            Edge(1, 3),
            Edge(2, 5),
            Edge(4, 2),
            Edge(3, 4),
        ]
        g.add_edges(edges)
        assert g._adj[1][3] == Mark.Arrow and g._adj[3][1] == Mark.Tail
        assert g._adj[2][5] == Mark.Arrow and g._adj[5][2] == Mark.Tail
        assert g._adj[4][2] == Mark.Arrow and g._adj[2][4] == Mark.Tail
        assert g._adj[3][4] == Mark.Arrow and g._adj[4][3] == Mark.Tail
        assert all(j not in g._adj[i].keys() for i in range(1, 6) for j in range(1, 6) if i != j \
                   and (i, j) not in {(1, 3), (3, 1), (2, 5), (5, 2), (4, 2), (2, 4), (3, 4), (4, 3)})

    def test_get_edge(self):
        g = Graph(range(1, 6))
        g._adj[1][4], g._adj[4][1] = Mark.Arrow, Mark.Tail
        e = g.get_edge(1, 4)
        assert e.node_u == 1 and e.node_v == 4 and e.mark_u == Mark.Tail and e.mark_v == Mark.Arrow

    def test_edges(self):
        g = sample_graph
        edge_list = [Edge(1, 3), Edge(2, 4, Mark.Arrow, Mark.Tail), Edge(2, 5), Edge(3, 4), ]
        assert list(g.edges) == edge_list

    def test_number_of_edges(self):
        g = deepcopy(sample_graph)
        assert g.number_of_edges() == 4

    def test_remove_node(self):
        g = deepcopy(sample_graph)
        g.remove_node(2)
        # check removed nodes
        assert 2 not in g._adj[5].keys() and 2 not in g._adj[4].keys()
        assert all(2 not in s for s in (g._adj, g.node_set, g.node_list))
        # check left nodes
        assert (g._adj[1][3], g._adj[3][1]) == (Mark.Arrow, Mark.Tail)
        assert (g._adj[3][4], g._adj[4][3]) == (Mark.Arrow, Mark.Tail)

    def test_remove_nodes_from(self):
        g = deepcopy(sample_graph)
        g.remove_nodes_from([2, 4])
        # check removed nodes
        assert 2 not in g._adj[5].keys()
        assert 4 not in g._adj[3].keys()
        assert all(i not in j for i in (2, 4) for j in (g._adj, g.node_set,  g.node_list))
        # check left nodes
        assert (g._adj[1][3], g._adj[3][1]) == (Mark.Arrow, Mark.Tail)


    def test_remove_edge(self):
        g = deepcopy(sample_graph)
        g.remove_edge(3, 1)
        # check removed edges
        assert 1 not in g._adj[3].keys()
        assert 3 not in g._adj[1].keys()
        # check left edges
        assert (g._adj[2][5], g._adj[5][2]) == (Mark.Arrow, Mark.Tail)
        assert (g._adj[4][2], g._adj[2][4]) == (Mark.Arrow, Mark.Tail)
        assert (g._adj[3][4], g._adj[4][3]) == (Mark.Arrow, Mark.Tail)

    def test_remove_edges_from(self):
        pass

    def test_is_connected(self):
        g = deepcopy(sample_graph)
        # check conneccted
        assert all(g.is_connected(i, j)==True and g.is_connected(j, i)==True for i in range(1, 6) for j in range(1, 6) \
                   if (i, j) in {(1, 3), (2, 5), (4, 2), (3, 4)})
        # check not connected
        assert all(g.is_connected(i, j)==False and g.is_connected(j, i)==False for i in range(1, 6) for j in range(1, 6)\
                   if j > i and (i, j) not in {(1, 3), (2, 4), (2, 5), (3, 4)})

    def test_get_neighbours(self):
        g = deepcopy(sample_graph)
        nb = g.get_neighbours(3)
        assert list(nb) == [1, 4]
        nb = g.get_neighbours(2)
        assert list(nb) == [5, 4]

    def test_is_arrow(self):
        g = deepcopy(sample_graph)
        assert all(g.is_arrow(i, j)==True for (i, j) in {(1 ,3), (2, 5), (4, 2), (3, 4)})
        assert all(g.is_arrow(i, j)==False for i in range(1, 6) for j in range(1, 6) \
                   if (i, j) not in {(1 ,3), (2, 5), (4, 2), (3, 4)})

    def test_is_tail(self):
        g = deepcopy(sample_graph)
        assert all(g.is_tail(j, i)==True for (i, j) in {(1 ,3), (2, 5), (4, 2), (3, 4)})
        assert all(g.is_tail(j, i)==False for i in range(1, 6) for j in range(1, 6) \
                   if (i, j) not in {(1 ,3), (2, 5), (4, 2), (3, 4)})

    def test_is_circle(self):
        g = Graph(range(1, 6))
        g._adj[1][3], g._adj[3][1] = Mark.Circle, Mark.Tail
        g._adj[2][5], g._adj[5][2] = Mark.Circle, Mark.Tail
        g._adj[4][2], g._adj[2][4] = Mark.Circle, Mark.Tail
        g._adj[3][4], g._adj[4][3] = Mark.Circle, Mark.Tail
        assert all(g.is_circle(i, j)==True for (i, j) in {(1 ,3), (2, 5), (4, 2), (3, 4)})
        assert all(g.is_circle(i, j)==False for i in range(1, 6) for j in range(1, 6) \
                   if (i, j) not in {(1 ,3), (2, 5), (4, 2), (3, 4)})

    def test_is_fully_directed(self):
        g = deepcopy(sample_graph)
        assert all(g.is_fully_directed(i, j)==True for (i, j) in {(1 ,3), (2, 5), (4, 2), (3, 4)})
        assert all(g.is_fully_directed(i, j)==False for i in range(1, 6) for j in range(1, 6) \
                   if (i, j) not in {(1 ,3), (2, 5), (4, 2), (3, 4)})

    def test_is_fully_undirected(self):
        g = Graph(range(1, 6))
        g._adj[1][3], g._adj[3][1] = Mark.Tail, Mark.Tail
        g._adj[2][5], g._adj[5][2] = Mark.Tail, Mark.Tail
        g._adj[4][2], g._adj[2][4] = Mark.Tail, Mark.Tail
        g._adj[3][4], g._adj[4][3] = Mark.Tail, Mark.Tail
        assert all(g.is_fully_undirected(i, j)==True and g.is_fully_undirected(j, i)==True \
                   for (i, j) in {(1 ,3), (2, 5), (4, 2), (3, 4)})
        assert all(g.is_fully_undirected(i, j)==False for i in range(1, 6) for j in range(1, 6) \
                   if (i, j) not in {(1 ,3), (3, 1), (2, 5), (5, 2), (4, 2), (2, 4), (3, 4), (4, 3)})
