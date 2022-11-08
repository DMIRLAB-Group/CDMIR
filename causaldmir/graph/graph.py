import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .edge import Edge
from .mark import Mark


class Graph(object):
    def __init__(self, nodes: Iterable = None):
        if nodes is None:
            self.node_set = set()
            self.node_list = []
        else:
            self.node_set = set(nodes)
            self.node_list = list(nodes)

        self._adj = dict()
        self.init_graph()

    def init_graph(self):
        for node_u in self.node_set:
            self._adj[node_u] = dict()

    def add_node(self, node_u):
        if node_u in self.node_set:
            warnings.warn(f'The node {node_u} is already in the class {type(self).__name__} object.', UserWarning)
        else:
            self.node_set.add(node_u)
            self.node_list.append(node_u)
            self._adj[node_u] = dict()

    def add_nodes(self, nodes):
        for node_u in nodes:
            self.add_node(node_u)

    @property
    def nodes(self) -> Iterable:
        for node in self.node_list:
            yield node

    def number_of_nodes(self):
        return sum(1 for _ in self.nodes)

    def add_edge(self, edge: Edge, overwrite=False):
        node_u, node_v, mark_u, mark_v = edge

        self.check_missing_node(node_u)
        self.check_missing_node(node_v)

        if not overwrite:
            assert not self.is_connected(node_u, node_v), \
                f'Node {node_u} and node {node_v} already have edges connected.'

        self._adj[node_u].update({node_v: mark_v})
        self._adj[node_v].update({node_u: mark_u})

    def add_edges(self, edge_list: Iterable[Edge], overwrite=False):
        for edge in edge_list:
            self.add_edge(edge, overwrite=overwrite)

    def get_edge(self, node_u, node_v):
        if self.is_connected(node_u, node_v):
            return Edge(node_u, node_v, self._adj[node_v][node_u], self._adj[node_u][node_v])
        return None

    @property
    def edges(self):
        n = len(self.node_list)
        for i in range(n):
            for j in range(i + 1, n):
                edge = self.get_edge(self.node_list[i], self.node_list[j])
                if edge:
                    yield edge

    def number_of_edges(self):
        return sum(1 for _ in self.edges)

    def remove_node(self, node_u):
        self.check_missing_node(node_u)

        # Remove the edge connected to the node.
        for node_v in self.node_set:
            self._adj[node_v].pop(node_u, None)

        # Remove the node
        self._adj.pop(node_u)
        self.node_set.remove(node_u)
        self.node_list.remove(node_u)

    def remove_nodes_from(self, nodes):
        tuple(map(self.remove_node, nodes))

    def remove_edge(self, node_u, node_v):
        if not self.is_connected(node_u, node_v):
            warnings.warn(
                f'There are no edges connecting node {node_u} and node {node_v} in the class {type(self).__name__} object.',
                UserWarning)
        else:
            self._adj[node_u].pop(node_v, None)
            self._adj[node_v].pop(node_u, None)

    def remove_edges_from(self):

        ...

    def is_connected(self, node_u, node_v) -> bool:
        self.check_missing_node(node_u)
        self.check_missing_node(node_v)
        return node_v in self._adj[node_u] and node_u in self._adj[node_v]

    def get_neighbours(self, node_u) -> Iterable:
        for node_v in self._adj[node_u].keys():
            yield node_v

    def is_arrow(self, node_u, node_v):
        return self.is_connected(node_u, node_v) and self._adj[node_u][node_v] == Mark.ARROW

    def is_tail(self, node_u, node_v):
        return self.is_connected(node_u, node_v) and self._adj[node_u][node_v] == Mark.Tail

    def is_circle(self, node_u, node_v):
        return self.is_connected(node_u, node_v) and self._adj[node_u][node_v] == Mark.CIRCLE

    def is_fully_directed(self, node_u, node_v):
        return self.is_connected(node_u, node_v) and self.is_tail(node_v, node_u) and self.is_arrow(node_u, node_v)

    def is_fully_undirected(self, node_u, node_v):
        return self.is_connected(node_u, node_v) and self.is_tail(node_v, node_u) and self.is_tail(node_u, node_v)

    def to_numpy(self, transpose=False) -> np.ndarray:
        n = len(self.node_list)
        node_index = {node: i for i, node in enumerate(self.node_list)}
        adj_matrix = np.zeros(shape=(n, n), dtype=int)

        for i, node_u in enumerate(self.node_list):
            for node_v in self._adj[node_u]:
                adj_matrix[node_index[node_v], i] = self._adj[node_u][node_v].value
        if transpose:
            return adj_matrix.T
        else:
            return adj_matrix

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_numpy(), index=self.node_list, columns=self.node_list, dtype=int)

    def check_missing_node(self, node_u):
        assert node_u in self.node_set, f'Node {node_u} doesn\'t exist in the class {type(self).__name__} object.'

    def check_mark(self, mark_u, valid_marks):
        assert mark_u in valid_marks, f'The mark {mark_u} should not appear in objects of class {type(self).__name__}.'

    def __str__(self):
        edge_list_num_str = [f'{i + 1}. {edge}' for i, edge in enumerate(self.edges)]

        return "\n".join(
            [
                f'Graph Class: {type(self).__name__}',
                'Graph Nodes:',
                ", ".join([str(node_name) for node_name in self.node_list]),
                'Graph Edges:',
                "\n".join(edge_list_num_str)
            ]
        )

    def debug(self):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        for node in self.nodes:
            pp.pprint(f'{node}: {self._adj[node]}')
        for edge in self.edges:
            print(edge)