import warnings
from typing import Iterable

import numpy as np
import pandas as pd


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
        self.add_node_np = np.vectorize(self.add_node)

    def init_graph(self):
        for node_u in self.node_set:
            self._adj[node_u] = dict()

    def add_node(self, node_u):
        if node_u in self.node_set:
            warnings.warn(f'The node {node_u} is already in the class {self.__class__.__name__} object.', UserWarning)
        else:
            self.node_set.add(node_u)
            self.node_list.append(node_u)
            self._adj[node_u] = dict()

    def add_nodes(self, nodes):
        for node_u in nodes:
            self.add_node(node_u)

    def add_nodes_map(self, nodes):
        tuple(map(self.add_node, nodes))

    def add_nodes_np(self, nodes):
        # add_node_np = np.vectorize(self.add_node)
        self.add_node_np(nodes)

    def add_edge(self, node_u, node_v, mark_u, mark_v, overwrite=False):
        self.check_missing_node(node_u)
        self.check_missing_node(node_v)

        if not overwrite:
            assert self.is_connected(node_u, node_v), f'Node {node_u} and node {node_v} already have edges connected.'

        self._adj[node_u][node_v] = mark_v
        self._adj[node_v][node_u] = mark_u

    def remove_node(self, node_u):
        self.check_missing_node(node_u)

        # Remove the edge connected to the node.
        for node_v in self.node_set:
            self._adj[node_v].pop(node_u, None)

        # Remove the node
        self._adj.pop(node_u)
        self.node_set.remove(node_u)
        self.node_list.remove(node_u)

    def remove_nodes(self, nodes):
        tuple(map(self.remove_node, nodes))

    def remove_edge(self, node_u, node_v):
        if self.is_connected(node_u, node_v):
            warnings.warn(
                f'There are no edges connecting node {node_u} and node {node_v} in the class {self.__class__.__name__} object.',
                UserWarning)
        else:
            self._adj[node_u].pop(node_v, None)
            self._adj[node_v].pop(node_u, None)

    def is_connected(self, node_u, node_v):
        self.check_missing_node(node_u)
        self.check_missing_node(node_v)

        return not (self._adj[node_u].get(node_v) is None or self._adj[node_v].get(node_u) is None)

    def get_neighbours(self, node_u):
        for node_v in self._adj[node_u].keys():
            yield node_v

    def to_numpy(self):
        n = len(self.node_list)
        node_index = {node: i for i, node in enumerate(self.node_list)}
        adj_matrix = np.zeros(shape=(n, n), dtype=int)

        for i, node_u in enumerate(self.node_list):
            for node_v in self._adj[node_u]:
                adj_matrix[node_index[node_v], i] = self._adj[node_u][node_v]

        return adj_matrix

    def to_dataframe(self):
        return pd.DataFrame(self.to_numpy(), index=self.node_list, columns=self.node_list, dtype=int)

    def check_missing_node(self, node_u):
        assert node_u in self.node_set, f'Node {node_u} doesn\'t exist in the class {self.__class__.__name__} object.'

    def check_mark(self, mark_u, valid_marks):
        assert mark_u in valid_marks, f'The mark {mark_u} should not appear in objects of class {self.__class__.__name__}.'
