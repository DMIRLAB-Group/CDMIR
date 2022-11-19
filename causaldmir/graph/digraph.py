from collections import deque
from itertools import product
from typing import Iterable

from . import Edge, Graph, Mark


class DiGraph(Graph):
    def add_edge(self, edge: Edge, overwrite=False):
        self.check_mark(edge.mark_u, [Mark.Tail])
        self.check_mark(edge.mark_v, [Mark.Arrow])
        super().add_edge(edge, overwrite=overwrite)

    def get_parents(self, node_u):
        for node_v in self.get_neighbours(node_u):
            if self.is_arrow(node_v, node_u):
                yield node_v

    def get_children(self, node_u):
        for node_v in self.get_neighbours(node_u):
            if self.is_arrow(node_u, node_v):
                yield node_v

    def get_reachable_nodes(self, x, z: Iterable = None):
        '''
        Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques (1st ed.). The MIT Press.

        Parameters
        ----------
        x
        z

        Returns
        -------

        '''

        # Phase I: Insert all ancestors of z into a
        node_l_queue = deque(z)
        a = set()
        node_l_visited = {node: False for node in self.nodes}
        for node in z:
            node_l_visited[node] = True

        while len(node_l_queue) != 0:
            y = node_l_queue.popleft()
            if y not in a:
                for pa_y in self.get_parents(y):
                    if not node_l_visited[pa_y]:
                        node_l_queue.append(pa_y)
                        node_l_visited[pa_y] = True
            a |= {y}

        # Phase II: Traverse active trails starting from x

        # 0 means trailing up through y.
        # 1 means trailing down through y.
        node_direction_l_queue = deque([(x, 1)])
        node_direction_l_visited = {(node, direction): False for node, direction in product(self.nodes, [0, 1])}
        node_direction_l_visited[(x, 1)] = True

        def push_queue(key):
            if not node_direction_l_visited[key]:
                node_direction_l_queue.append(key)
                node_direction_l_visited[key] = True

        while len(node_direction_l_queue) != 0:
            y, direction = node_direction_l_queue.popleft()
            if y not in z:
                yield y
            if direction == 1 and y not in z:
                for pa_y in self.get_parents(y):
                    push_queue((pa_y, 1))
                for ch_y in self.get_children(y):
                    push_queue((ch_y, 0))

            elif direction == 0:
                if y not in z:
                    for ch_y in self.get_children(y):
                        push_queue((ch_y, 0))
                if y in a:
                    for pa_y in self.get_parents(y):
                        push_queue((pa_y, 1))

    def is_d_separate(self, x, y, z: Iterable = ()):
        # print(f'{x = }, {y = } | {list(z) = }')
        return y not in self.get_reachable_nodes(x, z)

    def in_degree(self, node_u):
        return sum(1 for _ in self.get_parents(node_u))

    def out_degree(self, node_u):
        return sum(1 for _ in self.get_children(node_u))

    def topo_sort(self):
        found_parents = {node: 0 for node in self.nodes}
        sorted_node_list = []
        q = deque()
        v = {node: False for node in self.nodes}
        for node_u in self.nodes:
            if self.in_degree(node_u) == found_parents[node_u]:
                q.append(node_u)
                v[node_u] = True
        while len(q) != 0:
            node_u = q.popleft()
            sorted_node_list.append(node_u)
            for ch_u in self.get_children(node_u):
                found_parents[ch_u] = found_parents[ch_u] + 1
                if not v[ch_u] and (self.in_degree(ch_u) == found_parents[ch_u]):
                    q.append(ch_u)
                    v[ch_u] = True

        return sorted_node_list
    def is_acyclic(self):
        return len(self.topo_sort()) == self.number_of_nodes()
