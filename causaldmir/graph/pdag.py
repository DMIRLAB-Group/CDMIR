import warnings
from itertools import combinations, permutations

from . import Edge, Graph, Mark


class PDAG(Graph):
    def add_edge(self, edge: Edge, overwrite=False):
        self.check_mark(edge.mark_u, Mark.pdag_marks())
        self.check_mark(edge.mark_v, Mark.pdag_marks())
        super().add_edge(edge, overwrite=overwrite)

    def create_complete_undirected_graph(self):
        for node_u, node_v in combinations(self.nodes, 2):
            self.add_edge(Edge(node_u, node_v, Mark.Tail, Mark.Tail))

    def rule0(self, sep_set, verbose=False):
        for node_u, node_v in combinations(self.nodes, 2):
            if self.is_connected(node_u, node_v):
                continue
            for node_w in self.nodes:
                if node_w in [node_u, node_v]:
                    continue
                if not (self.is_connected(node_u, node_w) and self.is_connected(node_v, node_w)):
                    continue
                if node_w not in sep_set[(node_u, node_v)]:
                    self.remove_edge(node_u, node_w)
                    self.add_edge(Edge(node_u, node_w, Mark.Tail, Mark.ARROW))
                    self.remove_edge(node_v, node_w)
                    self.add_edge(Edge(node_v, node_w, Mark.Tail, Mark.ARROW))
                    if verbose:
                        print(f'Rule0: Orient {node_u} --- {node_w} --- {node_v} into {node_u} --> {node_w} <-- {node_v}.')

    def rule1(self, verbose=False):
        '''

        Parameters
        ----------
        verbose

        Returns
        -------

        '''
        changed = False
        for node_u, node_v in permutations(self.nodes, 2):
            if not self.is_fully_directed(node_u, node_v):
                continue
            for node_w in self.nodes:
                if node_w in [node_u, node_v]:
                    continue
                if (not self.is_connected(node_u, node_w)) and self.is_fully_undirected(node_v, node_w):
                    self.remove_edge(node_v, node_w)
                    self.add_edge(Edge(node_v, node_w, Mark.Tail, Mark.ARROW))
                    changed = True
                    if verbose:
                        print(f'Rule1: Orient {node_v} --- {node_w} into {node_v} --> {node_w}.')
        return changed

    def rule2(self, verbose=False):
        '''

        Parameters
        ----------
        verbose

        Returns
        -------

        '''
        changed = False
        for node_u, node_v in permutations(self.nodes, 2):
            if not self.is_fully_directed(node_u, node_v):
                continue
            for node_w in self.nodes:
                if node_w in [node_u, node_v]:
                    continue
                if not self.is_fully_directed(node_v, node_w):
                    continue
                if self.is_fully_undirected(node_u, node_w):
                    self.remove_edge(node_u, node_w)
                    self.add_edge(Edge(node_u, node_w, Mark.Tail, Mark.ARROW))
                    changed = True
                    if verbose:
                        print(f'Rule2: Orient {node_u} --- {node_w} into {node_u} --> {node_w}.')
        return changed

    def rule3(self, verbose=False):
        changed = False
        for node_u, node_w in permutations(self.nodes, 2):
            if not self.is_fully_undirected(node_u, node_w):
                continue
            for node_v in self.nodes:
                if node_v in [node_u, node_w]:
                    continue
                if not (self.is_fully_undirected(node_u, node_v) and self.is_fully_directed(node_v, node_w)):
                    continue
                for node_x in self.nodes:
                    if node_x in [node_u, node_v, node_w]:
                        continue
                    if not (self.is_fully_undirected(node_u, node_x) and self.is_fully_directed(node_x, node_w)):
                        continue
                    self.remove_edge(node_u, node_w)
                    self.add_edge(Edge(node_u, node_w, Mark.Tail, Mark.ARROW))
                    if verbose:
                        print(f'Rule3: Orient {node_u} --- {node_w} into {node_u} --> {node_w}.')
                    changed = True
        return changed

    def rule4(self, verbose=False):
        changed = False
        for node_u, node_w in permutations(self.nodes, 2):
            if self.is_connected(node_u, node_w):
                continue
            for node_v in self.nodes:
                if node_v in [node_u, node_w]:
                    continue
                if not (self.is_fully_undirected(node_u, node_v) and self.is_fully_directed(node_v, node_w)):
                    continue
                for node_x in self.nodes:
                    if node_x in [node_u, node_v, node_w]:
                        continue
                    if not (self.is_fully_undirected(node_u, node_x) and self.is_fully_undirected(node_x, node_w)):
                        continue
                    if self.is_connected(node_v, node_x):
                        self.remove_edge(node_x, node_w)
                        self.add_edge(Edge(node_x, node_w, Mark.Tail, Mark.ARROW))
                        if verbose:
                            print(f'Rule4: Orient {node_x} --- {node_w} into {node_x} --> {node_w}.')
                        changed = True
        return changed
