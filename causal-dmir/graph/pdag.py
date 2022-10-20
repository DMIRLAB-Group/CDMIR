from . import Graph, Mark


class PDAG(Graph):
    def add_edge(self, node_u, node_v, mark_u, mark_v, overwrite=False):
        self.check_mark(mark_u, Mark.pdag_marks())
        self.check_mark(mark_v, Mark.pdag_marks())
        super().add_edge(node_u, node_v, mark_u, mark_v, overwrite=overwrite)