from typing import Iterable

from causaldmir.graph import DiGraph

from .basic_independence import ConditionalIndependentTest


class Dsep(ConditionalIndependentTest):
    def __init__(self, data, var_names=None, true_graph: DiGraph = None):
        super().__init__(data, var_names=var_names)
        assert true_graph is not None
        self.true_graph = true_graph

    def cal_stats(self, x_id: int, y_id: int, z_ids: Iterable[int] = None):
        zs_name = [self.var_names[z_id] for z_id in z_ids]
        if self.true_graph.is_d_separate(self.var_names[x_id], self.var_names[y_id], zs_name):
            return 1.0, 1.0
        else:
            return 0.0, 0.0
