from __future__ import annotations

from math import lgamma
from typing import Iterable

from numpy import ndarray, shape, log, unique, asarray
from pandas import DataFrame

from ._base import BaseLocalScoreFunction


class BDeuScore(BaseLocalScoreFunction):

    def __init__(self, data: ndarray | DataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if not kwargs.__contains__('sample_prior'):
            self.sample_prior = 1
        else:
            self.sample_prior = kwargs["sample_prior"]
        if not kwargs.__contains__('structure_prior'):
            self.structure_prior = 1
        else:
            self.structure_prior = kwargs["structure_prior"]
        self.r_i_map = {i: len(unique(asarray(self.data[:, i]))) for i in range(shape(self.data)[1])}

    def _score_function(self, i: int, parent_i: Iterable[int]):
        # calculate the local score with BDeu for the discrete case
        #
        # INPUT
        # i: current index
        # PAi: parent indexes
        # OUTPUT:
        # local BDeu score

        parent_i = list(parent_i)
        # calculate q_{i}
        q_i = 1
        for pa in parent_i:
            q_i *= self.r_i_map[pa]

        # calculate N_{ij}
        names = ['x{}'.format(var) for var in range(shape(self.data)[1])]
        Data_pd = DataFrame(self.data, columns=names)
        parant_names = ['x{}'.format(var) for var in parent_i]
        Data_pd_group_Nij = Data_pd.groupby(parant_names)
        Nij_map = {key: len(Data_pd_group_Nij.indices.get(key)) for key in Data_pd_group_Nij.indices.keys()}
        Nij_map_keys_list = list(Nij_map.keys())

        # calculate N_{ijk}
        Nijk_map = {ij: Data_pd_group_Nij.get_group(ij).groupby('x{}'.format(i)).apply(len).reset_index() for ij in
                    Nij_map.keys()}
        for v in Nijk_map.values():
            v.columns = ['x{}'.format(i), 'times']

        BDeu_score = 0
        # first term
        vm = shape(self.data)[0] - 1
        BDeu_score += len(parent_i) * log(self.structure_prior / vm) + (vm - len(parent_i)) * log(1 - (self.structure_prior / vm))

        # second term
        for pa in range(len(Nij_map_keys_list)):
            Nij = Nij_map.get(Nij_map_keys_list[pa])
            first_term = lgamma(self.sample_prior / q_i) - lgamma(Nij + self.sample_prior / q_i)

            second_term = 0
            Nijk_list = Nijk_map.get(Nij_map_keys_list[pa])['times'].to_numpy()
            for Nijk in Nijk_list:
                second_term += lgamma(Nijk + self.sample_prior / (self.r_i_map[i] * q_i)) - lgamma(self.sample_prior / (self.r_i_map[i] * q_i))

            BDeu_score += first_term + second_term

        return BDeu_score

    def __call__(self, i: int, parent_i: Iterable[int], *args, **kwargs):
        return self._score(i, parent_i, self._score_function)

