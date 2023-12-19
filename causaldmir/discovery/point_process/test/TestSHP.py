import numpy as np

from unittest import TestCase

from causaldmir.datasets.simlulators import HawkesSimulator
from causaldmir.utils.metrics.graph_evaluation import get_performance

from ..SHP import SHP

import logging

logging.basicConfig(level=logging.DEBUG,
            format=' %(levelname)s :: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p')

class TestSHP(TestCase):
    def test_shp_hc(self):
        event_table, real_edge_mat, real_alpha, real_mu, events = HawkesSimulator.generate_data(n = 20, sample_size=2000,
                                                                                                out_degree_rate=1.5,
                                                                                                mu_range_str="0.00005,0.0001",
                                                                                                alpha_range_str="0.5,0.7",
                                                                                                NE_num=40,decay=5,
                                                                                                seed=0)
        param_dict = {
            "decay": 5,
            "reg": 0.85,
            "time_interval": 5,
            "penalty": "BIC"
        }
        self = SHP(event_table, **param_dict)
        res_shp = self.Hill_Climb()
        likelihood, fited_alpha, fited_mu = res_shp
        res = get_performance(fited_alpha, real_edge_mat)
        print(res)

    def test_shp_no_hc(self):
        event_table, real_edge_mat, real_alpha, real_mu, events = HawkesSimulator.generate_data(n = 20, sample_size=2000,
                                                                                                out_degree_rate=1.5,
                                                                                                mu_range_str="0.00005,0.0001",
                                                                                                alpha_range_str="0.5,0.7",
                                                                                                NE_num=40,decay=5,
                                                                                                seed=0)
        param_dict = {
            "decay": 5,
            "reg": 0.85,
            "time_interval": 5,
            "penalty": "BIC"
        }
        self = SHP(event_table, **param_dict)
        res_shp = self.EM_not_HC(np.ones([self.n, self.n]) - np.eye(self.n, self.n))
        likelihood, fited_alpha, fited_mu = res_shp
        res = get_performance(fited_alpha, real_edge_mat)
        print(res)