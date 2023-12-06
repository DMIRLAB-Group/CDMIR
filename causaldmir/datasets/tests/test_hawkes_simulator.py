from unittest import TestCase
import numpy as np
import pandas as pd
from causaldmir.datasets.simlulators import HawkesSimulator
from causaldmir.datasets.utils import erdos_renyi, generate_lag_transitions


class TestHawkesSimulator(TestCase):

    def test_INSEM_data(self):
        sample_size = 10000
        lambda_x = 1
        theta = 0.5
        lambda_e = 1
        seed = 42
        insem_data = HawkesSimulator.INSEM_data(sample_size, lambda_x, theta, lambda_e, seed)
        assert isinstance(insem_data, pd.DataFrame)
        assert insem_data.shape[1] == 3

    def test_generate_data(self):
        mu_range_str = '0,1'
        alpha_range_str = '0,1'
        n = 3
        sample_size = 30000
        out_degree_rate = 1.5
        NE_num = 40
        decay = 0.1
        seed = 42
        event_table, edge_mat, alpha, mu, events = HawkesSimulator.generate_data(
            n, mu_range_str, alpha_range_str, sample_size, out_degree_rate, NE_num, decay, seed)

        assert isinstance(event_table, pd.DataFrame)
        assert event_table.shape[1] == 3  # Three columns: seq_id, time_stamp, event_type
        assert edge_mat.shape == (n, n)
        assert alpha.shape == (n, n)
        assert mu.shape[0] == n
        assert len(events) == NE_num
