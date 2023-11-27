import logging
from unittest import TestCase


import utils
from linear_ent import run
from nonlinear_ent import run as run_nonlinear

logging.basicConfig(level=logging.DEBUG,
                    format=' %(levelname)s :: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


class TestEntropyBasedDiscovery(TestCase):

    def test_linear_entropy(self):
        # n = sample size
        # d = num of nodes
        # s0 = num of edges
        n, d, s0, graph_type, sem_type = 1000, 5, 10, 'ER', 'uniform'
        lambda1 = 1e-3

        utils.set_seed(123)
        acc = run(n=n, d=d, s0=s0, graph_type=graph_type, sem_type=sem_type, lamda1=lambda1,
            w_ranges=((-2.0, -0.5), (0.5, 2.0)), scale_low=1., scale_high=1.)

        print(acc)

    def test_nonlinear_entropy(self):
        # n = sample size
        # d = num of nodes
        # s0 = num of edges
        n, d, s0, graph_type, sem_type = 1000, 5, 10, 'ER', 'uniform'
        lambda1, lambda2 = 1e-3, 1e-3

        utils.set_seed(123)
        acc = run_nonlinear(n=n, d=d, s0=s0, graph_type=graph_type, noise_dist=sem_type, sem_type='mim',
        lambda1=lambda1, lambda2=lambda2, low=1., high=1.)

        print(acc)
