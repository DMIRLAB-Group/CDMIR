from causaldmir.discovery.funtional_based.anm.ANM import ANM
import numpy as np
import pandas as pd
from unittest import TestCase


class TestANM(TestCase):

    def test_anm_using_simulation(self):
        # simulated data y = 3^x + e
        np.random.seed(1000)
        X = np.random.uniform(size=10000)
        Y = np.power(3, X) + np.random.uniform(size=10000)
        anm = ANM()
        p_value_forward, p_value_backward = anm.cause_or_effect(X, Y)

        assert  p_value_forward < p_value_backward
