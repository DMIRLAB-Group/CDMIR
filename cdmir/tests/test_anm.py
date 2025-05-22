from unittest import TestCase

import numpy as np
import pandas as pd

from cdmir.discovery.funtional_based.anm.ANM import ANM


class TestANM(TestCase):

    def test_anm_using_simulation(self):
        # simulated data y = 3^x + e
        np.random.seed(2025)
        X = np.random.uniform(size=10000)
        Y = np.power(X, 3) + np.random.uniform(size=10000)
        anm = ANM()
        nonindepscore_forward, nonindepscore_backward = anm.cause_or_effect(X, Y)

        assert nonindepscore_forward < nonindepscore_backward
