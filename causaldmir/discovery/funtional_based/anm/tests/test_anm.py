from causaldmir.discovery.funtional_based.anm.ANM import ANM
import numpy as np
import pandas as pd
from unittest import TestCase


class TestANM(TestCase):

    def test_anm_using_dataset(self):
        """abalone dataset from paper"""
        data = pd.read_csv("./dataset/abalone.data", sep=",", header=None)
        data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
                        'Shell weight', 'Rings']
        X = data['Rings'].values.reshape(-1, 1)
        Y = data['Length'].values.reshape(-1, 1)

        anm = ANM()
        p_value_forward, p_value_backward = anm.cause_or_effect(X, Y)
        assert p_value_forward > p_value_backward

    def test_anm_using_simulation(self):
        # simulated data y = 5 * exp(x) + e
        S = np.loadtxt('./dataset/anm_simulation.txt', delimiter=',')
        X = S[:, 0].reshape(-1, 1)
        Y = S[:, 1].reshape(-1, 1)
        anm = ANM()
        p_value_forward, p_value_backward = anm.cause_or_effect(X, Y)
        assert p_value_forward > p_value_backward

        # simulated data y = 3^x + e
        np.random.seed(1000)
        X = np.random.uniform(size=10000)
        Y = np.power(3, X) + np.random.uniform(size=10000)
        anm = ANM()
        p_value_forward, p_value_backward = anm.cause_or_effect(X, Y)
        assert p_value_forward > p_value_backward
