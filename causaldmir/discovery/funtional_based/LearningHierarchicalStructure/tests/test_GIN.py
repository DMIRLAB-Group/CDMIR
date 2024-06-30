from causaldmir.discovery.funtional_based.LearningHierarchicalStructure.GIN2 import GIN
import causaldmir.discovery.funtional_based.LearningHierarchicalStructure.Paper_simulation as SimulationData
import numpy as np
import pandas as pd
from unittest import TestCase


class TestGIN(TestCase):

    def test_gin_using_simulation(self):
        # simulated data y = 3^x + e
        np.random.seed(1000)
        data = SimulationData.CaseI(10000)
        X = ['X1', 'X2', 'x3']
        Z = ['X4', 'x5']

        result = GIN(X,Z,data)

        assert result == True, "GIN函数应该返回True，但返回了False"
