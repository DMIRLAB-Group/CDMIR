from unittest import TestCase

from numpy import isclose, log, log1p
from numpy.random import uniform


class TestLog1p(TestCase):
    def test_case(self):
        x = uniform(-0.8, 0.8, size=(10000,))
        y1 = 0.5 * log1p(2 * x / (1 - x))
        y2 = 0.5 * log((1 + x) / (1 - x))
        assert isclose(y1, y2).all()