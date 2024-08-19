import numpy as np

from cdmir.utils.independence import FisherZ

def test_fisherz():
    np.random.seed(10)
    X = np.random.randn(300, 1)
    X_prime = np.random.randn(300, 1)
    Y = X + 0.5 * np.random.randn(300, 1)
    Z = Y + 0.5 * np.random.randn(300, 1)
    data = np.hstack((X, X_prime, Y, Z))

    f = FisherZ(data=data)
    p_value, stat = f.cal_stats(0, 3, [2])
    assert p_value > 0.01

    p_value, stat = f.cal_stats(0, 3, None)
    assert p_value < 0.01

    p_value, stat = f.cal_stats(3, 1, None)
    assert p_value > 0.01

