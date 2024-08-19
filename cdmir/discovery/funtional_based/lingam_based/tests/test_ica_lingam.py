import numpy as np

from cdmir.discovery.funtional_based import ICA_LINGAM


def test_ica_lingam():
    size = 1000

    # generate data
    x1 = np.random.uniform(size=size)
    x2 = 3 * x1 + np.random.uniform(size=size)
    x3 = 2 * x1 + np.random.uniform(size=size)
    x4 = 4 * x2 + 4 * x3 + np.random.uniform(size=size)
    mat = np.asarray([x1, x2, x3, x4]).T

    gt = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 1, 0]])

    model = ICA_LINGAM(wald_alpha=.5)
    model.fit(mat)

    assert np.all((model.get_coef() > 1) == gt)
    assert np.all((model.get_causal_graph() == 1) == gt)
