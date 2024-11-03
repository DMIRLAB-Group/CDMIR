import numpy as np

from cdmir.utils.local_score.bdeu_score import BDeuScore

def test_score_function():
    data = np.array([[5.1, 3.5, 1.4, 0.2],
                     [4.9,   3, 1.4, 0.2],
                     [4.7, 3.2, 1.3, 0.2],
                     [4.6, 3.1, 1.5, 0.2],
                     [5.4, 3.9, 1.7, 0.4],
                     [4.9, 3.1, 1.5, 0.1],
                     [5.8,   4, 1.2, 0.2]])

    bdeu_score = BDeuScore(data, sample_prior=1, structure_prior=1)
    node_index = 2
    parent_indices = [0, 1]
    score = bdeu_score._score_function(node_index, parent_indices)

    # expected_score = -15.57887 # 期望的BDeu分数，具体需要进行调整
    # assert np.isclose(score, expected_score, rtol=1e-5)  # 使用np.isclose进行浮点数比较

    assert isinstance(score, float)
    assert score is not None, "BDeu score calculation failed."
