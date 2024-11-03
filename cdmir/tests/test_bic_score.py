import numpy as np

from cdmir.utils.local_score.bic_score import BICScore

def test_score_function():
    data = np.array([[5.1, 3.5, 1.4, 0.2],
                     [4.9, 3, 1.4, 0.2],
                     [4.7, 3.2, 1.3, 0.2],
                     [4.6, 3.1, 1.5, 0.2],
                     [5.4, 3.9, 1.7, 0.4],
                     [4.9, 3.1, 1.5, 0.1],
                     [5.8, 4, 1.2, 0.2]])

    bic_score = BICScore(data, lambda_value=1)
    node_index = 2
    parent_indices = [0, 1]
    score = bic_score._score_function(node_index, parent_indices)

    # expected_score = 10 # 期望的BIC分数，具体需要进行调整
    # assert np.isclose(score, expected_score, rtol=1e-5)  # 使用np.isclose进行浮点数比较

    assert isinstance(score, float)
    assert score is not None, "BIC score calculation failed."
    assert not np.any(np.isinf(score)), "BIC score contains infinity."
