import numpy as np
import pandas as pd
from numpy.random import normal
from scipy import stats
import random
from causaldmir.discovery.score_based import GES
from causaldmir.utils.local_score import BICScore


random.seed(3407)
np.random.seed(3407)
sample_size = 100000
X1 = normal(size=(sample_size, 1))
X2 = X1 + normal(size=(sample_size, 1))
X3 = X1 + normal(size=(sample_size, 1))
X4 = X2 + X3 + normal(size=(sample_size, 1))
X = np.hstack((X1, X2, X3, X4))
# X = stats.zscore(X, ddof=1, axis=0)


score_function = BICScore(data=X, lambda_value=2)
s = score_function(0, [1])

ges = GES(score_function)
ges.fit()
print(ges.get_causal_graph())

print(1)