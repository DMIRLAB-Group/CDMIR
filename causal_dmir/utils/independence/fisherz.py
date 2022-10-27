from numpy import corrcoef

from .basic_independence import Independence
from .functional import fisherz_via_corr


class FisherZ(Independence):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._num_records = dataset.shape[0]
        self._corr = corrcoef(self._dataset, rowvar=False)

    def cal_stats(self, x, y, S=None):
        return fisherz_via_corr(corr=self._corr, num_records=self._num_records, x=x, y=y, S=S)