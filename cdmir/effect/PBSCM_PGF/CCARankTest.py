import numpy as np
from statsmodels.multivariate.cancorr import CanCorr
from math import log, pow
from scipy.stats import chi2


class CCARankTester:
    def __init__(self, data, alpha=0.05):

        # Centre the data
        data = data - np.mean(data, axis=0)
        self.data = np.array(data)
        self.n = data.shape[0]
        self.alpha = alpha

    # Test null hypothesis that rank is less than or equal to r
    # Return True if reject
    def test(self, pcols, qcols, r=1):

        p = len(pcols)
        q = len(qcols)
        X = self.data[:, pcols]
        Y = self.data[:, qcols]
        cca = CanCorr(X, Y, tolerance=1e-8)
        l = cca.cancorr[r:]

        testStat = 0
        for li in l:
            testStat += log(1 - pow(li, 2))
        testStat = testStat * -(self.n - 0.5*(p+q+3))

        dfreedom = (p-r) * (q-r)
        criticalValue = chi2.ppf(1-self.alpha, dfreedom)
        # print(f"testStat: {testStat}, crit: {criticalValue}")

        return testStat > criticalValue

    def test_my(self, matrix, r=1):

        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        p = 2
        q = 2

        l = S[r:]
        testStat = 0
        for li in l:
            testStat += log(1 - pow(li, 2))
        # print(S, pow(l[0], 2), testStat)
        testStat = testStat * -(self.n - 0.5*(p+q+3))

        dfreedom = (p-r) * (q-r)
        criticalValue = chi2.ppf(1-self.alpha, dfreedom)
        # print(f"testStat: {testStat}, crit: {criticalValue}")

        return testStat > criticalValue, [testStat, S[1], pow(l[0], 2)]


    def prob(self, pcols, qcols, r=1):
        p = len(pcols)
        q = len(qcols)
        X = self.data[:, pcols]
        Y = self.data[:, qcols]

        cca = CanCorr(X, Y)
        l = cca.cancorr[r:]

        testStat = 0
        for li in l:
            testStat += log(1 - pow(li, 2))
        testStat = testStat * -(self.n - 0.5*(p+q+3))

        dfreedom = (p-r) * (q-r)
        criticalValue = chi2.ppf(1-self.alpha, dfreedom)
        #print(f"testStat: {testStat}, crit: {criticalValue}")

        return 1-chi2.cdf(testStat, dfreedom)




