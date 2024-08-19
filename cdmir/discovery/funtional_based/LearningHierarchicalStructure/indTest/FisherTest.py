import math

from scipy.stats import chi2

#independent test by Fisher'method
def FisherTest(pvals,alph=0.01):
    Fisher_Stat=0
    L = len(pvals)
    for i in range(0,L):
        if pvals[i] ==0:
            TP = 1e-05
        else:
            TP = pvals[i]

        Fisher_Stat = Fisher_Stat-2*math.log(TP)


    Fisher_pval = 1-chi2.cdf(Fisher_Stat, 2*L)

    #print(Fisher_pval)
    if Fisher_pval >alph:
        return True,Fisher_pval
    else:
        return False,Fisher_pval



