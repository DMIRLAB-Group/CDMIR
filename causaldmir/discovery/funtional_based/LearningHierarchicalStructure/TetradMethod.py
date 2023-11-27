#-------------------------------------------------------------------------------
# Name:        模块1
# Purpose:
#
# Author:      YY
#
# Created:     13/12/2022
# Copyright:   (c) YY 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import VanishedTest as VT
import numpy as np
import pandas as pd
import itertools

'''
Function: Check whether a variable set C is a causal cluster
Parameter
    C: List, e.g., C=['x1','x2']
        causal cluster
    data: DataFrame
        observational data
    alpha: float
        signification level of independence

return
    boolean: True or False
        C is a causal cluster (True) or C is not a causal cluster (False)
'''
def CheckCausalCluster(C, data, alpha=0.01):
    CSets=itertools.combinations(C, 2)
    indexs=list(data.columns)
    indexs=set(indexs)-set(C)

    if len(indexs) < 2:
        return False

    for s in CSets:
        V_i=s[0]
        V_j=s[1]
        for s_2 in itertools.combinations(list(indexs), 2):
            V_k=s_2[0]
            V_s=s_2[1]

            flag=VT.vanishes(data[V_i],data[V_j],data[V_k],data[V_s],alpha)
            #print(V_i,V_j,V_k,V_s,flag)

            if not flag:
                return False

    return True


'''
Function: Check whether a variable set C is a pure cluster
Parameter
    C: List, e.g., C=['x1','x2']
        causal cluster with dim=2
    data: DataFrame
        observational data
    alpha: float
        signification level of independence

return
    boolean: True or False
        C is a purity (True) or C is impurity (False)
'''
def JudgePureCluster(C, data, alpha=0.01):
    #Note that if dim(C)>= 3, C must be impurity in one-factor model
    V_i=C[0]
    V_j=C[1]
    indexs=list(data.columns)
    indexs=set(indexs)-set(C)
    for s in itertools.combinations(list(indexs), 2):
        V_k=s[0]
        V_s=s[1]

        #{V_i,V_k} and {V_j,V_s} follow Tetrad Constraints, then V_i and V_j is pure
        flag=VT.vanishes(data[V_i],data[V_k],data[V_j],data[V_s],alpha)
        #print(V_i,V_j,V_k,V_s,flag)
        if flag:
            return True

    return False






import Paper_simulation as SD
def main():
    data=SD.CaseI(3000)
    C=['x3','x4']
    print(JudgePureCluster(C,data))


if __name__ == '__main__':
    main()
