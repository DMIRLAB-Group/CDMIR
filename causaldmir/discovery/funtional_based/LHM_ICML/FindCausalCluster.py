import numpy as np
import pandas as pd
import itertools
import networkx as nx
import warnings
import GIN2 as GINS
import Utils
from itertools import combinations
import SimulationData as SD

warnings.filterwarnings("ignore")


#According to Paper:Algorithm 2 FindGlobalCausalClusters
#input:
#   data type: DataFrame
#   Impure: already identify Impure group
#   alph: independent test alph
#output:
#  ClusterList: find cluster from data
#  Impure: update Impure list if there exist impure cluster with length >2
def FindCausalCluster(data,Impure,alph=0.01):
    indexs=list(data.columns)
    Cluster=[]
    A = indexs.copy()
    B=A.copy()
    #1-factor :  pair with pair learning for pure cluster!
    Set_P=FindCombination(B,2)
    for P in Set_P:
        tind=indexs.copy()
        for t in P:
            tind.remove(t)  #   tind= ALLdata\P
        if GIN(P,tind,data,alph):
            Cluster.append(list(P))


    TempList=Getlist(Cluster)
    for t in TempList:
        if t in B:
            B.remove(t)

    #merger overlap
    Cluster=Utils.merge_list(Cluster)
    #print(Cluster)
    #for impure(len >=3) learning
    ImpureLen=3
    while len(B)>=ImpureLen and len(indexs) >ImpureLen+1:

        Impure_Set=FindCombination(B,ImpureLen)
        for P in Impure_Set:
            tind=indexs.copy()
            for t in P:
                tind.remove(t)  #   tind= ALLdata\P
            if ImpuerTest(P,tind,data,alph):
                Cluster.append(list(P))
                Impure.append(list(P))
        TempList=Getlist(Cluster)
        for t in TempList:
            if t in B:
                B.remove(t)
        ImpureLen+=1
    print('learning cluster as follow:',Cluster)
    return Cluster,Impure




#GIN condition test
#Input:
#X and Z are index, such as X=['x1','x2'],Z=['x3']
#data: DataFrame type
#alph: independent test alph
#output:
#Boolean type: True(independent) or False(dependent)
def GIN(X,Z,data,alph=0.01):
    X=list(X)
    Z=list(Z)
    m=GINS.GIN(X,Z,data,alph)
    return m

#get combinations fron List with length=N
def FindCombination(Lists,N):
    #return itertools.permutations(Lists,N)
    return itertools.combinations(Lists,N)

#test subset of group whether satisfy GIN condition
def ImpuerTest(group,tind,data,alph):
    Set =FindCombination(group,2)
    flag = True
    for j in Set:
        if not GIN(list(j),list(tind),data,alph):
            flag = False
            break

    if flag:
        return True
    else:
        return False


#transfer List[list] to list
#example:
#input: r=[['x1','x2'],['x3','x4']]
#output:  ['x1','x2','x3','x4']
def Getlist(r):
    a=[]
    for i in r:
        for j in i:
            a.append(j)
    return a

