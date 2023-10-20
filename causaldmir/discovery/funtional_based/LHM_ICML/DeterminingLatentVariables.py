import numpy as np
import pandas as pd
from itertools import combinations
import itertools
import GIN2 as GINS
import Utils


#Test the purity of the current clusters
#input:
#   Cluster: current clusters learned by Algorithm 2
#   data: active data
#   Pure: identified Pure clusters
#   Impure: Identified Impure clusters
#   LatentIndex
#   ora_data: Original data
#output:
#   Pure:update pure list
#   Impure: update impure list
def GetPure(Cluster,data,Pure,Impure,LatentIndex,ora_data):
    for clu in Cluster:
        if len(clu)>3:
            if Utils.JudgeCon(clu,Impure):
                continue

        flag = TestPure(clu,Cluster,data,LatentIndex,ora_data)
        if flag:
            #print('pure',c)
            Pure.append(clu)
        else:
            #print('impure',c)
            Impure.append(clu)

    return Pure,Impure



#test whether C is pure according to Paper.Lemma 1
#input:
#   C: testing cluster
#   Cluster: current clusters learned by Algorithm 2
#   data,LatentIndex,ora_data: follow GetPure() function
#output:
#   True: C is Pure; False: C is Impure
def TestPure(C,Cluster,data,LatentIndex,ora_data):
    if len(C) >2:#overlap
        return True
    #set the observed variable to pure
    flag=True
    for i in C:
        if 'x' not in i:
            flag=False
    if flag:
        return flag

    LatentName=list(LatentIndex.keys())
    ObservedIndex=GetIndexByObserved(LatentIndex)
    Clusters=Cluster.copy()
    A=list(data.columns)
    B=A.copy()
    for i in Clusters:
        for j in i:
            if j in B:
                B.remove(j)
    if C in Clusters:
        Clusters.remove(C)
    for i in B:
        Clusters.append([i])

    # test by HSIC when there are no enough variables
    if len(Clusters) <2 and len(Clusters) >0:
        X=[C[0]]
        L=Clusters[0][0]
        if L in LatentName:
            temp=GetMoreIndex(L,LatentIndex)
            pop=Clusters.pop()

            print('extend observed!',temp,Clusters)
            for c in temp:
                Clusters.append([c])
                data[c]=ora_data[c]

            #try exchange test
            X3=X.copy()
            X3.append(pop[0])
            Z3=[C[1]]
            print('try no enough GIN test!',X3,Z3)
            if GIN(X3,Z3,data):
                return True
        else:
            X.append(Clusters[0][0])
            Z=[C[1]]
            print('no enough GIN test!',X,Z)
            if GIN(X,Z,data):
                return True
            else:
                return False
    elif len(Clusters)==0:
        return True

    TestVk=[]
    for V_list in Clusters:
        Vk = V_list[0]
        Z=A.copy()
        for i in C:
            if i in Z:
                Z.remove(i)

        for i in V_list:
            if i in Z:
                Z.remove(i)

        X=list(C).copy()
        X.append(Vk)
        if GIN(X,Z,data):
            TestVk.append(Vk)
    if len(TestVk) == 0:  #C is pure
        return True


    for Vk in TestVk:
        X=[]
        Z=[]
        X.append(C[0])
        X.append(Vk)
        Z.append(C[1])
        if GIN(X,Z,data):
            return True
    for Vk in TestVk:
        X=[]
        Z=[]
        X=list(C).copy()
        X.append(Vk)

        for P in Clusters:
            if set(P)<=set(C) or set(C) <= set(P):
                continue
            if Vk in P:
                continue
            Z=[P[0]]
            flag = True
            for Vi in C:#subset of Y can not satisfy the GIN condition, then select the P
                X1=X.copy()
                X1.remove(Vi)
                Z1 = Z.copy()
                Z1.append(Vi)

                if GIN(X1,Z1,data):
                    print('subset satisfy GIN:',X1,Z1)
                    flag = False
                    break
            if not flag:
                continue

            for Vi in C:
                # if Vi in latentName,Vi = LatentName[vi][1]
                Z2=[]
                Z=list(set(Z))
                Z2=Z.copy()
                tempdata=data.copy()

                if Vi in LatentName:
                    Vi = ObservedIndex[Vi][1]
                    tempdata[Vi]=ora_data[Vi]
                if Vi not in Z2:
                    Z2.append(Vi)
                    Z2=list(set(Z2))
                print('Impure test:',X,Z2)
                if GIN(X,Z2,tempdata):
                    print('Impure test finished!',X,Z2)
                    return False

    return True


def GIN(X,Z,data,alph=0.01):
    X=list(X)
    Z=list(Z)
    m=GINS.GIN(X,Z,data,alph)
    #m=GINS.FisherGIN(X,Z,data,alph)
    return m
def FindCombination(Lists,N):
    #return itertools.permutations(Lists,N)
    return itertools.combinations(Lists,N)

#get observed index for latent variable
#return all latent variable with observed index
def GetIndexByObserved(LatentIndex):
    LatentName=list(LatentIndex.keys())
    result={}

    for i in LatentName:
        dex1 = LatentIndex[i][0]
        dex2 = LatentIndex[i][1]
        while dex1 in LatentName:
            dex1=LatentIndex[dex1][0]
        while dex2 in LatentName:
            dex2=LatentIndex[dex2][1]
        result[i]=[dex1,dex2]

    return result


#get the observed index for L
def GetMoreIndex(L,LatentIndex):
    Observed = GetIndexByObserved(LatentIndex)
    return Observed[L]


