import numpy as np
import pandas as pd
import itertools
import networkx as nx
import warnings
import Utils
from itertools import combinations
import GIN2 as GINS
import DeterminingLatentVariables as DLV
warnings.filterwarnings("ignore")


# Paper.Algorithm 3.DetermineLatentVariables according to Proposition 2 (Merging Rules) and Corollary 1
#Input:
#   Cluster:current clusters learned by Algorithm 2
#   data:active data
#   Ora_data:Original data
#   Pure,Impure: identified Pure/Impure clusters
#   LatentIndex: recording Latent with their direct childrens
#   Record: Record the same level of information that has been learned
#output:
#   LatentIndex: recording Latent with their direct childrens
#   Pure,Impure: update Pure and Impure List
#   Cluster:Merged clusters
def MergingRule(Cluster,data,Ora_data,Pure,Impure,LatentIndex,Record):
    Pure,Impure=DLV.GetPure(Cluster,data,Pure,Impure,LatentIndex,Ora_data)
    #Notic that the R1 rule be used in the findcluster stage, we only need to test Rule 2,3
    print('Get Pure and Impure as follow:',Pure,Impure)

    print('stage: merger basic')
    for C1 in Cluster:
        if Utils.JudgeCon(C1,Impure):
            TempClu=Cluster.copy()
            TempClu.remove(C1)
            for C2 in TempClu:
                Cluster,Pure,Impure=MergeRule_basic(C1,C2,Pure,Impure,Cluster,data)

    print('stage: early learning_1, inner cluster')
    LatentNames=LatentIndex.keys()
    for C1 in Cluster:
        if JudgePeerLevel(C1,LatentIndex,Record):  # peer level latent can not be merger in early learning
            for L in C1:
                C2=C1.copy()
                C2.remove(L)
                if L in LatentNames:
                    Cluster,LatentIndex,Pure,Impure=CorollaryMerge(L,C2,LatentIndex,Impure,Pure,Cluster,data,Ora_data)

    print('stage: early learning_2, remain latent')
    A=list(data.columns)
    Lists=Getlist(Cluster)
    B=[]
    for i in A:
        if (i not in Lists) and (i in LatentNames):
            B.append(i)
    for C2 in Cluster:
        for L in B:
            print(' remain latent test:',L,C2)
            Cluster,LatentIndex,Pure,Impure=CorollaryMerge(L,C2,LatentIndex,Impure,Pure,Cluster,data,Ora_data)

    return LatentIndex,Impure,Pure,Cluster



#Test whether the latent variable in C appears in the same layer
#input:
#   C: Clustering to test whether the latent variable in C appears in the same layer
#   LatentIndex: Latent with the direct childrens
#   record_all: Record the same level of information that has been learned
#output:
#   True: there exist at least two latent variable in C appears in the same layer
def JudgePeerLevel(C,LatentIndex,record_all):
    LatentNames=LatentIndex.keys()
    LatentC=[]
    for i in C:
        if i in LatentNames:
            LatentC.append(i)
    if len(LatentC) <=1:
        return True
    for x in LatentC:
        for records in record_all:
            if x in records:
                for y in LatentC:
                    if x == y:
                        continue
                    if y in records and len(records) > 2:
                        return False
    return True




#test Mergering Rule (Corollary 1)
#Input:
#   C1: the set of children of L
#   C2: C2 be a new cluster
#   L: parent of C1
#   LatentIndex:recording Latent with their direct childrens
#   Impure,Pure:identified Pure/Impure clusters
#   Cluster:current clusters learned by Algorithm 2
#   tdata:Contains updated data for C1 index
#output:
#   Cluster:update cluster
#   LatentIndex:update latentindex
#   Pure,Impure:update pure and impure list
def Coro_MergeRule(C1,C2,L,LatentIndex,Impure,Pure,Cluster,tdata):
    data = tdata.copy()
    A=list(data.columns)
    for i in C1:
        A.remove(i)
    for i in C2:
        A.remove(i)

    if Utils.JudgeCon(C2,Pure):
        for i in Pure:
            if set(i)==set(C2):
                C2=list(i)

    X=[]
    Z=[]

    if Utils.JudgeCon(C1,Pure) and Utils.JudgeCon(C2,Pure):
        X.append(C1[0])
        X.append(C2[0])

        Z.append(C1[1])
        if len(C2) >=2:
            Z.append(C2[1])
    elif Utils.JudgeCon(C1,Pure) and Utils.JudgeCon(C2,Impure):
        Z=A.copy()
        X.append(C1[0])
        X.append(C2[0])
        for i in range(1,len(C1)):
            X.append(C1[i])
    elif Utils.JudgeCon(C1,Impure) and Utils.JudgeCon(C2,Pure):
        Z=A.copy()
        X.append(C1[0])
        X.append(C2[0])
        if len(C2) >=2:
            Z.append(C2[1])
    elif Utils.JudgeCon(C1,Impure) and Utils.JudgeCon(C2,Impure):
        Z=A.copy()
        X.append(C1[0])
        X.append(C2[0])
    else:
        print(' there are some cluster no be contained in Pure or Impure')

    if GIN(X,Z,data):#merge C1 and C2
        LatentIndex,Cluster,Pure,Impure=MergerEarlyLearning(C1,C2,L,LatentIndex,Cluster,Pure,Impure)

    return Cluster,LatentIndex,Pure,Impure


#test Mergering Rule (Corollary 1) and update testing data
# L is a latent vairbale that has been introduced in the previous iterations
#others follow the Coro_MergeRule Note
def CorollaryMerge(L,C2,LatentIndex,Impure,Pure,Cluster,data,Ora_data):
    key=LatentIndex.keys()
    key = list(key)
    if L in key:
        C1=LatentIndex[L]

    indexs = list(data.columns)
    LatentNames=list(LatentIndex.keys())

    tdata = data.copy()
    del tdata[L]
    for i in C1:
        if i not in LatentNames:
            tdata[i]=Ora_data[i]
        else:
            dex = LatentIndex[i][0]
            while dex in LatentNames:
                dex = LatentIndex[dex][0]
            tdata[i]=Ora_data[dex]
    Cluster,LatentIndex,Pure,Impure = Coro_MergeRule(C1,C2,L,LatentIndex,Impure,Pure,Cluster,tdata)
    return Cluster,LatentIndex,Pure,Impure



#test the Merger Rule R2-R3 for C1 and C2
#C1 is impure
def MergeRule_basic(C1,C2,Pure,Impure,Cluster,data):
    A=list(data.columns)
    for i in C1:
        A.remove(i)
    for i in C2:
        A.remove(i)

    X=[]
    Z=[]

    if Utils.JudgeCon(C2,Impure):#C2 is impure: Proposition R3
        Z=A.copy()
        X.append(C1[0])
        X.append(C2[0])
    else:                        #C2 is pure:  Proposition R2
        X.append(C1[0])
        X.append(C2[0])
        for i in range(1,len(C1)):
            Z.append(C1[i])
        for i in range(1,len(C2)):
            Z.append(C2[i])

    if GIN(X,Z,data):
        Cluster,Pure,Impure=MergerTwoCluster(C1,C2,Cluster,Pure,Impure)

    return Cluster,Pure,Impure

#merger two cluster in Corollary 1 situation, update LatentIndex,Cluster,Pure,Impure
#C1 is eary learning, index is L, need to update the Latentindex[L]
def MergerEarlyLearning(C1,C2,L,LatentIndex,Cluster,Pure,Impure):
    print('Merge eary learning!',C1,C2,L)
    #remove C2 from Cluster
    RMC2=C2
    for clu in Cluster:
        if len(clu)-1 == len(C2): #inner eary learning
            if (set(C2) < set(clu)) and (L in clu):
                RMC2=clu
        elif set(C2) == set(clu): #remain eary learning
            RMC2=clu

    if RMC2 in Cluster:
        Cluster.remove(RMC2)
    else:
        print('Already remove! there may by some test wrong')
        return LatentIndex,Cluster,Pure,Impure

    #Add new Purecluster
    PureC=[]
    PureC.append(C1[0])
    PureC.append(C2[0])
    for t in range(1,len(C1)):
        PureC.append(C1[t])
    for t in range(1,len(C2)):
        PureC.append(C2[t])
    Pure.append(PureC)

    #update LatentIndex[L]
    Temp=C1.copy()
    for j in C2:
        if j not in Temp:
            Temp.append(j)
    LatentIndex[L]=Temp
    return LatentIndex,Cluster,Pure,Impure

#Merger Cluster C1 and C2 and update Cluster,Pure/Impure
def MergerTwoCluster(C1,C2,Cluster,Pure,Impure):
    print('Merge two cluster',C1,C2)
    if C1 not in Cluster or C2 not in Cluster:
        print('there are some error!')
        exit(-1)

    #remove C1 and C2 from Cluster
    Cluster.remove(C1)
    Cluster.remove(C2)

    #add the pure record
    PureC=[]
    PureC.append(C1[0])
    PureC.append(C2[0])
    for i in range(1,len(C1)):
        PureC.append(C1[i])
    for i in range(1,len(C2)):
        PureC.append(C2[i])
    Pure.append(PureC)

    #add the merger new cluster
    TC=C1.copy()
    for j in C2:
        if j not in TC:
            TC.append(j)

    Cluster.append(TC)

    return Cluster,Pure,Impure


#tool function
def Getlist(r):
    a=[]
    for i in r:
        for j in i:
            a.append(j)
    return a

def GIN(X,Z,data,alph=0.01):
    X=list(X)
    Z=list(Z)
    m=GINS.GIN(X,Z,data,alph)
    #m=GINS.FisherGIN(X,Z,data,alph)
    return m

