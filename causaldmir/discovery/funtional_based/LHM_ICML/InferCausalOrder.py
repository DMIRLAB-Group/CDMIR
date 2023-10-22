import numpy as np
import pandas as pd
import itertools
import networkx as nx
import warnings
import Utils
from itertools import combinations
import GIN2 as GINS


#Paper.Algorithm 5 LocallyInferCausalOrder
#Input:
#   Impure:identified Impure cluster
#   LatentIndex:Record each variable and their direct child variables
#   Ora_data:Original data
#   Record: recording same level informatin
#output:
#   Order:Infered causal order
def InferOrderImpure(Impure,LatentIndex,Ora_data,Record):
    Order=[]
    for C in Impure:
        C2=InferOrder(C,LatentIndex,Ora_data,Record)
        Order.append(C2)
    return Order


# infer causal order according to Paper.Proposition 4
#Input:
#   C: impure cluster
#   LatentIndex:Record each variable and their direct child variables
#   Ora_data:Original data
#   Record: recording same level informatin
#output:
#   Infered causal order of C
def InferOrder(C,LatentIndex,Ora_data,Record):

    Adj=GetSibling(C,LatentIndex)
    if len(Adj) ==0:
        print('there are some wrong in select sibling!')
        return C
    Adj2=[]
    if len(Adj) < 2:
        Adj2=GetAdj(C,Record,LatentIndex)
    print('Inference causal order with adj',C,Adj,Adj2)
    ObservedIndex=GetIndexByObserved(LatentIndex)
    Obserkeys=list(ObservedIndex.keys())

    dataIndex=[]
    for i in C:
        dataIndex.append(i)
    for i in Adj:
        dataIndex.append(i)
    for i in Adj2:
        if i not in dataIndex:
            dataIndex.append(i)
    data=Ora_data.copy()
    for root in C:
        X=[]
        Z=[]
        X=C.copy()
        X.append(Adj[0])
        if len(Adj) >1:
            Z.append(Adj[1])
        else:
            for i in Adj2:
                if i not in Adj:
                    Z.append(i)
                    break
        Z.append(root)
        #transfer latent to obserd index
        obserdX=[]
        obserdZ=[]
        for i in X:
            if i in Obserkeys:
                obserdX.append(ObservedIndex[i][0])
            else:
                obserdX.append(i)
        for i in range(0,len(Z)):
            i_z=Z[i]
            if i_z in Obserkeys:
                if i == len(Z)-1: # root must select different observed
                    obserdZ.append(ObservedIndex[i_z][1])
                else:
                    obserdZ.append(ObservedIndex[i_z][0])
            else:
                obserdZ.append(i_z)
        print('Transfer latent to observed as',X,Z)
        print(obserdX,obserdZ)

        #based on hsic pval
        if GIN(obserdX,obserdZ,data):
            order=[]
            order.append(root)
            for i in C:
                if i ==root:
                    continue
                order.append(i)
            return order

    return C


#get the Sibling of C according to LatentIndex
def GetSibling(C,LatentIndex):
    key = LatentIndex.keys()
    Adj=[]
    for i in key:
        clu=LatentIndex[i]
        if set(C) < set(clu):
            print(C,clu)
            Adj=clu.copy()
            for j in C:
                Adj.remove(j)
            return Adj

    return Adj

#get the adjcent of C
def GetAdj(C,Record,LatentIndex):
    Adj=[]
    for level in Record:
        flag =True
        for i in C:  #Find C in a Level
            if i not in level:
                flag =False
                break
        if flag:
            Adj=level
            for j in C:
                Adj.remove(j)
            break
    # if C no in level
    Lname=''
    if len(Adj)==0:
        key = LatentIndex.keys()
        for i in key:
            clu =LatentIndex[i]
            if set(C) <= set(clu):
                Lname=i

        for i in key:
            clu =LatentIndex[i]
            if Lname in clu:
                Adj=clu
                Adj.remove(Lname)
    return Adj

#Utils function be noted in other filed
def GIN(X,Z,data,alph=0.01):
    X=list(X)
    Z=list(Z)
    m=GINS.GIN(X,Z,data,alph)
    #m=GINS.FisherGIN(X,Z,data,alph)
    return m

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
