import numpy as np
import pandas as pd
import itertools
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import Utils
import SimulationData as SD



#Paper.Algorithm 4 UpdateActiveData
#input:
#   Cluster:Currently learned cluster
#   L_index:Number of learned latent variable
#   data:active data
#   LatentIndex:Record each variable and their direct child variables
#   Pure,Impure: identified Pure/Impure list
#output:
#   A:index of updated data
#   tdata:updated data
#   L_index:update number of latent variable
#   LatentIndex: update LatentIndex
def UpdateLatent(Cluster,L_index,data,LatentIndex,Pure,Impure):
    A = list(data.columns)
    Latent_index=[]
    Latent=[]

    for clu in Cluster:
        #to set the latent index with observed varaible and introduct new latent variable name
        clu = GetPureIndex(clu.copy(),Pure,Impure,LatentIndex)
        Latent.append(clu[0])
        str1='L'+str(L_index)
        Latent_index.append([str1,clu[0]])
        L_index+=1
        #update latentIndex
        LatentIndex[str1]=clu
        for k in clu:#A\S
            if k in A:
                A.remove(k)
    #remove the variable be merge in early learning
    Lkey=list(LatentIndex.keys())
    for i in Lkey:
        C = LatentIndex[i]
        for j in C:
            if j in A:
                A.remove(j)
    #add new latent variable in A
    for i in Latent:
        A.append(i)
    tdata=data[A]
    #update the latent index with new latent name
    for clu in Latent_index:
        a = clu[0]
        b=clu[1]
        ind=A.index(b)
        A[ind]=a
    tdata.columns=A
    return A,tdata,L_index,LatentIndex

#get the Index according to Pure list
def GetPureIndex(clu,Pure,Impure,LatentIndex):
    for i in Pure:
        if set(clu) == set(i):
            clu=i
    return clu