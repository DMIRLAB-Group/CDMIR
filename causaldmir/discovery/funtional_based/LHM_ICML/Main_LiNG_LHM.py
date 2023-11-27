import numpy as np
import pandas as pd
import itertools
import networkx as nx
import warnings
import SimulationData as SD
import FindCausalCluster as FC
import MergeRule as MR
import UpdateLatentVariable as ULV
import MakeGraph
import InferCausalOrder


#Latent_Hierarchical_Model_Estimation(LHME)
#input:
#   data: DataFrame type,  sample-by-dims size
#   alph: significance level of the independence test.
#output:
#   m: the adjacency matrix of the causal structure
#   Make_graph(LatentIndex):  Plot a causal graph

def LHME(data,alph=0.01):
    indexs=list(data.columns)
    Level=[] #record each level of alg estimated
    L_index=1 #Latent Number
    LatentIndex={} #dataType: dic; record the relationship of all variables[dic:key] and their direct children[dic:value](DataType: List)
    Pure=[]
    Impure=[]
    Record=[]
    tdata=data
    while True:
        #Algorithm 2 Find Global Causal Clusters
        Cluster,Impure=FC.FindCausalCluster(tdata,Impure)
        if len(Cluster)==0:
            break

        #Algorithm 3 Determine Latent Variables
        LatentIndex,Impure,Pure,Cluster = MR.MergingRule(Cluster,tdata,data,Pure,Impure,LatentIndex,Record)

        #Algorithm 4 UpdateActiveData
        if len(Cluster)!=0:
            A,tdata,L_index,LatentIndex = ULV.UpdateLatent(Cluster,L_index,tdata,LatentIndex,Pure,Impure)
            Record.append(A)
        else:
            break

    #Algorithm 5 inference causal order in impure groupe
    Impure=InferCausalOrder.InferOrderImpure(Impure,LatentIndex,data,Record)

    #Make Graph
    for C in Impure:
        key = list(LatentIndex.keys())
        if C[0] not in key:
            LatentIndex[C[0]]=[C[1]]
        elif C[0] in key:
            TS=LatentIndex[C[0]]
            TS.append(C[1])
            LatentIndex[C[0]]=TS

    #get the adj matrix from LHM algorithm
    m=UpdateGraph(indexs,LatentIndex)

    #draw the graph!
    MakeGraph.Make_graph(LatentIndex)
    print(LatentIndex)
    return m


#get the adj matrix from LatentIndex
def UpdateGraph(obserd,LatentIndex):
    key =LatentIndex.keys()
    Variables=[]
    for i in obserd:
        Variables.append(i)
    for i in key:
        if i not in Variables:
            Variables.append(i)
    n=len(Variables)
    indexs=Variables
    matrix=pd.DataFrame(np.zeros((n,n),dtype=np.int))
    matrix.columns=indexs
    matrix.index=indexs
    for i in key:
        clu=LatentIndex[i]
        for j in clu:
            matrix[i][j]=1
    print(matrix)
    return matrix

