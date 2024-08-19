import GIN2 as GIN
import Infer_Causal_Order
import numpy as np
import pandas as pd
import Utils

def Orientation_Cluster(Ora_data, LatentIndex, PureClusters, AllCausalCluster):
    pass


def Orientation_ImpureCluster(Ora_data, LatentIndex, PureClusters, AllCausalCluster, ImpureClusters):
    ImpureOrder=[]
    for C in ImpureClusters:
        MeasuredSet=[]
        Order_latent=[]
        for i in C:
            Tc=GetMeasuredVariables(i, LatentIndex,PureClusters, AllCausalCluster)
            MeasuredSet.append(Tc)

        Order = Infer_Causal_Order.LearnCausalOrder(MeasuredSet, Ora_data)
        for clu in Order:
            tindex = MeasuredSet.index(clu)
            Order_latent.append(C[tindex])

        ImpureOrder.append(Order_latent)

    return ImpureOrder

def GetMeasuredVariables(i, LatentIndex,PureClusters, AllCausalCluster):
    Lkey = LatentIndex.keys()
    clu=LatentIndex[i]

    Mset=[]

    for C in AllCausalCluster:
        if set(C) < set(clu):
            if Utils.GeneralSetContains(C, PureClusters):
                Mset.append(C[0])
                Mset.append(C[1])
            else:
                Mset.append(C[0])
                Tclu=set(clu)-set(C)
                Mset.append(Tclu[0])

    if len(Mset) == 0:
        Mset.append(clu[0])
        Mset.append(clu([len(clu)-1]))

    MeasuredSet=[]
    for j in Mset:
        while j in Lkey:
            j=LatentIndex[j]
        MeasuredSet.append(j)

    return MeasuredSet










def main():
    pass

if __name__ == '__main__':
    main()
