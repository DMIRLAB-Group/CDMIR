import numpy as np
import pandas as pd
import Utils

# roll back the latent variable into their measured varaible set
'''
    # roll back L into their measured set and test merge rule between MeasuredSet(L) and TC
    # we need tdata=data\L U L_Mset, where L_Mset is only two general pure measured variables of L
    # So the AllCausalCluters need to be used to select two pure measured varaibles
'''


def roll_back(L, Ora_data, data, LatentIndex, AllCausalCluters, PureClusters):
    """
    DEBUG+++++++++
    """
    print('roll back to :', L)

    MeasuredIndexs = list(Ora_data.columns)

    Lkey = list(LatentIndex.keys())

    if L not in Lkey:
        print('There are not early learning! Beacuse LatentIndex is not this index, ', L)
        # raise Exception
        exit(-1)

    LMeasured = LatentIndex[L]

    LPure = set()
    for S in PureClusters:
        if set(S) < set(LMeasured) or set(S) == set(LMeasured):
            LPure = LPure.union(set(S))

    LMeasured = set(LMeasured) - set(LPure)

    LGPure = set()

    '''
    DEBUG+++++++++
    '''
    print(LPure)
    print(LGPure)

    Lobserved = []
    LL = []

    for s in LPure:
        if s in MeasuredIndexs:
            Lobserved.append(s)
        else:
            LL.append(s)

    for s in LGPure:
        if s in MeasuredIndexs:
            Lobserved.append(s)
        else:
            LL.append(s)

    # roll_back LL into observed variables

    LLObserved = []

    for i in LL:
        OL = GetObservedForLatent(i, LatentIndex, MeasuredIndexs)
        LLObserved.append(OL)

    roll_back_dataset_indexs = Lobserved + LLObserved

    data_indexs = list(data.columns)

    rollback_data = pd.concat([data, Ora_data[roll_back_dataset_indexs]], axis=1)
    del rollback_data[L]

    '''
    DEBUG+++++++++
    '''
    print(rollback_data.columns)

    Lmset = []
    for i in range(0, 2):
        Lmset.append(roll_back_dataset_indexs[i])

    return rollback_data, Lmset


def GetObservedForLatent(L, LatentIndex, ObservedIndex):
    Lkey = list(LatentIndex.keys())

    L2 = L

    while L2 in Lkey:
        Temp = LatentIndex[L2]
        for i in Temp:
            if i in ObservedIndex:
                L2 = i
                break
        L2 = Temp[0]

    return L2


# updata the actived data set according to the current learned cluster, i.e., introducing latent varaible
def UpdataData(Merge_Results, L_index, LatentIndex, data):
    indexs = list(data.columns)

    MergeCluster = Merge_Results[0]  # for C = C1 U C2
    EarlyLearningImpureClusters = Merge_Results[1]  # e.g., [x1->x2->x3], which we will not introduce latent to this cluster according to remark 1
    EarlyLearningRemoveClusters = Merge_Results[2]  # early latent cluster, e.g., [L1, x2], where L1=[x1, x3] and L1->{x1 x2 x3} in ground truth
    IntroduceLatent_PureClusters = Merge_Results[3]  # for pure cluster but not be merged into another cluster
    RemainingVariables = Merge_Results[4]

    # remove early learning
    EarlyLearningLatent = []  # go to the next procedure
    Removes = set()

    for C in EarlyLearningRemoveClusters:
        EarlyLearningLatent.append(C[0])
        C.remove(C[0])
        Removes = Removes.union(set(C))

    indexs = set(indexs) - set(Removes)

    # Introducing latent variable for MergeCluster and IntroduceLatent_PureClusters

    Latent = []
    Lname = []
    for S in MergeCluster:
        # Introducing latent variables for S
        str1 = 'L' + str(L_index)
        L_index += 1
        LatentIndex[str1] = S

        # get the new surrogate for L_s
        Latent.append(list(S)[0])
        Lname.append(str1)
        indexs = set(indexs) - set(S)

    for S in IntroduceLatent_PureClusters:
        # Introducing latent variables for S
        str1 = 'L' + str(L_index)
        L_index += 1
        LatentIndex[str1] = S

        # get the new surrogate for L_s
        Latent.append(S[0])
        Lname.append(str1)
        indexs = set(indexs) - set(S)

    # add new latent variable in data

    tdata = data[Latent]

    t2data = data[list(indexs)]

    updateData = pd.concat([tdata, t2data], axis=1)

    upindexs = list(indexs) + Lname

    updateData.columns = upindexs

    return updateData, L_index, LatentIndex


# For orientation phase, causal discovery in measurment model
def GetMeasurementModel():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
