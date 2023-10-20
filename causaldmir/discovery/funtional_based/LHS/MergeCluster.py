import VanishedTest as VT
import numpy as np
import pandas as pd
import itertools
import Utils
import UpdataData





def MergeCausalCluster(LearnedClusters, PureClusters, ImpureClusters, AllCausalCluster, GeneralPureClusters, LatentIndex, data, Ora_data, alpha=0.01):
    '''
    Function: Merge the learned causal clusters that share one latent parent in the current finding phase
    Parameter
        LearnedClusters: List
            the causal cluster learned by current finding phase
        PureCluster: List
            Record the Pure causal cluster, where for each cluster C =[V_i, V_j], V_i _|_ V_j | L_p
        ImpureClusters: List
            Record the Impure causal cluster
        AllCausalCluster: List
            Record all causal cluster learned in previous finding phase
        GeneralPureCluster: List
            Record the merger cluster, e.g., L1-> [['x1','x2'],['x3','x4']], where  ['x1','x2'] and ['x3','x4'] can be treated as a general purity
        data: DataFrame
            current actived dataset
        Ora_data: DataFrame
            All observed variable data set
        LatentIndex: dic
            Record the relations between latent an their measured set
    Return
        Update All Parameter except for data, Ora_data

    '''





    # We will divide LearnedCluster into four part, it is clear to update data and record perporty
    MergeCluster=[] # for C = C1 U C2
    EarlyLearningImpureClusters=[] # e.g., [x1->x2->x3], which we will not introduce latent to this cluster according to remark 1
    EarlyLearningRemoveClusters=[] # early latent cluster, e.g., [L1, x2], where L1=[x1, x3] and L1->{x1 x2 x3} in ground truth
    IntroduceLatent_PureClusters=[] # for pure cluster but not be merged into another cluster
    indexs=list(data.columns)
    RemainingVariables=indexs.copy()
    for c in LearnedClusters:
        RemainingVariables=set(RemainingVariables)-set(c)


    # if not any cluster is learned for actived data, introduce latent for current data
    if len(LearnedClusters) == 0:
        indexs=list(data.columns)
        #LearnedClusters.append(indexs) # measurement model, may learning learning
        ImpureClusters.append(indexs) # due to measurement model, which is early learning
        RemainingVariables=[]

    '''
    DEBUG++++++++++++++++++++
    '''
    print('RemainingVariables:', RemainingVariables)

    InnerELClusters={}
    OutterELClusters={}

    '''
    DEBUG++++++++++++++++++++
    '''
    print('Early Learning Detecting !')
    #record early learing cluster
    for S in LearnedClusters:
        if Utils.GeneralSetContains(S, ImpureClusters):
            L,Flag = OutterEarlyLearning(RemainingVariables, S, data, Ora_data, LatentIndex, PureClusters, AllCausalCluster, alpha)
            if Flag:
                OutterELClusters[L]=S
                continue
            L,Flag = Inner_EarlyLearning(S, data, Ora_data, LatentIndex, PureClusters, AllCausalCluster, alpha)
            if Flag:
                InnerELClusters[L]=S
##
##            L,Flag = Inner_EarlyLearning(S, data, Ora_data, LatentIndex, PureClusters, AllCausalCluster, alpha)
##            if Flag:
##                InnerELClusters[L]=S
##                continue
##            L,Flag = OutterEarlyLearning(RemainingVariables, S, data, Ora_data, LatentIndex, PureClusters, AllCausalCluster, alpha)
##            if Flag:
##                OutterELClusters[L]=S
        else:
            L,Flag = Inner_EarlyLearning(S, data, Ora_data, LatentIndex, PureClusters, AllCausalCluster, alpha)
            if Flag:
                InnerELClusters[L]=S

    print('Early Learned: ', InnerELClusters, OutterELClusters)



    #check the merge rules for any pair cluster
    while len(LearnedClusters) >= 1:
        C1=LearnedClusters.pop()
        HopetoMerge=[]
        for C2 in LearnedClusters: # test whether C1 and C2 share one latent parents
            flag=False
            if Utils.GeneralSetContains(C1, PureClusters) and Utils.GeneralSetContains(C2, PureClusters):
                print('Pure with Pure, Continue!')
                continue
            elif Utils.GeneralSetContains(C1, ImpureClusters) and Utils.GeneralSetContains(C2, ImpureClusters):
                flag = TestImpureWithImpure(C1, C2, data, alpha)
            elif Utils.GeneralSetContains(C1, PureClusters) and Utils.GeneralSetContains(C2, ImpureClusters):
                flag = TestPureWithImpure(C1, C2, data, alpha)
            elif Utils.GeneralSetContains(C1, ImpureClusters) and Utils.GeneralSetContains(C2, PureClusters):
                flag = TestPureWithImpure(C2, C1, data, alpha)
            else:
                print('something error ! Because C1 or C2 are not in the record list, such as pure/impure cluster, generalpurity cluster...')

            if flag:
                HopetoMerge.append(C2)
                flag=False

        #只有C1 C2 C3..Cn都是share同一个early learning才会合并并且early learn，否则，先early learn更新，再合并聚类
        EarlyLatentSet=[]
        if len(HopetoMerge) >0:
            TC=set(C1)
            #add early learned record analysis, if and only if one cluster has early learning in the merged clusters
            EarlyLatent=''
            if Utils.GeneralDicContains(C1, InnerELClusters):
                EarlyLatent=Utils.GetEarlyLearningByDic(C1, InnerELClusters)
            elif Utils.GeneralDicContains(C1, OutterELClusters):
                EarlyLatent=Utils.GetEarlyLearningByDic(C1, OutterELClusters)

            EarlyLatentSet.append(EarlyLatent)

            for C2 in HopetoMerge:
                EarlyLatent=''
                if Utils.GeneralDicContains(C2, InnerELClusters):
                    EarlyLatent=Utils.GetEarlyLearningByDic(C2, InnerELClusters)
                elif Utils.GeneralDicContains(C2, OutterELClusters):
                    EarlyLatent=Utils.GetEarlyLearningByDic(C2, OutterELClusters)
                EarlyLatentSet.append(EarlyLatent)

                if C2 in LearnedClusters:
                    LearnedClusters.remove(C2)
            #The above detect all early in C_i, if C_i and C_j share a common latent early learn, merge into the early latent.
            #otherwise, update the early learning cluster into their early introduced latent
            #Function EarlyLearningUpdata()
            THopetoMerge = HopetoMerge.copy()
            THopetoMerge.insert(0, C1)


            for earlyL in range(0,len(EarlyLatentSet)):
                EarlyLatent = EarlyLatentSet[earlyL]
                #update the early learning variable
                if EarlyLatent != '':
                    MC = THopetoMerge[earlyL]
                    LatentIndex, AllCausalCluster = UpdateLatentIndex(EarlyLatent, MC, LatentIndex, AllCausalCluster)
                    UpTC = list(MC.copy())
                    if EarlyLatent in UpTC:
                        UpTC.remove(EarlyLatent)
                    UpTC.insert(0,EarlyLatent)
                    EarlyLearningRemoveClusters.append(UpTC)
                    GeneralPureClusters.append(list(set(TC)-set(EarlyLatent)))
                    THopetoMerge[earlyL]=[EarlyLatent]
                    if EarlyLatent in RemainingVariables:
                        RemainingVariables.remove(EarlyLatent)


            TMergeC=set()
            for C_i in THopetoMerge:
                TMergeC = TMergeC.union(set(C_i))

            MergeCluster.append(list(TMergeC))


##
##            if EarlyLatent == '':
##                GeneralPureClusters.append(TC)
##                MergeCluster.append(TC)
##            else:
##
##                LatentIndex, AllCausalCluster = UpdateLatentIndex(EarlyLatent, TC, LatentIndex, AllCausalCluster)
##
##                UpTC = list(TC.copy())
##                if EarlyLatent in UpTC:
##                    UpTC.remove(EarlyLatent)
##                UpTC.insert(0,EarlyLatent)
##
##                EarlyLearningRemoveClusters.append(UpTC)
##                GeneralPureClusters.append(list(set(TC)-set(EarlyLatent)))

        else:
            #add early learned record analysis, for a single cluster, which is early learning
            if Utils.GeneralDicContains(C1, InnerELClusters):

                EarlyLatent=Utils.GetEarlyLearningByDic(C1, InnerELClusters)

                UpTC = list(C1.copy())
                UpTC.remove(EarlyLatent)
                UpTC.insert(0,EarlyLatent)

                EarlyLearningRemoveClusters.append(UpTC)

                LatentIndex, AllCausalCluster = UpdateLatentIndex(EarlyLatent, C1, LatentIndex, AllCausalCluster)

            elif Utils.GeneralDicContains(C1, OutterELClusters):

                EarlyLatent=Utils.GetEarlyLearningByDic(C1, OutterELClusters)

                UpTC = list(C1.copy())

                UpTC.insert(0, EarlyLatent)

                EarlyLearningRemoveClusters.append(UpTC)

                LatentIndex, AllCausalCluster = UpdateLatentIndex(EarlyLatent, C1, LatentIndex, AllCausalCluster)

            elif Utils.GeneralSetContains(C1, PureClusters):
                IntroduceLatent_PureClusters.append(C1)
            elif Utils.GeneralSetContains(C1, ImpureClusters):
                EarlyLearningImpureClusters.append(C1)
            else:
                print('Something error! Because C1 not be merged but not pure or impure!')



    return [MergeCluster, EarlyLearningImpureClusters, EarlyLearningRemoveClusters, IntroduceLatent_PureClusters, RemainingVariables], PureClusters, ImpureClusters, AllCausalCluster, GeneralPureClusters, LatentIndex






##            elif Utils.GeneralSetContains(C1, GeneralPureCluster) and Utils.GeneralSetContains(C2, ImpureClusters):
##                flag = GeneralPurity(C1, C2, ALLCausalCluster, data, alpha)
##            elif Utils.GeneralSetContains(C2, GeneralPureCluster) and Utils.GeneralSetContains(C1, ImpureClusters):
##                flag = GeneralPurity(C2, C1, ALLCausalCluster, data, alpha)
##            elif Utils.GeneralSetContains(C1, GeneralPureCluster) and Utils.GeneralSetContains(C2, GeneralPureCluster):
##                flag = GeneralPurity_withPurity(C1, C2, ALLCausalCluster, data, alpha)
##            elif Utils.GeneralSetContains(C1, GeneralPureCluster) and Utils.GeneralSetContains(C2, PureClusters):
##                flag = GeneralPurity_withPurity(C1, C2, ALLCausalCluster, data, alpha)
##            elif Utils.GeneralSetContains(C2, GeneralPureCluster) and Utils.GeneralSetContains(C1, PureClusters):
##                flag = GeneralPurity_withPurity(C2, C1, ALLCausalCluster, data, alpha)

def UpdateLatentIndex(EarlyLatent, C1, LatentIndex, AllCausalCluster):
    #inner early learning and outter early learning
    if EarlyLatent in C1:
        TC1=C1.copy()
        TC1.remove(EarlyLatent)
    else:
        TC1=C1

    Lkey=list(LatentIndex.keys())

    if EarlyLatent not in Lkey:
        print('Something error in update early learning for LatentIndex')
        return LatentIndex

    Tclu=LatentIndex[EarlyLatent]

    Tclu=set(Tclu).union(set(TC1))

    LatentIndex[EarlyLatent]=list(Tclu)


    #Update AllCausalCluster
    if C1 in AllCausalCluster:
        AllCausalCluster.remove(C1)
        if TC1 not in AllCausalCluster:
            AllCausalCluster.append(TC1)

    return LatentIndex, AllCausalCluster







def OutterEarlyLearning(RemainingVariables, C, data, Ora_data, LatentIndex, PureClusters, AllCausalCluster, alpha=0.01):
    '''
    DEBUG++++++++++++++++++++
    '''
    print('Detecting Outter Early learning:', C)
    Lkey=list(LatentIndex.keys())
    TC=C
    for L in RemainingVariables:
        if L in Lkey:
            tdata,L_Mset=UpdataData.roll_back(L, Ora_data, data, LatentIndex, AllCausalCluster, PureClusters)
            if len(tdata) < 4:
                print('Outter Early learning_ Roll back dataset: ',tdata.columns)
                continue
            if TestPureWithImpure(L_Mset, C, tdata, alpha):
                 return L, True

    return '', False




#Rule 4
#early learning only in a purity cluster or a merged cluster
def Inner_EarlyLearning(C, data, Ora_data, LatentIndex, PureClusters, AllCausalCluster, alpha=0.01):
    '''
    Function: Test whether a new latent varaible need to be introduced into the Causal cluster C
    Parameter
        C: List, e.g., C=['x1','x2']
            Causal cluster
        data: DataFrame
            observational data
        alpha: float
            signification level of independence
        AllCausalCluters: list
            all learned causal cluster
        LatentIndex: dic
            record the latent index with their measured varaible (directly children set)

    return
        LatentIndex: dic
            Update the laten index if existence of early learning cluster
        TC: list,
            EarlyLearning, C\L (L is early learning)
        Boolean:
            C is early learning cluster (True) or not (False)
    '''

    '''
    DEBUG++++++++++++++++++++
    '''
    print('Detecting Inner Early learning:', C)

    C=list(C)
    Lkey=list(LatentIndex.keys())
    for L in C:
        if L in Lkey:
            TC=C.copy()
            TC.remove(L)
            '''
                # roll back L into their measured set and test merge rule between MeasuredSet(L) and TC
                # we need tdata=data\L U L_Mset, where L_Mset is only two general pure measured variables of L
                # So the AllCausalCluters need to be used to select two pure measured varaibles
            '''

            tdata,L_Mset=UpdataData.roll_back(L, Ora_data, data, LatentIndex, AllCausalCluster, PureClusters)
            if len(tdata) < 4:
                print('Inner Early learning_ Roll back dataset: ',tdata.columns)
                continue
            #recall the pure with impure merge rule for L_Mset and TC, the actived data are tdata
            if Utils.GeneralSetContains(C, PureClusters) or Utils.GeneralSetContains(TC, PureClusters):
                flag = TestPureWithPure(L_Mset, TC, tdata, alpha)
            else:
                flag = TestPureWithImpure(L_Mset, TC, tdata, alpha)

            if flag:
                print('L is early learning !')
                '''
                    The reason for return TC is to help Update actived dataset in the next recursive procedure
                    Since L is learnly, we only update the TC into the Measured_Set(L)
                '''
                return L, True


    return '', False


##                TL=LatentIndex[L]
##                TL=set(TL).union(set(TC))
##                LatentIndex[L]=list(TL)
##                #Update AllCausalCluster
##                if C in AllCausalCluters:
##                    AllCausalCluters.remove(C)
##                    if TC not in AllCausalCluters:
##                        AllCausalCluters.append(TC)
##                return LatentIndex, TC, True







#Rule 3
def TestImpureWithImpure(C1, C2, data, alpha=0.01):
    '''
    Function: Test Merge Rule between two impure cluster
    Parameter
        C1: List, e.g., C=['x1','x2']
            impure causal cluster
        C2: List, e.g., C=['x3','x4']
            impure causal cluster
        data: DataFrame
            observational data
        alpha: float
            signification level of independence

    return
        boolean: True or False
            C1 and C2 are share a common latent parent (True) or not (False)
    '''

    '''
    DEBUG++++++++++++++++++++
    '''
    print('Test Impure with Impure :', C1, C2)

    indexs=list(data.columns)
    indexs=set(indexs)-set(C1)
    indexs=set(indexs)-set(C2)
    V_i=C1[0]
    V_j=C2[0]
    for S in itertools.combinations(list(indexs), 2):
        V_k=S[0]
        V_s=S[1]
        flag=VT.vanishes(data[V_i],data[V_j],data[V_k],data[V_s],alpha)
        if not flag:
            return False

    return True


def TestPureWithPure(C1, C2, data, alpha=0.01):

    '''
    DEBUG++++++++++++++++++++
    '''
    print('Test Pure with Pure :', C1, C2)

    if len(C2) >=2:
        V_i=C1[0]
        V_j=C1[1]
        V_k=C2[0]
        V_s=C2[1]
        flag1=VT.vanishes(data[V_i],data[V_k],data[V_j],data[V_s],alpha)
        flag2=VT.vanishes(data[V_i],data[V_j],data[V_k],data[V_s],alpha)
        flag3=VT.vanishes(data[V_i],data[V_s],data[V_j],data[V_k],alpha)
        print('Pure with Pure All Tetrad Test: ',V_i,V_j,V_k,V_s,flag1,flag2,flag3)
    else:
        V_i=C1[0]
        V_j=C1[1]
        V_k=C2[0]
        Hoset=set(list(data.columns))-set(C1)
        Hoset=Hoset-set(C2)
        for V_s in Hoset:
            flag=VT.vanishes(data[V_i],data[V_k],data[V_j],data[V_s],alpha)
            if not flag:
                return False
        return True


    return (flag1 and flag2 and flag3)





#Rule 2
def TestPureWithImpure(C1, C2, data, alpha=0.01):
    '''
    Function: Test Merge Rule between impure cluster and pure cluster
    Parameter
        C1: List, e.g., C=['x1','x2']
            pure causal cluster with dim(C1)=2
        C2: List, e.g., C=['x3','x4']
            impure causal cluster
        data: DataFrame
            observational data
        alpha: float
            signification level of independence

    return
        boolean: True or False
            C1 and C2 are share a common latent parent (True) or not (False)
    '''

    '''
    DEBUG++++++++++++++++++++
    '''
    print('Test Pure with Impure :', C1, C2)

    indexs=list(data.columns)
    indexs=set(indexs)-set(C1)
    indexs=set(indexs)-set(C2)
    V_i=C1[0]
    V_j=C1[1]
    V_k=C2[0]

    for s in indexs:
        V_s=s
        flag=VT.vanishes(data[V_i],data[V_k],data[V_j],data[V_s],alpha)
        print(V_i,V_k,V_j,V_s,flag)
        if not flag:
            return False

    return True

#Rule 1: Set opeator version
def TestOverlap(C1, C2):
    '''
    Function: Test Overlap between two cluster
    Parameter
        C1: List, e.g., C=['x1','x2']
            causal cluster
        C2: List, e.g., C=['x3','x4']
            causal cluster

    return
        boolean: True or False
            C1 and C2 are overlap (True) or not (False)
    '''
    Cup=set(C1)&set(C2)
    if len(Cup) >0:
        return True
    else:
        return False



'''
    General pure causal cluster contains (1)overlap cluster (2)merged cluster by II(Impure with Impure) or IP (Impure with Pure)
    C1 or C2 is general pure causal cluster
'''
def GeneralPurity_withPurity(C1, C2, ALLCausalCluster, data, alpha=0.01):
    '''
    DEBUG++++++++++++++++++++
    '''
    print('General Purity_with Purity :', C1, C2)

    indexs=list(data.columns)
    indexs=set(indexs)-set(C1)
    indexs=set(indexs)-set(C2)

    GeneralPurity_C1=False
    GeneralPurity_C2=False
    for S in ALLCausalCluster:
        if set(S) < C1:
            GeneralPurity_C1=True
            break
    for S2 in ALLCausalCluster:
        if set(S2) < C2:
            GeneralPurity_C2=True
            break

    if GeneralPurity_C1 and GeneralPurity_C1:
        V_i=list(S)[0]
        TC1=list(set(C1)-set(S))
        V_j=TC1[0]
        V_k=list(S2)[0]
        TC2=list(set(C2)-set(S2))
        V_s=TC2[0]
        flag=VT.vanishes(data[V_i],data[V_k],data[V_j],data[V_s],alpha)
        if flag:
            return True
        else:
            return False

    if GeneralPurity_C1:
        V_i=list(S)[0]
        TC1=list(set(C1)-set(S))
        V_j=TC1[0]
        V_k=C2[0]
        V_s=C2[1]
        flag=VT.vanishes(data[V_i],data[V_k],data[V_j],data[V_s],alpha)
        if flag:
            return True
        else:
            return False

    if GeneralPurity_C2:
        V_i=list(S2)[0]
        TC2=list(set(C2)-set(S2))
        V_j=TC2[0]
        V_k=C1[0]
        V_s=C1[1]
        flag=VT.vanishes(data[V_i],data[V_k],data[V_j],data[V_s],alpha)
        if flag:
            return True
        else:
            return False

    if not GeneralPurity_C1 and not GeneralPurity_C2:
        return TestImpureWithImpure(C1, C2, data, alpha)
    else:
        print('something error in GeneralPurity_withPurity function! where C1 and C2 are ', C1, C2)




def GeneralPurity(C1, C2, ALLCausalCluster, data, alpha=0.01):
    '''
    DEBUG++++++++++++++++++++
    '''
    print('General Purity Detecting :', C1, C2)

    indexs=list(data.columns)
    indexs=set(indexs)-set(C1)
    indexs=set(indexs)-set(C2)

    GeneralPurity=False
    for S in ALLCausalCluster:
        if set(S) < C1:
            GeneralPurity=True
            break
    if GeneralPurity:
        V_i=list(S)[0]
        TC1=list(set(C1)-set(S))
        V_j=TC1[0]
        V_k=C2[0]
        for V_s in indexs:
            flag=VT.vanishes(data[V_i],data[V_k],data[V_j],data[V_s],alpha)
            if not flag:
                return False
        return True
    else:
        #it is impure with impure, recall IIFUNCTION
        return TestImpureWithImpure(C1, C2, data, alpha)



def main():
    pass

if __name__ == '__main__':
    main()
