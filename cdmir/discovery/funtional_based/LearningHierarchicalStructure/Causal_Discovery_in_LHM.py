import FindCausalCluster
import MakeGraph
import MergeCluster as MC
import numpy as np
import Orientation
import pandas as pd
import UpdataData
import Utils


def Causal_Discovery_LHM(data, alpha=0.01):
    """
    Function: Causal Discovery in Linear Latent Hierarchical Structure
    Parameter
        data: DataFrame (pandas)
            the observational data set
        alpha: float
            the signification level of independence
    Return
        LatentIndex: dic
            the relations between each latent and their direct measured set
        Graph (selected)
            the Causal graph of hierarchical structure

    """
    # Initize Variable
    Current_Clusters = []
    PureClusters = []
    GeneralPureClusters = []
    ImpureClusters = []

    AllCausalCluster = []

    LatentIndex = {}
    LatentNum = 1
    Ora_data = data.copy()

    while True:

        '''Begin recursive procedure'''

        # Phase I: Finding causal cluster and judge the purity or impurity
        Current_Clusters, PureClusters, ImpureClusters, PClusters = FindCausalCluster.FindCausalCluster(data,
                                                                                                        PureClusters,
                                                                                                        ImpureClusters,
                                                                                                        alpha)
        AllCausalCluster = Utils.ExtendList(AllCausalCluster, PClusters)
        AllCausalCluster = Utils.ExtendList(AllCausalCluster, Current_Clusters)

        # debug
        print('Finished Finding Causal Cluster: ', Current_Clusters, PureClusters, ImpureClusters, PClusters)

        # Phase II: Check merge rule for the learned clusters and update record variables

        Merge_Results, PureClusters, ImpureClusters, AllCausalCluster, GeneralPureClusters, LatentIndex = MC.MergeCausalCluster(
            Current_Clusters, PureClusters, ImpureClusters, AllCausalCluster, GeneralPureClusters, LatentIndex, data,
            Ora_data, alpha)

        MergeCluster = Merge_Results[0]
        EarlyLearningImpureClusters = Merge_Results[1]
        EarlyLearningRemoveClusters = Merge_Results[2]
        IntroduceLatent_PureClusters = Merge_Results[3]
        RemainingVariables = Merge_Results[4]

        print('Merge_Results: ', Merge_Results)
        print(LatentIndex)

        if len(MergeCluster) == 0 and len(IntroduceLatent_PureClusters) == 0:
            print('This is nothing be learned !')
            if len(RemainingVariables) == 0 and len(EarlyLearningImpureClusters) != 0:
                print('There are something wrong! In the merger Results !!', EarlyLearningImpureClusters)
                exit(-1)
            elif len(RemainingVariables) <= 3:
                print('Recursive Procedure Finished ! The structure is identified up to a Markov equivalent class.')
                print(LatentIndex, ImpureClusters)
                break

        # Phase III: Introduce latent variable into the graph and update the actived data set
        data, LatentNum, LatentIndex = UpdataData.UpdataData(Merge_Results, LatentNum, LatentIndex, data)

        if len(data) <= 1:
            print(LatentIndex, ImpureClusters)
            break

        print(data, LatentNum, LatentIndex, ImpureClusters)

        '''End recursive procedure'''

    MakeGraph.Make_graph_Impure(LatentIndex, ImpureClusters)

    # Phase IV: orientation causal direction among latent variable, including latent measured
    Orientation.Orientation_Cluster(Ora_data, LatentIndex, PureClusters, AllCausalCluster)
    # ImpureOrder = Orientation.Orientation_ImpureCluster(Ora_data, LatentIndex, PureClusters, AllCausalCluster, ImpureClusters)
