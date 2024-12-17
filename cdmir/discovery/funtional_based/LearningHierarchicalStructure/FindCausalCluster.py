import itertools

import MergeCluster
import numpy as np
import Overlap_Merge
import pandas as pd
import TetradMethod
import VanishedTest as VT


def FindCausalCluster(data, PureCluster, ImpureCluster, alpha=0.01):
    """
    Function: Find Causal Cluster from the actived data set and check the purity
    Parameter
        data: DataFrame
            observational data
        alpha: float
            signification level of independence
        PureCluster: List
            PureCluster set
        ImpureCluster: List
            ImpureCluster Set

    return
        LearnedClusters: List
            In current procedure, the learned cluster
        PureCluster: List
            Update PureCluster set by adding the current learned clusters
        ImpureCluster: List
            Update ImpureCluster set by adding the current learned clusters
        PClusters: List
            Return only two element cluster
    """
    LearnedClusters = []
    PClusters = []
    indexs = list(data.columns)  # all observed data in current procedure
    B = indexs.copy()  # remain variables
    ClusterLength = 2
    for S in itertools.combinations(list(B), ClusterLength):
        if TetradMethod.CheckCausalCluster(list(S), data, alpha):
            LearnedClusters.append(S)
            PClusters.append(S)
            B = set(B) - set(S)

    # overlap merge, updata the causal cluster and add into PureCluster
    # only recall the overlap merge function and check whether the cluster with more than three elements
    LearnedClusters = Overlap_Merge.merge_list(LearnedClusters)

    for S in LearnedClusters:
        if len(S) > 2:  # Overlap merged cluster add into Purecluster
            PureCluster.append(S)
        else:  # run the identifying pure cluster function
            Pure_flag = TetradMethod.JudgePureCluster(S, data, alpha)
            if Pure_flag:
                PureCluster.append(S)
            else:
                ImpureCluster.append(S)

    ClusterLength += 1
    while len(B) >= ClusterLength and len(indexs) > (ClusterLength + 2):
        for S in itertools.combinations(list(B), ClusterLength):
            S = list(S)
            if TetradMethod.CheckCausalCluster(list(S), data, alpha):
                LearnedClusters.append(S)
                ImpureCluster.append(S)
                B = set(B) - set(S)
        ClusterLength += 1

    return LearnedClusters, PureCluster, ImpureCluster, PClusters


def main():
    pass


if __name__ == '__main__':
    main()
