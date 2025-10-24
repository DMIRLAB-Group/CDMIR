import itertools
# from src.utils import get_causal_chains, plot
from itertools import combinations, chain
from scipy.stats import norm, pearsonr
import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
import tensorly as tl
import cdmir.discovery.Tensor_Rank.Gtest as Gtest
import random

LSupp = 2

def calculate_joint_distribution6(df, var1, var2, var3, var4, var5, var6):
    """
        Calculate the joint probability tensor.
    :param df: array-like
        Input data.
    :param var1: str
        Index of variables.
    :param var2: str
        Index of variables.
    :param var3: str
        Index of variables.
    :param var4: str
        Index of variables.
    :param var5: str
        Index of variables.
    :param var6: str
        Index of variables.
    :return:
        The joint probability distribution tensor of variables.
    """
    # Calculate the joint frequency table of six variables
    joint_freq = pd.crosstab(index=[df[var1], df[var2], df[var3], df[var4], df[var5]], columns=df[var6])

    # Create a six-dimensional tensor and initialize it to 0
    tensor_shape = (len(df[var1].unique()), len(df[var2].unique()),
                    len(df[var3].unique()), len(df[var4].unique()),
                    len(df[var5].unique()), len(df[var6].unique()))

    joint_freq_tensor = np.zeros(tensor_shape)

    # Filling tensor
    for i, val1 in enumerate(sorted(df[var1].unique())):
        for j, val2 in enumerate(sorted(df[var2].unique())):
            for k, val3 in enumerate(sorted(df[var3].unique())):
                for l, val4 in enumerate(sorted(df[var4].unique())):
                    for m, val5 in enumerate(sorted(df[var5].unique())):
                        for n, val6 in enumerate(sorted(df[var6].unique())):
                            if (val1, val2, val3, val4, val5) in joint_freq.index and val6 in joint_freq.columns:
                                joint_freq_tensor[i, j, k, l, m, n] = joint_freq.loc[(val1, val2, val3, val4, val5), val6]
                            else:
                                joint_freq_tensor[i, j, k, l, m, n] = 0

    # Calculate the joint probability tensor
    joint_prob_tensor = joint_freq_tensor / joint_freq_tensor.sum()

    return joint_freq_tensor
def calculate_joint_distribution4(df, var1, var2, var3, var4):
    """
        Calculate the joint probability tensor.
    :param df: array-like
        Input data.
    :param var1: str
        Index of variables.
    :param var2: str
        Index of variables.
    :param var3: str
        Index of variables.
    :param var4: str
        Index of variables.
    :return:
        The joint probability distribution tensor of variables.
    """
    # Calculate the joint frequency table of four variables
    joint_freq = pd.crosstab(index=[df[var1], df[var2], df[var3]], columns=df[var4])

    # Create a four-dimensional tensor and initialize it to 0
    tensor_shape = (len(df[var1].unique()), len(df[var2].unique()),
                    len(df[var3].unique()), len(df[var4].unique()))

    joint_freq_tensor = np.zeros(tensor_shape)

    # Filling tensor
    for i, val1 in enumerate(sorted(df[var1].unique())):
        for j, val2 in enumerate(sorted(df[var2].unique())):
            for k, val3 in enumerate(sorted(df[var3].unique())):
                for l, val4 in enumerate(sorted(df[var4].unique())):
                    if (val1, val2, val3) in joint_freq.index and val4 in joint_freq.columns:
                        joint_freq_tensor[i, j, k, l] = joint_freq.loc[(val1, val2, val3), val4]
                    else:
                        joint_freq_tensor[i, j, k, l] = 0

    # Calculate the joint probability tensor
    # joint_prob_tensor = joint_freq_tensor / joint_freq_tensor.sum()

    joint_prob_tensor = joint_freq_tensor

    return joint_freq_tensor

def subset(iterable):
    """
        Generates all subsets of iterable objects.
    :param iterable: array-like
        Iterable objects.
    :return:
        All subsets of iterable objects.
    """
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))



def indepTest(x1,y1,Z1,labels1,data1,cluster,alpha = 0.2):
    """
        Conditional independence test to identify structural models.
    :param x1: int
        Index of latent variables.
    :param y1: int
        Index of latent variables.
    :param Z1: list
        List of conditional variables.
    :param labels1: list
        Real variable name.
    :param data1: array-like
        Data for all observed variables.
    :param cluster: dict
        Causal clustering composed of observed variables corresponding to hidden variables.
    :param alpha: float
        Significance level for conditional independent test.
    :return:
        Conditional independent tensor rank test results.
    """
    labels = labels1.copy()
    data = data1.copy()
    Z = Z1.copy()
    x = labels[x1]
    y = labels[y1]
    if len(Z) ==0:

        return False

    # Gets subvariables of hidden variables
    L1 = cluster[x][0]
    L2 = cluster[y][0]

    # For each condition variable, get its child nodes to test.
    LS = []

    for i in Z:
        sub = cluster[i]
        LS.append(sub[0])
        LS.append(sub[1])

    Variable_set = LS.copy()
    Variable_set.append(L1)
    Variable_set.append(L2)


    #print('test the rank constraints for: ', Variable_set)

    # The condition set is 1
    if len(Z) == 1:
        v1 = Variable_set[0]
        v2 = Variable_set[1]
        v3 = Variable_set[2]
        v4 = Variable_set[3]
        tensor = calculate_joint_distribution4(data,v1,v2,v3,v4)


        # Test whether the tensor rank is r
        pval = Gtest.test_goodness_of_fit(tensor, LSupp)

        print('test the tensor rank for ', v1,v2,v3,v4,pval)

        if pval > alpha:
            return True

    # The condition set is 2
    elif len(Z) == 2:
        v1 = Variable_set[0]
        v2 = Variable_set[1]
        v3 = Variable_set[2]
        v4 = Variable_set[3]
        v5 = Variable_set[4]
        v6 = Variable_set[5]
        tensor = calculate_joint_distribution6(data,v1,v2,v3,v4,v5,v6)

        # Test whether the tensor rank is r^{Lp}
        pval = Gtest.test_goodness_of_fit(tensor, LSupp * LSupp)

        print('test the tensor rank for ', v1,v2,v3,v4,v5,v6,pval)

        if pval > alpha:
            return True



    else:
        print('This is out of my testability!')
        exit(-1)


    return False




def test(data1,la,cluster,alpha = 0.2):
    """
        Conditional independent testing for each variable yields a causal adjacency matrix.
    :param data1: array-like
        Data for all observed variables.
    :param la: list
        Hidden set of variable names.
    :param cluster: dict
        Causal clustering composed of observed variables corresponding to hidden variables.
    :param alpha: float
        Significance level for conditional independent test.
    :return:
        Causal adjacency matrix of variables.
    """
    data = data1.copy()
    labels = la.copy()

    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]
    # Initialization graph
    G = [[True for i in range(len(labels))] for i in range(len(labels))]
    for i in range(len(labels)):
        G[i][i] = False

    # done flag
    done = False

    # Count records
    ord = 0
    n_edgetests = {0: 0}
    while done != True and any(G):
        ord1 = ord + 1
        n_edgetests[ord1] = 0
        done = True

        # Collect edges that need to be manipulated.
        ind = []
        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j] == True:
                    ind.append((i, j))


        G1 = G.copy()
        for x, y in ind:
            if G[x][y] == True:
                # Find the index of a variable with an edge with x.
                neighborsBool = [row[x] for row in G1]
                neighborsBool[y] = False

                neighbors = [i for i in range(len(neighborsBool)) if neighborsBool[i] == True]

                # Whether the number of neighbors can constitute the corresponding condition set.
                if len(neighbors) >= ord:
                    # |adj(C, x) / {y}|>ord
                    if len(neighbors) > ord:
                        done = False
                    # |adj(C, x) / {y}|=ord
                    # Combination variables constitute a condition set
                    for neighbors_S in set(itertools.combinations(neighbors, ord)):
                        Z = neighbors_S
                        Z =list(Z)
                        ls = labels.copy()
                        S = []
                        # Get names
                        for i in Z:
                            S.append(ls[i])

                        if len(Z) >2:
                            break

                        # Conditional independence test
                        pval = indepTest(x,y,S,labels,data,cluster)


                        if pval:
                            print('d-separation relations: ',labels[x],labels[y])
                            G[x][y] = G[y][x] = False
                            sepset[x][y] = list(neighbors_S)
                            break




        ord += 1

    return np.array(G)

