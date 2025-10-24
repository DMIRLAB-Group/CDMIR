import numpy as np
import pandas as pd
from itertools import permutations, combinations
from numpy.linalg import matrix_rank
# import GBNdata
import tensorly as tl
import random
import cdmir.discovery.Tensor_Rank.Gtest as Gtest
import random




def calculate_joint_distribution2(df, var1, var2, var3, var4):
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
    joint_prob_tensor = joint_freq_tensor / joint_freq_tensor.sum()

    return joint_freq_tensor



def LearnCausalCluster(data, LSupp=2, alhpa=0.05):
    """
    Identify causal clusters.
    :param data: array-like
        Input data.
    :param LSupp: int
        Support set for hidden variables.
    :param alhpa: float
        Confidence level for the test rank of goodness of fit test.
    :return:
        Causal clustering learned.
    """

    # Generate a 3-variable combination based on all observed variables.
    indexs = list(data.columns)
    #indexs = ['O1a', 'O1b', 'O1c', 'O2a', 'O2b', 'O2c']

    combinations_list = list(combinations(indexs, 3))

    List_Result = []
    E_Result = []

    CausalCluster = []

    flag = True

    # Test the tensor rank of the corresponding probability distribution for all but three variables
    for clist in combinations_list:
        tempindex = indexs.copy()
        for i in clist:
            if i in tempindex:
                tempindex.remove(i)

        for v4 in tempindex:
            v1 = clist[0]
            v2 = clist[1]
            v3 = clist[2]

            # Calculate the joint probability tensor
            tensor = calculate_joint_distribution2(data,v1,v2,v3,v4)

            # Goodness of fit test tests the tensor rank
            pval = Gtest.test_goodness_of_fit(tensor, LSupp)

            print(clist,v4,pval)

            if pval < alhpa:
                flag = False
                break


        print('-------------------')
        if flag:
            print('This is a causal cluster: ', clist)
            CausalCluster.append(list(clist))
            print('-------------------')
        flag = True



    return CausalCluster











