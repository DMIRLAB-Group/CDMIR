# pgmpy<1.0.0 tensorly<0.9.0
import cdmir.discovery.Tensor_Rank.LearnCausalCluster as LCC
import cdmir.discovery.Tensor_Rank.DiscretePC as PC
import random
import pandas as pd

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling


def Gdata(Num=3000):
    """
    Generate a probability graph model with a chain structure of hidden variables and each hidden variable has three sub-observation variables.
    :param Num: int
        Number of data samples to generate
    :return:
        A specified number of data sampled through a given probability graph structure
    """

    # Define the structure of the Bayesian Network
    model = BayesianNetwork([
        ('L1', 'L2'), ('L2', 'L3'),
        ('L1', 'O1a'), ('L1', 'O1b'), ('L1', 'O1c'),
        ('L2', 'O2a'), ('L2', 'O2b'), ('L2', 'O2c'),
        ('L3', 'O3a'), ('L3', 'O3b'), ('L3', 'O3c')
    ])

    # Define the CPTs for the latent variables
    # Define a chained hidden variable structure L1->L2->L3 and corresponding probability distribution
    cpd_l1 = TabularCPD(variable='L1', variable_card=2, values=[[0.7], [0.3]])
    cpd_l2 = TabularCPD(variable='L2', variable_card=2, values=[[0.6, 0.5], [0.4, 0.5]], evidence=['L1'], evidence_card=[2])
    cpd_l3 = TabularCPD(variable='L3', variable_card=2, values=[[0.2, 0.9], [0.8, 0.1]], evidence=['L2'], evidence_card=[2])
    #cpd_l3 = TabularCPD(variable='L3', variable_card=2, values=[[0.8, 0.5, 0.6, 0.1], [0.2, 0.5, 0.4, 0.9]], evidence=['L1', 'L2'], evidence_card=[2, 2])

    # Define CPDs for the observed variables (each with three states)
    # Note: We are assigning unique CPD for each observed variable
    cpds_obs = [
        TabularCPD(variable=f'O{i}{j}', variable_card=3,
                  values=[
                      [0.7, 0.2],  # Probability of state 0
                      [0.2, 0.6],  # Probability of state 1
                      [0.1, 0.2]], # Probability of state 2
                  evidence=[f'L{i}'], evidence_card=[2])
        for i in range(1, 4) for j in ['a', 'b', 'c']
    ]

    # Update some CPDs to make them unique
    cpds_obs[1] = TabularCPD(variable='O1b', variable_card=3,
                             values=[
                                 [0.6, 0.3],
                                 [0.3, 0.4],
                                 [0.1, 0.3]],
                             evidence=['L1'], evidence_card=[2])
    cpds_obs[2] = TabularCPD(variable='O1c', variable_card=3,
                             values=[
                                 [0.5, 0.4],
                                 [0.4, 0.3],
                                 [0.1, 0.3]],
                             evidence=['L1'], evidence_card=[2])
    # Continue updating other CPDs similarly to ensure uniqueness


    cpds_obs[3] = TabularCPD(variable='O2a', variable_card=3,
                             values=[
                                 [0.15, 0.3],
                                 [0.25, 0.4],
                                 [0.6, 0.3]],
                             evidence=['L2'], evidence_card=[2])
    cpds_obs[4] = TabularCPD(variable='O2b', variable_card=3,
                             values=[
                                 [0.5, 0.1],
                                 [0.4, 0.2],
                                 [0.1, 0.7]],
                             evidence=['L2'], evidence_card=[2])

    cpds_obs[5] = TabularCPD(variable='O2c', variable_card=3,
                             values=[
                                 [0.6, 0.2],
                                 [0.3, 0.4],
                                 [0.1, 0.4]],
                             evidence=['L2'], evidence_card=[2])

    cpds_obs[6] = TabularCPD(variable='O3a', variable_card=3,
                             values=[
                                 [0.3, 0.4],
                                 [0.4, 0.1],
                                 [0.3, 0.5]],
                             evidence=['L3'], evidence_card=[2])

    cpds_obs[7] = TabularCPD(variable='O3b', variable_card=3,
                             values=[
                                 [0.6, 0.1],
                                 [0.2, 0.5],
                                 [0.2, 0.4]],
                             evidence=['L3'], evidence_card=[2])
    cpds_obs[8] = TabularCPD(variable='O3c', variable_card=3,
                             values=[
                                 [0.8, 0.45],
                                 [0.1, 0.35],
                                 [0.1, 0.2]],
                             evidence=['L3'], evidence_card=[2])



    # Add all CPDs to the model
    model.add_cpds(cpd_l1, cpd_l2, cpd_l3, *cpds_obs)

    # Check model validity
    assert model.check_model()

    # Sampling from the model
    sampler = BayesianModelSampling(model)

    sample = sampler.forward_sample(size=Num)

    sample = sample[['O1a', 'O1b', 'O1c', 'O2a', 'O2b', 'O2c', 'O3a',
           'O3b', 'O3c']]

    #print(sample)
    return sample


def Gdata2(Num=3000):
    """
    Generate a probability graph model with a chain structure of hidden variables and each hidden variable has three sub-observation variables.
    :param Num: int
        Number of data samples to generate
    :return:
        A specified number of data sampled through a given probability graph structure
    """

    # Define the structure of the Bayesian Network
    model = BayesianNetwork([
        ('L1', 'L2'), ('L2', 'L3'),
        ('L1', 'O1a'), ('L1', 'O1b'), ('L1', 'O1c'),
        ('L2', 'O2a'), ('L2', 'O2b'), ('L2', 'O2c'),
        ('L3', 'O3a'), ('L3', 'O3b'), ('L3', 'O3c')
    ])

    # Define the CPTs for the latent variables
    # Define a chained hidden variable structure L1->L2->L3 and corresponding probability distribution
    cpd_l1 = TabularCPD(variable='L1', variable_card=2, values=[[0.7], [0.3]])
    cpd_l2 = TabularCPD(variable='L2', variable_card=2, values=[[0.6, 0.5], [0.4, 0.5]], evidence=['L1'], evidence_card=[2])
    cpd_l3 = TabularCPD(variable='L3', variable_card=2, values=[[0.2, 0.9], [0.8, 0.1]], evidence=['L2'], evidence_card=[2])
    #cpd_l3 = TabularCPD(variable='L3', variable_card=2, values=[[0.8, 0.5, 0.6, 0.1], [0.2, 0.5, 0.4, 0.9]], evidence=['L1', 'L2'], evidence_card=[2, 2])

    # Define CPDs for the observed variables (each with three states)
    # Note: We are assigning unique CPD for each observed variable
    cpds_obs = [
        TabularCPD(variable=f'O{i}{j}', variable_card=3,
                  values=[
                      [0.7, 0.2],  # Probability of state 0
                      [0.2, 0.6],  # Probability of state 1
                      [0.1, 0.2]], # Probability of state 2
                  evidence=[f'L{i}'], evidence_card=[2])
        for i in range(1, 4) for j in ['a', 'b', 'c']
    ]

    # Update some CPDs to make them unique
    cpds_obs[1] = TabularCPD(variable='O1b', variable_card=3,
                             values=[
                                 [0.6, 0.3],
                                 [0.3, 0.4],
                                 [0.1, 0.3]],
                             evidence=['L1'], evidence_card=[2])
    cpds_obs[2] = TabularCPD(variable='O1c', variable_card=3,
                             values=[
                                 [0.5, 0.4],
                                 [0.4, 0.3],
                                 [0.1, 0.3]],
                             evidence=['L1'], evidence_card=[2])
    # Continue updating other CPDs similarly to ensure uniqueness


    cpds_obs[3] = TabularCPD(variable='O2a', variable_card=3,
                             values=[
                                 [0.15, 0.3],
                                 [0.25, 0.4],
                                 [0.6, 0.3]],
                             evidence=['L2'], evidence_card=[2])
    cpds_obs[4] = TabularCPD(variable='O2b', variable_card=3,
                             values=[
                                 [0.5, 0.1],
                                 [0.4, 0.2],
                                 [0.1, 0.7]],
                             evidence=['L2'], evidence_card=[2])

    cpds_obs[5] = TabularCPD(variable='O2c', variable_card=3,
                             values=[
                                 [0.6, 0.2],
                                 [0.3, 0.4],
                                 [0.1, 0.4]],
                             evidence=['L2'], evidence_card=[2])

    cpds_obs[6] = TabularCPD(variable='O3a', variable_card=3,
                             values=[
                                 [0.3, 0.4],
                                 [0.4, 0.1],
                                 [0.3, 0.5]],
                             evidence=['L3'], evidence_card=[2])

    cpds_obs[7] = TabularCPD(variable='O3b', variable_card=3,
                             values=[
                                 [0.6, 0.1],
                                 [0.2, 0.5],
                                 [0.2, 0.4]],
                             evidence=['L3'], evidence_card=[2])
    cpds_obs[8] = TabularCPD(variable='O3c', variable_card=3,
                             values=[
                                 [0.8, 0.45],
                                 [0.1, 0.35],
                                 [0.1, 0.2]],
                             evidence=['L3'], evidence_card=[2])



    # Add all CPDs to the model
    model.add_cpds(cpd_l1, cpd_l2, cpd_l3, *cpds_obs)

    # Check model validity
    assert model.check_model()

    # Sampling from the model
    sampler = BayesianModelSampling(model)
    sample = sampler.forward_sample(size=Num)

    sample = sample[['O1a', 'O1b', 'O1c', 'O2a', 'O2b', 'O2c']]

    #print(sample)
    return sample









'''
A toy example illustrating the tensor rank condition for learning discrete latent variable models with a three-pure-children structure is presented.
The proposed two-stage algorithm first identifies causal clusters from the observed variables and then infers the d-separation relationships among the latent variables.

Reference:
    [1] Chen Z, Cai R, Xie F, et al. Learning Discrete Latent Variable Structures with Tensor Rank Conditions[C]//The Thirty-eighth Annual Conference on Neural Information Processing Systems.

'''
def main():
    #Causal Cluster learning by Tensor Rank Condition, ground truth: [['O1a', 'O1b', 'O1c'], ['O2a', 'O2b', 'O2c']]
    test1()


    #Test the causal skeleton learning, ground truth: L1-L2-L3
    test2()



def test1():
    data = Gdata2(100000)


    print(data.columns)


    Cluster = LCC.LearnCausalCluster(data)

    print('##########-----------------------------------------')
    print('The learned causal cluster is : ',Cluster)
    print('##########-----------------------------------------')


def test2():

    data = pd.read_csv('testdata/out.csv')

    labels = ['L1','L2','L3']

    cluster = {'L1':['O1a','O1b', 'O1c'],'L2':['O2a','O2b', 'O2c'],'L3':['O3a','O3b', 'O3c']}


    p = PC.test(data,labels,cluster)

    m = p+0

    print('###########-----------------------------------------')
    print('The adjacent matrix among L1, L2 and L3 is: ')
    print(m)
    print('###########-----------------------------------------')



if __name__ == '__main__':
    main()