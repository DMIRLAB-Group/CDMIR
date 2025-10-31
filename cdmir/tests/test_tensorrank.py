# pgmpy<1.0.0 tensorly<0.9.0
import cdmir.discovery.Tensor_Rank.LearnCausalCluster as LCC
import cdmir.discovery.Tensor_Rank.DiscretePC as PC
import random
import pandas as pd

from cdmir.datasets.pgmdata import Gdata2


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