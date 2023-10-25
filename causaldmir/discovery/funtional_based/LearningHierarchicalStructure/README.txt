Some General Identification Results for Linear Latent Hierarchical Causal Structure -IJCAI2023

Overview
	This project learns the causal structure of the linear latent hierarchical model, including both the relations among latent variables and those between latent and observed variables.  



Main Function

Causal_Discovery_in_LHM.py : Causal_Discovery_LHM(data, alpha=0.01)

Input:
	data: DataFrame (pandas)
     		the observational data set
        	alpha: float
            		the signification level of independence

Output:
        LatentIndex: dic
            the relations between each latent and their direct measured set
        Graph (selected)
            the Causal graph of hierarchical structure



One may use the "test_LHS.py" to test our method, in which a latent tree structure is simulated.


Notes
Our method relies heavily on independence tests. One may carefully adjust some parameters, like kernel width, in the kerpy.GaussianKernel, to ensure accuracy.