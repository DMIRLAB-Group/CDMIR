Estimation of Linear Non-Gaussian Latent Hierarchical Structure

Overview
	This project estimates the causal structure of the linear non-Gaussian latent hierarchical model, including both the relations among latent variables and those between latent and observed variables.  


Main Function

Main_LiNG-LHM.py : Estimate_LHM(data,alph)

Input:
	data: DataFrame type,  sample-by-dims size
	alph: significance level of the independence test.

Output:
	m: the adjacency matrix of the causal structure
	Make_graph(LatentIndex):  Plot a causal graph 



Simulation: Case 1-Case 4 

One may use the Simulation_LHM.py to test our method.


Notes
Our method relies heavily on independence tests. One may carefully adjust some parameters, like kernel width, in the kerpy.GaussianKernel, to ensure accuracy.