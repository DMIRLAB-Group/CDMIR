LaHiCaSI (Latent Hierarchical Causal Structure Learning)
==========================================================

Introduction
------------

LaHiCaSI is a causal discovery method that focuses on learning hierarchical causal structures in the presence of latent variables. It operates in two main phases: first locating latent variables by identifying causal clusters, and then inferring the causal structure among these latent variables.

Usage
-----

.. code-block:: python

    from cdmir.discovery.LaHiCaSl.LaHiCaSl import Latent_Hierarchical_Causal_Structure_Learning
    import pandas as pd
    import numpy as np
    
    # Load or generate your dataset
    # Example: Generate random data with 10 variables and 1000 samples
    data = pd.DataFrame(np.random.randn(1000, 10), columns=[f'X{i}' for i in range(10)])
    
    # Set significance level
    alpha = 0.05
    
    # Run LaHiCaSI algorithm
    Latent_Hierarchical_Causal_Structure_Learning(data, alpha)

Parameters
----------

- **data**: Dataset of observed variables, typically a pandas DataFrame or numpy array.
- **alpha**: Statistical significance threshold (default: 0.05), used to determine the significance of causal relationships during the learning process.

Returns
-------

The function prints the resulting causal structure in the form of an adjacency matrix. It also generates intermediate results during the two-phase learning process.

Algorithm Overview
------------------

LaHiCaSI consists of two main phases:

1. **Phase I: Locate latent variables**
   - **Stage I-S1**: Identify global causal clusters using `IdentifyGlobalCausalClusters`
   - **Stage I-S2**: Determine latent variables by merging clusters using `Determine_Latent_Variables`
   - **Stage I-S3**: Update active data and cluster information using `UpdateActiveData`

2. **Phase II: Infer causal structure among latent variables**
   - Use `LocallyInferCausalStructure` to learn the causal relationships between the identified latent variables

The algorithm iteratively identifies clusters of variables that share common latent causes, updates the data representation to include these latent variables, and then infers the causal structure among them.

References
----------

[1] Xie F, Huang B, Chen Z, et al. Generalized independent noise condition for estimating causal structure with latent variables[J]. Journal of Machine Learning Research, 2024, 25(191): 1-61.