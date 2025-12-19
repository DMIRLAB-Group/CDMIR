Tensor Rank Causal Discovery
=============================

Introduction
------------

Tensor rank causal discovery is a method for learning discrete latent variable models with a three-pure-children structure. It uses tensor rank conditions to identify causal clusters from observed variables and then infers d-separation relationships among latent variables.

This method consists of three main components:

1. **LCC (LearnCausalCluster)**: Identifies causal clusters from observed variables using tensor rank conditions
2. **Gtest**: Performs goodness of fit tests to determine tensor ranks
3. **DiscretePC**: Learns causal skeleton relationships among latent variables

Usage
-----

.. code-block:: python

    from cdmir.discovery.Tensor_Rank.LearnCausalCluster import LearnCausalCluster
    import cdmir.discovery.Tensor_Rank.DiscretePC as PC
    import pandas as pd
    from cdmir.datasets.pgmdata import Gdata2
    import pkg_resources

    # Example 1: Learn causal clusters
    data = Gdata2(100000)
    clusters = LearnCausalCluster(data, LSupp=2)
    print("Learned causal clusters:", clusters)

    # Example 2: Learn causal skeleton
    csv_path = pkg_resources.resource_filename('cdmir', 'tests/testdata/out.csv')
    data = pd.read_csv(csv_path)
    labels = ['L1', 'L2', 'L3']
    cluster = {'L1': ['O1a', 'O1b', 'O1c'], 'L2': ['O2a', 'O2b', 'O2c'], 'L3': ['O3a', 'O3b', 'O3c']}
    adjacency_matrix = PC.test(data, labels, cluster)
    print("Causal adjacency matrix of latent variables:", adjacency_matrix)

Parameters
----------

**LCC (LearnCausalCluster) Parameters:**

- **data**: ndarray.

  Input data containing observed variables.
- **LSupp**: int, optional, default: 2.

  Support set size for hidden variables.
- **alpha**: float, optional, default: 0.05.

  Confidence level for the goodness of fit test.

**Gtest (test_goodness_of_fit) Parameters:**

- **tensor**: ndarray.

  The original four-way tensor to test.
- **rank**: int.

  The rank of the CP decomposition.

**DiscretePC (test) Parameters:**

- **data1**: ndarray.

  Data for all observed variables.
- **la**: list.

  List of hidden variable names.
- **cluster**: dict.

  Causal clustering composed of observed variables corresponding to hidden variables.
- **alpha**: float, optional, default: 0.2.

  Significance level for conditional independent test.

Returns
-------

**LCC (LearnCausalCluster) Returns:**

- **CausalCluster**: list.

  List of identified causal clusters, where each cluster is a list of variable names.

**Gtest (test_goodness_of_fit) Returns:**

- **chi_square_p_value**: float.

  P-value from the Chi-square goodness of fit test.

**DiscretePC (test) Returns:**

- **adjacency_matrix**: ndarray.

  Causal adjacency matrix of variables, where True indicates a direct causal relationship.

References
----------

.. [1] Chen Z, Cai R, Xie F, et al. Learning Discrete Latent Variable Structures with Tensor Rank Conditions[C]//The Thirty-eighth Annual Conference on Neural Information Processing Systems.
