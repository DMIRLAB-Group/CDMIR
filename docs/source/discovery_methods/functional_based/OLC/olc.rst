OLC (One-Component Latent Confounder Detection)
====================================================

Introduction
------------

OLC is a functional-based causal discovery method that detects latent confounders using higher-order cumulants. Based on the paper "Causal Discovery with Latent Confounders Based on Higher-Order Cumulants", this algorithm identifies causal relationships and latent confounders by leveraging the properties of higher-order cumulants and conditional independence tests.

Usage
-----

.. code-block:: python

    import numpy as np
    from cdmir.discovery.funtional_based.one_component.olc import olc
    
    # Generate or load data
    # Example: 1000 samples, 5 variables
    data = np.random.randn(1000, 5)
    
    # Set significance thresholds
    alpha = 0.05  # Primary significance level
    beta = 0.01   # Secondary significance level for more stringent tests
    
    # Run OLC algorithm
    adjmat, coef = olc(data, alpha=alpha, beta=beta, verbose=False)
    
    # Print results
    print("Adjacency Matrix:")
    print(adjmat)
    print("\nCoefficient Matrix:")
    print(coef)

Parameters
----------

- **data**: Input data matrix of shape (n_samples, n_variables), where rows represent samples and columns represent variables.
- **alpha**: Significance threshold for initial edge orientation tests (default: 0.05).
- **beta**: Significance threshold for more stringent tests involving higher-order cumulants (default: 0.01).
- **verbose**: If True, prints detailed information during the algorithm execution (default: False).

Returns
-------

- **adjmat**: Adjacency matrix of the discovered causal graph. The matrix has shape (n_variables + n_latents, n_variables + n_latents), where:
  - 0: No edge
  - 1: Directed edge
  - 2: Undirected edge (ambiguous direction)
  - Latent variables are indexed from n_variables onwards.

- **coef**: Coefficient matrix of the discovered causal relationships. It has the same shape as adjmat and contains the estimated coefficients for each directed edge.

Algorithm Overview
------------------

OLC follows a structured approach to causal discovery with latent confounder detection:

1. **Initialization**
   - Create an undirected graph (UDG) with all possible edges
   - Create an empty directed graph (CG) for causal relationships
   - Initialize KCI (Kernel-based Conditional Independence) test for independence testing

2. **Edge Orientation Phase**
   - Test edge orientations using linear regression and KCI tests
   - Remove edges and orient them in the directed graph based on significance tests
   - Normalize residuals and update data

3. **Clique Detection and Latent Confounder Detection**
   - Identify cliques in the undirected graph
   - Use surrogate regression to handle complex relationships
   - Apply higher-order cumulant (4th order) analysis to detect latent confounders
   - Update the adjacency matrix with detected latent confounders

4. **Refinement**
   - Iteratively refine the graph structure using conditional independence tests
   - Update surrogate variables and exogenous variables
   - Adjust edge orientations based on cumulant-based tests

Key Techniques
--------------

- **Higher-Order Cumulants**: Uses 4th order cumulants to detect latent confounders that cannot be identified using traditional covariance-based methods.

- **KCI Tests**: Employs Kernel-based Conditional Independence tests for robust independence testing between variables and residuals.

- **Surrogate Regression**: Implements surrogate regression to handle complex causal relationships involving multiple variables.

- **Fisher's Combination Test**: Combines multiple p-values to enhance statistical power.

References
----------

.. [1] Cai R, Huang Z, Chen W, et al. Causal discovery with latent confounders based on higher-order cumulants[C]//International conference on machine learning. PMLR, 2023: 3380-3407.