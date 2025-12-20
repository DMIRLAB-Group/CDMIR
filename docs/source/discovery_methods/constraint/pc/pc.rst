PC (Peter-Clark Algorithm)
==========================

Introduction
------------

PC is a constraint-based causal discovery algorithm that infers causal relationships between variables from observational data. It starts with a complete undirected graph and iteratively removes edges based on conditional independence tests, then applies a set of rules to orient edges, resulting in a Partially Directed Acyclic Graph (PDAG) that represents causal relationships.

Usage
-----

.. code-block:: python

    from cdmir.discovery.constraint.pc import PC
    from cdmir.utils.independence import ConditionalIndependentTest

    # Initialize PC algorithm with default parameters
    pc = PC(alpha=0.05, verbose=False)

    # Fit the model to data
    pc.fit(data, var_names, ConditionalIndependentTest)

    # Access results
    causal_graph = pc.causal_graph
    skeleton = pc.skeleton
    sep_set = pc.sep_set

Parameters
----------

PC Class Parameters:

- alpha: Significance level for independence tests (default: 0.05)
- adjacency_search_method: Function for adjacency search phase (default: adjacency_search)
- verbose: Whether to print algorithm progress (default: False)

fit() Method Parameters:

- data: Input dataset containing variable observations
- var_names: List of variable names corresponding to the columns in data
- indep_cls: Conditional independence test class implementing the ConditionalIndependentTest interface
- args: Positional arguments passed to the independence test constructor
- kwargs: Keyword arguments passed to the independence test constructor

Returns
-------

- causal_graph: Partially Directed Acyclic Graph (PDAG) representing inferred causal relationships
- skeleton: Undirected graph representing the skeleton of causal relationships
- sep_set: Separation sets for node pairs, stored as a dictionary where keys are node pairs and values are sets of separating nodes

References
----------

[1] Spirtes, P., Glymour, C. N., Scheines, R., & Heckerman, D. (2000). Causation, prediction, and search. MIT press.