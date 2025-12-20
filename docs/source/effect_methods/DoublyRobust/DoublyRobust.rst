Doubly Robust Estimator
=======================

Introduction
------------

The Doubly Robust Estimator is a causal effect estimation method that combines outcome regression and propensity score weighting. It is "doubly robust" because it only requires one of the two components (either the outcome model or the propensity score model) to be correctly specified for consistent estimation of the Average Treatment Effect (ATE).

This implementation provides two versions:

1. **Basic Doubly Robust Estimator**: A traditional implementation that works with tabular data.
2. **Network-Aware Doubly Robust Estimator**: An advanced implementation that incorporates network structure information using GCN encoders and B-spline components.

Usage
-----

Basic Doubly Robust Estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from cdmir.effect.ate_estimator import double_robust_estimator
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Generate sample data
    n, p = 1000, 5
    X = np.random.normal(1, 1, (n, p))
    T = np.random.binomial(1, 0.5, n)
    tau = X[:, 0]
    y0 = X @ np.array([1, 2, 0, 0, 0]) + np.random.normal(0, 1, n)
    y1 = y0 + tau
    Y = T * y1 + (1 - T) * y0
    T, Y = T[:, None], Y[:, None]

    # Estimate ATE using basic Doubly Robust Estimator
    tau_dr = double_robust_estimator(X, T, Y, outcome_model=LinearRegression())
    print(f"Estimated ATE: {tau_dr}")

Network-Aware Doubly Robust Estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from cdmir.effect.DoublyRobust.src.targetedModel_DoubleBSpline import TargetedModel_DoubleBSpline

    # Generate sample network data
    batch_size = 100
    Xshape = 5
    hidden = 32
    dropout = 0.1

    # Adjacency matrix (batch_size x batch_size)
    A = torch.rand(batch_size, batch_size) > 0.8
    A = A.float().cuda()

    # Individual features (batch_size x Xshape)
    X = torch.randn(batch_size, Xshape).cuda()

    # Treatment variables (batch_size)
    T = torch.randint(0, 2, (batch_size,)).cuda()

    # Initialize the network-aware Doubly Robust model
    model = TargetedModel_DoubleBSpline(Xshape=Xshape, hidden=hidden, dropout=dropout).cuda()

    # Forward pass to get model outputs
    g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT = model(A, X, T)

    # Infer potential outcomes
    potential_outcomes = model.infer_potential_outcome(A, X, T)

Parameters
----------

Basic Doubly Robust Estimator Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **X**: numpy.ndarray, input data of shape (n_samples, n_features)
  Covariates for each sample.

- **T**: numpy.ndarray, treatment indicator of shape (n_samples,) or (n_samples, 1)
  Binary treatment assignment (1 for treated, 0 for control).

- **Y**: numpy.ndarray, outcome of shape (n_samples,) or (n_samples, 1)
  Outcome variable of interest.

- **outcome_model**: sklearn estimator, default=LinearRegression()
  Regression model used to estimate the potential outcomes.

Network-Aware Doubly Robust Estimator Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Xshape**: int
  Input feature dimension.

- **hidden**: int
  Hidden layer dimension for neural network components.

- **dropout**: float
  Dropout probability for regularization.

- **num_grid**: int, default=None
  Number of B-spline grids (defaults to 20).

- **init_weight**: bool, default=True
  Whether to initialize weights.

- **tr_knots**: float, default=0.25
  Knot density for truncated power basis.

Forward Method Parameters (Network-Aware)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **A**: torch.tensor, adjacency matrix of shape (batch_size, batch_size)
  Network adjacency matrix.

- **X**: torch.tensor, individual features of shape (batch_size, Xshape)
  Covariate features for each node.

- **T**: torch.tensor, treatment variables of shape (batch_size)
  Binary treatment assignment for each node.

- **Z**: torch.tensor, neighbor exposure variables of shape (batch_size), optional
  Pre-computed neighbor exposure (defaults to average of neighbors' treatments).

Returns
-------

Basic Doubly Robust Estimator Returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **tau**: float
  Estimated Average Treatment Effect (ATE).

Network-Aware Doubly Robust Estimator Returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forward Method Returns
^^^^^^^^^^^^^^^^^^^^^^

- **g_T_hat**: torch.tensor
  Estimated treatment propensity scores.

- **g_Z_hat**: torch.tensor
  Estimated neighbor exposure probabilities.

- **Q_hat**: torch.tensor
  Estimated potential outcomes.

- **epsilon**: torch.tensor
  Perturbation terms for bias correction.

- **embeddings**: torch.tensor
  Node embeddings from GCN encoder.

- **neighborAverageT**: torch.tensor
  Average treatment of neighbors.

Infer Potential Outcome Method Returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **potential_outcomes**: torch.tensor
  Adjusted potential outcomes with doubly robust correction.

References
----------

.. [1] Chen W, Cai R, Yang Z, et al. Doubly robust causal effect estimation under networked interference via targeted learning[J]. arXiv preprint arXiv:2405.03342, 2024.