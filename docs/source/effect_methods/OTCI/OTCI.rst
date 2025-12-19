OTCI (Optimal Transport-based Causal Inference)
================================================

Introduction
------------

OTCI is an Optimal Transport-based Causal Inference method for estimating the Average Treatment Effect on the Treated (ATT). It leverages optimal transport theory to construct weights for control group samples that best match the distribution of the treatment group samples.

The algorithm provides two modes:

1. **Basic Mode**: Uses only feature distance information
2. **GW-Enhanced Mode**: Incorporates Gromov-Wasserstein (GW) distance to preserve geometric structure in the data

OTCI is implemented with CUDA acceleration for efficient computation on large datasets.

Usage
-----

.. code-block:: python

    import torch
    from cdmir.effect.OTCI.src.otci import optimal_transport_weighting

    # Generate or load data (ensure data is on CUDA)
    X_t = torch.randn(100, 10).cuda()  # Treatment group features
    X_c = torch.randn(200, 10).cuda()  # Control group features
    Y_t = torch.randn(100).cuda()       # Treatment group outcomes
    Y_c = torch.randn(200).cuda()       # Control group outcomes

    # Basic usage with default parameters
    pred_att, weights, transport_matrix = optimal_transport_weighting(X_t, X_c, Y_t, Y_c)
    print(f"Estimated ATT: {pred_att.item()}")

    # Advanced usage with GW distance and custom parameters
    pred_att, weights, transport_matrix = optimal_transport_weighting(
        X_t, X_c, Y_t, Y_c,
        gamma=1e-4,                 # Negative entropy regularization strength
        eta_base=0.5,               # Initial learning rate
        eta_update_steps=20,        # Learning rate update frequency
        decay_rate=0.98,            # Learning rate decay rate
        max_iter=500,               # Maximum number of iterations
        abstol=1e-3,                # Early stopping threshold
        eps=1e-5,                   # Small value to avoid log(0)
        with_GW=True,               # Enable GW distance
        alpha=0.95                  # Weight between original cost and GW distance
    )
    print(f"Estimated ATT with GW: {pred_att.item()}")

Parameters
----------

- **X_t**: torch.Tensor, treatment group features of shape (n_t, d)
  Features of samples in the treatment group.

- **X_c**: torch.Tensor, control group features of shape (n_c, d)
  Features of samples in the control group.

- **Y_t**: torch.Tensor, treatment group outcomes of shape (n_t,)
  Outcome values for samples in the treatment group.

- **Y_c**: torch.Tensor, control group outcomes of shape (n_c,)
  Outcome values for samples in the control group.

- **gamma**: float, default=0.1
  Strength of negative entropy regularization.

- **eta_base**: float, default=0.01
  Initial learning rate for updating the transport matrix.

- **eta_update_steps**: int, default=10
  Frequency at which the learning rate is updated.

- **decay_rate**: float, default=0.95
  Decay rate for the learning rate.

- **max_iter**: int, default=2000
  Maximum number of iterations for the optimization process.

- **abstol**: float, default=1e-5
  Early stopping threshold based on the absolute difference between consecutive transport matrices.

- **eps**: float, default=1e-3
  Small value added to avoid log(0) operations.

- **with_GW**: bool, default=True
  Whether to incorporate Gromov-Wasserstein distance in the optimization.

- **alpha**: float, default=0.1
  Weight parameter balancing the original cost and GW distance when with_GW is True.

Returns
-------

- **pred_att**: torch.Tensor, scalar
  Estimated Average Treatment Effect on the Treated (ATT).

- **weights**: torch.Tensor, shape (n_c,)
  Optimal weights assigned to control group samples.

- **transport_matrix**: torch.Tensor, shape (n_c, n_t)
  Final optimal transport matrix between control and treatment groups.

References
----------

.. [1] Yan Y, Yang Z, Chen W, et al. Exploiting geometry for treatment effect estimation via optimal transport[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(15): 16290-16298.