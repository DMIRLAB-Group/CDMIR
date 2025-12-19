LASER (Latent Surrogate-Assisted Effect Regression)
==================================================

Introduction
------------

LASER is a causal effect regression method based on latent surrogate assistance, which uses an identifiable variational autoencoder (iVAE) to model causal effects. This method learns latent variable representations to capture information from surrogate variables, thereby estimating causal effects more accurately.

Usage
----

.. code-block:: python

    from cdmir.effect.LASER.laser import IVAE_tx_wrapper
    import numpy as np
    import torch
    
    # Generate or prepare data
    # Obs: tuple (xo, to, so, yo) - observational data (covariates, treatment, surrogates, outcomes)
    # Exp: tuple (xe, te, se, ye) - experimental data (covariates, treatment, surrogates, outcomes)
    # tau_real: float - true causal effect
    data = (Obs, Exp, tau_real)
    
    # Convert numpy arrays to tensors if needed
    xo, to, so, yo = Obs
    xe, te, se, ye = Exp
    xo, xe, so, yo, te, se, ye, to = torch.tensor(xo), torch.tensor(xe), torch.tensor(so), torch.tensor(yo), \
                                      torch.tensor(te), torch.tensor(se), torch.tensor(ye), torch.tensor(to)
    Obs = (xo, to, so, yo)
    Exp = (xe, te, se, ye)
    data = (Obs, Exp, tau_real)
    
    # Train LASER model
    losses, model = IVAE_tx_wrapper(
        data=data, 
        batch_size=100, 
        max_epoch=1000, 
        n_layers=3, 
        hidden_dim=200,
        learn_rate=1e-4,
        weight_decay=1e-4
    )
    
    # Use the trained model for prediction (with CUDA if available)
    if torch.cuda.is_available():
        y_pred = model.test(covariate=xe.cuda(), s=se.cuda(), treatment=te.cuda())
    else:
        y_pred = model.test(covariate=xe, s=se, treatment=te)

Parameters
----------

**IVAE_tx_wrapper Parameters:**

- **data**: tuple. 

    Input data in the format ``(Obs, Exp, tau_real)`` where:
  - ``Obs``: tuple ``(xo, to, so, yo)`` - observational data (covariates, treatment, surrogates, outcomes)
  - ``Exp``: tuple ``(xe, te, se, ye)`` - experimental data (covariates, treatment, surrogates, outcomes)
  - ``tau_real``: float - true causal effect

- **batch_size**: int, optional, default: 256.

  The size of a batch.

- **max_epoch**: int, optional, default: 2000.
  
  The maximum number of epochs.

- **n_layers**: int, optional, default: 3.

  The number of layers in the MLP.

- **hidden_dim**: list, optional, default: 200.

  The dimension of the hidden layers in the MLP.

- **learn_rate**: float, optional, default: 1e-3.

  Learning rate.

- **weight_decay**: float, optional, default: 1e-4.

  The weight decay coefficient of the optimizer.

- **activation**: str or list, optional, default: 'lrelu'.

  The activation function of the hidden layers.

- **slope**: float, optional, default: 0.1.

  The slope of the LeakyReLU activation function.

- **inference_dim**: int, optional.

  The output dimension of the inference network.

- **optm**: str, optional, default: 'Adam'.

  The optimizer.

- **min_lr**: float, optional, default: 1e-6.

  The minimum lower bound of the learning rate.

- **base_epoch**: int, optional, default: 200.

  Adjust the optimizer parameters after the specified epoch.

- **anneal**: bool, optional, default: False.

  Whether to perform annealing.

- **print_log**: bool, optional, default: True.

  Whether to output logs.

- **is_rct**: bool, optional, default: True.

  Whether it is a randomized controlled experiment.

- **cuda**: bool, optional, default: True.

  Whether to use CUDA.

- **normalization**: bool, optional, default: True.

  Whether to perform normalization.

- **beta**: float, optional, default: 1.

  The coefficient of the negative marginal log-likelihood of y.

- **theta**: float, optional, default: 1.

  The coefficient of the ELBO reconstruction term.

- **early_stop**: bool, optional, default: True.

  Whether early stopping is performed.

- **early_stop_epoch**: int, optional, default: 100.

  Early stop if performance does not improve at the specified epoch.

- **valid_rate**: float, optional, default: 0.2.

  The proportion used to split the validation set.

- **treatment_dim**: int, optional, default: 1.

  The dimension of the treatment vector.

- **treated**: Number, optional, default: 0.7.

  Screen out the treatment group data based on this value.

- **control**: Number, optional, default: 0.97.

  Screen out the control group data based on this value.

**iVAE_tx.test Parameters:**

- **s**: tensor.

  The data of surrogates.

- **covariate**: tensor.

  The data of covariates.

- **treatment**: tensor.
  The data of the treatment vector.

Returns
-------

**IVAE_tx_wrapper Returns:**

- **losses**: tensor.

    Loss values recorded during training.

- **model**: iVAE_tx.

  Trained iVAE model.

**iVAE_tx.test Returns:**

- **meany**: tensor.

  Predicted value of long-term outcome y.

Model Structure
--------

The core of the LASER method is the iVAE_tx class, which contains the following main components:

1. **Encoder**: Used to learn latent variable representations from surrogates, covariates, and treatments
2. **Decoder**: Used to reconstruct surrogates from latent variables
3. **Prior Network**: Used to model the prior distribution of latent variables
4. **Outcome Prediction Network (meany)**: Used to predict long-term outcomes from latent variables and covariates

Training Process
--------

The model is trained using a combination of Evidence Lower Bound (ELBO) and outcome prediction loss as the objective function:

1. First process observational data and experimental data
2. Construct training and validation sets
3. Initialize iVAE_tx model
4. Train using Adam optimizer
5. Adjust learning rate and training hyperparameters as needed

References
----------

.. [1] Cai, Ruichu, et al. "Long-term causal effects estimation via latent surrogates representation learning." Neural networks 176 (2024): 106336.
