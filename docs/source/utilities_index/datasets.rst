Datasets
=======

The dataset generation module provides various tools for generating data for causal discovery experiments, including Probabilistic Graphical Model (PGM) data, Independent and Identically Distributed (IID) data, and time series data.

.. toctree::
   :maxdepth: 1


PGM Data Generation
------------------

`pgmdata.py` provides PGM-based dataset generation.

Gdata
^^^^^

Generate PGM data with chain-structured hidden variables.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.pgmdata import Gdata
    data = Gdata(Num=5000)

Parameters
~~~~~~~~~~
- `Num`: Number of samples, default=3000

Returns
~~~~~~~
- pandas.DataFrame: Generated data with 9 observation variables

Gdata2
^^^^^

Generate PGM data with chain-structured hidden variables.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.pgmdata import Gdata2
    data = Gdata2(Num=5000)

Parameters
~~~~~~~~~~
- `Num`: Number of samples, default=3000

Returns
~~~~~~~
- pandas.DataFrame: Generated data with 6 observation variables


Simulators
----------

`simlulators.py` provides IID and time series data generators.

IIDSimulator
^^^^^^^^^^^

simulate_linear_anm
~~~~~~~~~~~~~~~~~~

Simulate dataset from linear additive-noise model.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.simlulators import IIDSimulator
    data = IIDSimulator.simulate_linear_anm(weight_mat, n_samples=1000)

Parameters
~~~~~~~~~~
- `weight_mat`: Weighted adjacency matrix of shape [n_nodes, n_nodes]
- `n_samples`: Number of samples
- `noise_type`: Noise type (default: 'gauss')
- `noise_scale`: Noise scale parameter
- `seed`: Random seed

Returns
~~~~~~~
- np.ndarray: Generated data of shape [n_samples, n_nodes]

simulate_nonlinear_anm
~~~~~~~~~~~~~~~~~~~~~

Simulate dataset from nonlinear additive-noise model.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.simlulators import IIDSimulator
    data = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples=1000)

Parameters
~~~~~~~~~~
- `adj_mat`: Adjacency matrix of shape [n_nodes, n_nodes]
- `n_samples`: Number of samples
- `noise_type`: Noise type (default: 'gauss')
- `noise_scale`: Noise scale parameter
- `seed`: Random seed
- `func_type`: Nonlinear function type (default: 'mlp')
- `hidden`: Number of hidden layers

Returns
~~~~~~~
- np.ndarray: Generated data of shape [n_samples, n_nodes]

simulate_pnl
~~~~~~~~~~~

Simulate dataset from post-nonlinear model.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.simlulators import IIDSimulator
    data = IIDSimulator.simulate_pnl(adj_mat, n_samples=1000)

Parameters
~~~~~~~~~~
- `adj_mat`: Adjacency matrix of shape [n_nodes, n_nodes]
- `n_samples`: Number of samples
- `noise_type`: Noise type (default: 'gauss')
- `noise_scale`: Noise scale parameter
- `seed`: Random seed
- `func1_type`: First nonlinear function type (default: 'tanh')
- `func2_type`: Second nonlinear function type (default: 'leaky_relu')
- `neg_slope`: Negative slope for leaky ReLU (default: 0.2)
- `hidden`: Number of hidden layers

Returns
~~~~~~~
- np.ndarray: Generated data of shape [n_samples, n_nodes]

TimeLagSimulator
^^^^^^^^^^^^^^^

simulate_linear_anm
~~~~~~~~~~~~~~~~~~

Simulate time series from linear additive-noise model.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.simlulators import TimeLagSimulator
    data = TimeLagSimulator.simulate_linear_anm(weight_mats, max_lag=2, length=1000)

Parameters
~~~~~~~~~~
- `weight_mats`: Weighted adjacency matrix
- `max_lag`: Maximum lag time
- `length`: Length of generated data
- `noise_type`: Noise type (default: 'gauss')
- `noise_scale`: Noise scale parameter
- `seed`: Random seed

Returns
~~~~~~~
- np.ndarray: Generated data of shape [length+max_lag, n_nodes]

simulate_nonlinear_anm
~~~~~~~~~~~~~~~~~~~~~

Simulate time series from nonlinear additive-noise model.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.simlulators import TimeLagSimulator
    data = TimeLagSimulator.simulate_nonlinear_anm(adj_mats, max_lag=2, length=1000)

Parameters
~~~~~~~~~~
- `adj_mats`: Adjacency matrix
- `max_lag`: Maximum lag time
- `length`: Length of generated data
- `noise_type`: Noise type (default: 'gauss')
- `noise_scale`: Noise scale parameter
- `seed`: Random seed
- `func_type`: Nonlinear function type (default: 'mlp')
- `hidden`: Number of hidden layers

Returns
~~~~~~~
- np.ndarray: Generated data of shape [length+max_lag, n_nodes]


Utility Functions
----------------

`utils.py` provides utility functions for dataset generation.

set_random_seed
^^^^^^^^^^^^^^

Set random seed for numpy, random and optionally PyTorch.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.utils import set_random_seed
    set_random_seed(42, set_torch_seed=True)

Parameters
~~~~~~~~~~
- `seed`: Random seed
- `set_torch_seed`: Whether to set PyTorch random seed, default=True

pd2np
^^^^^

Convert pandas.DataFrame to numpy array.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.utils import pd2np
    numpy_data = pd2np(pandas_data)

Parameters
~~~~~~~~~~
- `pd_data`: pandas.DataFrame object

Returns
~~~~~~~
- np.ndarray: Converted numpy array

nx2np
^^^^^

Convert networkx.Graph to numpy array.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.utils import nx2np
    adj_matrix = nx2np(networkx_graph)

Parameters
~~~~~~~~~~
- `nx_data`: networkx.Graph object

Returns
~~~~~~~
- np.ndarray: Converted adjacency matrix

np2nx
^^^^^

Convert numpy array to networkx.Graph.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.utils import np2nx
    graph = np2nx(adj_matrix, create_using=nx.DiGraph)

Parameters
~~~~~~~~~~
- `np_data`: numpy.ndarray adjacency matrix
- `create_using`: Graph type to create (e.g., nx.DiGraph)

Returns
~~~~~~~
- networkx.Graph: Converted graph object

leaky_relu
^^^^^^^^^

Apply Leaky ReLU activation function to input data.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.utils import leaky_relu
    output = leaky_relu(inputs, neg_slope=0.2)

Parameters
~~~~~~~~~~
- `inputs`: Input data
- `neg_slope`: Negative slope, default=0.2

Returns
~~~~~~~
- np.ndarray: Output after applying Leaky ReLU

check_data
^^^^^^^^^

Check data type and convert to ndarray if necessary.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.utils import check_data
    numpy_data = check_data(input_data)

Parameters
~~~~~~~~~~
- `inputs`: Input data (np.ndarray or pd.DataFrame)

Returns
~~~~~~~
- np.ndarray: Converted numpy array

erdos_renyi
^^^^^^^^^^

Generate Erdos-Renyi DAG adjacency matrix.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.utils import erdos_renyi
    adj_matrix = erdos_renyi(n_nodes=5, n_edges=10, weight_range=[0.5, 1.5], seed=42)

Parameters
~~~~~~~~~~
- `n_nodes`: Number of nodes
- `n_edges`: Number of edges
- `weight_range`: Weight range [min, max]
- `seed`: Random seed

Returns
~~~~~~~
- np.ndarray: Generated DAG adjacency matrix

generate_lag_transitions
^^^^^^^^^^^^^^^^^^^^^^^

Generate lag transition matrices for time series data.

Usage
~~~~~

.. code-block:: python

    from cdmir.datasets.utils import generate_lag_transitions
    transitions = generate_lag_transitions(n_nodes=3, max_lag=2, seed=42)

Parameters
~~~~~~~~~~
- `n_nodes`: Number of nodes
- `max_lag`: Maximum lag time
- `seed`: Random seed
- `accept_per`: Acceptable condition number percentile, default=25
- `niter4cond_thresh`: Iterations for condition number threshold, default=1e4

Returns
~~~~~~~
- np.ndarray: Generated lag transition matrix


Usage Examples
-------------

Linear Additive Noise Model Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from cdmir.datasets.simlulators import IIDSimulator
    from cdmir.datasets.utils import erdos_renyi

    # Generate DAG adjacency matrix
    adj_matrix = erdos_renyi(n_nodes=5, n_edges=10, weight_range=[0.5, 1.5], seed=42)

    # Generate linear ANM data
    data = IIDSimulator.simulate_linear_anm(
        weight_mat=adj_matrix,
        n_samples=1000,
        noise_type='gauss',
        noise_scale=0.5,
        seed=42
    )

Nonlinear Additive Noise Model Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from cdmir.datasets.simlulators import IIDSimulator
    from cdmir.datasets.utils import erdos_renyi

    # Generate DAG adjacency matrix
    adj_matrix = erdos_renyi(n_nodes=5, n_edges=10, seed=42)

    # Generate nonlinear ANM data
    data = IIDSimulator.simulate_nonlinear_anm(
        adj_mat=adj_matrix,
        n_samples=1000,
        noise_type='gauss',
        noise_scale=0.5,
        func_type='mlp',
        hidden=100,
        seed=42
    )

Time Series Data
^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from cdmir.datasets.simlulators import TimeLagSimulator
    from cdmir.datasets.utils import generate_lag_transitions

    # Generate lag transition matrix
    transitions = generate_lag_transitions(n_nodes=3, max_lag=2, seed=42)

    # Generate time series data
    data = TimeLagSimulator.simulate_linear_anm(
        weight_mats=transitions,
        max_lag=2,
        length=1000,
        noise_type='gauss',
        noise_scale=0.5,
        seed=42
    )
