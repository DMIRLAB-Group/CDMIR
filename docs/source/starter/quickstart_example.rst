Quickstart Example
==================

Causal Discovery
----------------

.. code-block:: python
    :linenos:

    from causaldmir.datasets.simlulators import IIDSimulator
    from causaldmir.datasets.utils import erdos_renyi

    from causaldmir.discovery.constraint import PC

    from causaldmir.utils.independence import FisherZ

    # Generate the random graph
    weight_mat = erdos_renyi(n_nodes=10, n_edges=20, weight_range=(0.5, 2.0), seed=42)

    # Generate simulation data based on the weight matrix
    X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples=5000, noise_type='gauss', seed=42)

    # Set the alpha value of the PC algorithm
    pc = PC(alpha=0.05)

    # Causal discovery by PC algorithm using FisherZ independence test
    pc.fit(X, indep_cls=FisherZ)

    print(pc.causal_graph)