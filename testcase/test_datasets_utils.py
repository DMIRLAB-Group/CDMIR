import numpy as np
import networkx as nx

from causal_dmir.datasets.utils import erdos_renyi, np2nx, generate_lag_transitions


def test_erdos_renyi():
    n_nodes = 10
    n_edges = 20
    seed = 10
    dag = erdos_renyi(n_nodes, n_edges, seed=seed)
    assert isinstance(dag, np.ndarray)
    assert dag.shape == (n_nodes, n_nodes)
    assert ((dag == 0.0) + (dag == 1.0)).all()
    assert nx.is_directed_acyclic_graph(np2nx(dag, create_using=nx.DiGraph))


def test_erdos_renyi_weights():
    n_nodes = 10
    n_edges = 20
    seed = 10
    weight_range = (0.5, 2.0)
    dag = erdos_renyi(n_nodes, n_edges, weight_range, seed)
    assert isinstance(dag, np.ndarray)
    assert dag.shape == (n_nodes, n_nodes)
    assert ((np.abs(dag) == 0.0) + ((np.abs(dag) >= weight_range[0]) * (np.abs(dag) < weight_range[1]))).all()
    assert nx.is_directed_acyclic_graph(np2nx(dag, create_using=nx.DiGraph))


def test_erdos_renyi_error():
    n_nodes = -10
    n_edges = 20
    seed = 10
    try:
        dag = erdos_renyi(n_nodes, n_edges, seed=seed)
    except Exception as e:
        assert isinstance(e, AssertionError)


def test_generate_lag_transitions():
    n_nodes = 10
    max_lag = 3
    seed = 10
    transitions = generate_lag_transitions(n_nodes, max_lag, seed=seed)
    assert isinstance(transitions, np.ndarray)
    assert transitions.shape == (max_lag, n_nodes, n_nodes)
