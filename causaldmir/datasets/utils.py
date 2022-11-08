import numpy as np
import pandas as pd
import networkx as nx
import torch
import random
from typing import List, Tuple, Union, Optional, Sequence


def set_random_seed(seed, set_torch_seed=True):
    # 设置随机数种子
    np.random.seed(seed)
    random.seed(seed)
    if set_torch_seed:
        torch.manual_seed(seed)


def pd2np(pd_data):
    return np.array(pd_data)


def nx2np(nx_data):
    return nx.to_numpy_array(nx_data)


def np2nx(np_data, create_using=None):
    return nx.from_numpy_array(np_data, create_using=create_using)


def leaky_relu(inputs, neg_slope=0.2):
    assert neg_slope > 0
    leaky_relu_1d = lambda x: x if x > 0 else x * neg_slope
    leaky1d = np.vectorize(leaky_relu_1d)
    return leaky1d(inputs)


def _random_permutation(mat):
    # np.random.permutation permutes first axis only
    perm_mat = np.random.permutation(np.eye(mat.shape[0]))
    return perm_mat.T @ mat @ perm_mat


def _random_acyclic_orientation(graph):
    dag = np.tril(_random_permutation(graph), k=-1)
    dag_perm = _random_permutation(dag)
    return dag_perm


def _adj2weights(adj_mat, mat_dim, w_range):
    uni_mat = np.random.uniform(low=w_range[0], high=w_range[1], size=[mat_dim, mat_dim])
    uni_mat[np.random.rand(mat_dim, mat_dim) < 0.5] *= -1  # reverse 50% of the weights
    weight_mat = (adj_mat != 0).astype(float) * uni_mat
    return weight_mat


def check_data(inputs):
    # 检查数据类型，一律转换为 ndarray
    assert isinstance(inputs, (np.ndarray, pd.DataFrame)), "plearse input ndarray or dataframe"
    if isinstance(inputs, pd.DataFrame):
        return pd2np(inputs)
    return inputs


def erdos_renyi(n_nodes: int, n_edges: int, weight_range: Union[Sequence[float], None] = None,
                seed: Optional[int] = None):
    assert n_nodes > 0, "The numbers of nodes must be larger than 0"
    set_random_seed(seed)
    # erdos renyi
    egdes_prob = (n_edges * 2) / (n_nodes ** 2)
    nx_graph = nx.erdos_renyi_graph(n=n_nodes, p=egdes_prob, seed=seed)
    np_graph = nx2np(nx_graph)
    np_dag = _random_acyclic_orientation(np_graph)
    if weight_range is None:
        return np_dag
    else:
        weights = _adj2weights(np_dag, n_nodes, weight_range)
    return weights


def _generate_uniform_mat(n_nodes, cond_thresh):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (n_nodes, n_nodes)) - 1
    for i in range(n_nodes):
        A[:, i] /= np.sqrt(((A[:, i]) ** 2).sum())

    while np.linalg.cond(A) > cond_thresh:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (n_nodes, n_nodes)) - 1
        for i in range(n_nodes):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A


def generate_lag_transitions(n_nodes: int, max_lag: int, seed: Optional[int] = None, accept_per: int = 25,
                             niter4cond_thresh: int = 1e4):
    assert n_nodes > 0
    assert max_lag > 0
    set_random_seed(seed)
    cond_list = []
    for _ in range(int(niter4cond_thresh)):
        A = np.random.uniform(1, 2, (n_nodes, n_nodes))
        for i in range(n_nodes):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        cond_list.append(np.linalg.cond(A))

    cond_thresh = np.percentile(cond_list, accept_per)
    transitions = []
    for lag in range(max_lag):
        B = _generate_uniform_mat(n_nodes, cond_thresh)
        transitions.append(B)
    transitions.reverse()

    return np.array(transitions)
