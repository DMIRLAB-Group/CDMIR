import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import networkx as nx
import torch
import matplotlib.pyplot as plt
import math
import os
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            std1 = np.sqrt(6 / (np.pi * np.pi))
            z = np.random.gumbel(scale=std1, size=n)
            z = scale * z
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-np.sqrt(12) / 2., high=np.sqrt(12) / 2., size=n)
            z = scale * z
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_linear_sem_variable_scale(W, n, sem_type, scale_low=1., scale_high=10.):
    """Simulate samples from linear SEM with specified type of noise whose scale is sampled from unifrom prob.


    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            std1 = np.sqrt(6/(np.pi*np.pi))
            z = np.random.gumbel(scale=std1, size=n)
            z = scale * z
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-np.sqrt(12) / 2., high=np.sqrt(12) / 2., size=n)
            z = scale * z
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    noise_scale = np.random.uniform(low=scale_low, high=scale_high, size=d)
    # print(noise_scale)
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type, noise_dist='uniform', noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        noise_dist:: distribution type of noise term
    Returns:
        X (np.ndarray): [n, d] sample matrix

    """

    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        # z = np.random.normal(scale=scale, size=n)
        # z = np.random.uniform(low=-scale, high=scale, size=n)
        z = None
        if noise_dist == 'gauss':
            z = np.random.normal(scale=1., size=n)
        elif noise_dist == 'exp':
            z = np.random.exponential(scale=1., size=n)
        elif noise_dist == 'gumbel':
            std1 = np.sqrt(6/(np.pi*np.pi))
            z = np.random.gumbel(scale=std1, size=n)
        elif noise_dist == 'uniform':
            z = np.random.uniform(low=-np.sqrt(12) / 2., high=np.sqrt(12) / 2., size=n)
        else:
            raise ValueError('unknown noise distribution type')
        z = z * scale

        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        elif sem_type == 'pow2':
            hidden = 10
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = np.power(X @ W1, 2) @ W2 + z
        elif sem_type == 'pow3':
            hidden = 10
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = np.power(X @ W1, 3) @ W2 + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    # print(noise_scale)
    # print(np.ones(d))
    scale_vec = noise_scale if noise_scale is not None else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X



def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


def drawGraph(adj):
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(len(adj))])
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] == 1:
                G.add_edge(i, j)
    position = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, position)
    nx.draw_networkx_edges(G, position)
    nx.draw_networkx_labels(G, position)
    # nx.draw(G)
    plt.show()


def simulate_pariwise(weight, n, linear=True, distribution='gauss', direct=0, var_cause=1., var_effect=1.):
    """
    生成pairwise数据
    :param direct: 0 means x0 -> x1, 1 means x1->x0
    :param linear:
    :param gauss:
    :param weight:
    :param n:
    :return:
    """
    # variance => standard deviation
    scale_cause = np.sqrt(var_cause)
    scale_effect = np.sqrt(var_effect)
    X = np.zeros([n, 2])
    if distribution == 'gauss':
        X[:, 0] = np.random.normal(loc=0., scale=scale_cause, size=n)
        X[:, 1] = np.random.normal(loc=0., scale=scale_effect, size=n)
    elif distribution == 'uniform':
        limit_cause = np.sqrt(3 * var_cause)
        limit_effect = np.sqrt(3*var_effect)
        X[:, 0] = np.random.uniform(low=-limit_cause, high=limit_cause, size=n)
        X[:, 1] = np.random.uniform(low=-limit_effect, high=limit_effect, size=n)
    elif distribution == 'gumbel':
        X[:, 0] = np.random.gumbel(loc=0., scale=scale_cause, size=n)
        X[:, 1] = np.random.gumbel(loc=0., scale=scale_effect, size=n)
    elif distribution == 'exp':
        X[:, 0] = np.random.exponential(scale=scale_cause, size=n)
        X[:, 1] = np.random.exponential(scale=scale_effect, size=n)
    elif distribution == 'logistic':
        X[:, 0] = np.random.binomial(n=1, p=0.6, size=n)
        X[:, 1] = np.random.binomial(n=1, p=0.6, size=n)
    elif distribution == 'poisson':
        X[:, 0] = np.random.poisson(size=n) * 1.0
        X[:, 1] = np.random.poisson(size=n) * 1.0
    else:
        raise ValueError('unknown distribution type')

    # 生成X0 -> X1
    if linear:
        X[:, 1] += weight * (X[:, 0])
    else:
        X[:, 1] += np.tanh(X[:, 0]) + np.cos(X[:, 0]) + np.sin(X[:, 0])

    # 如果反向，则为对换位置
    if direct == 1:
        X[:, [0,1]] = X[:, [1,0]]

    return X


def entropy_loss_knn(output, input, k=1, EPS=1e-5):
    loss = 0.
    residual = input - output
    d = input.size(1)  # num of nodes
    sample_size = input.size(0)
    # normalization for each xi
    data = torch.zeros_like(residual)
    for i in range(d):
        mu = torch.mean(residual[:, i])
        std = torch.std(residual[:, i])
        data[:, i] = (residual[:, i] - mu) / std
    # sort for quickly searching the nearest neighborhood
    sorted, indices = torch.sort(data, dim=0)
    for i in range(sample_size):
        for j in range(1, k + 1):
            if i < k:
                loss += torch.sum(torch.log(torch.pow(sorted[i] - sorted[i + j], 2) + EPS))
            elif i >= sample_size - k:
                loss += torch.sum(torch.log(torch.pow(sorted[i] - sorted[i - j], 2) + EPS))
            else:
                left = torch.log(torch.pow(sorted[i] - sorted[i - j], 2) + EPS)
                right = torch.log(torch.pow(sorted[i] - sorted[i + j], 2) + EPS)
                loss += torch.sum(left.min(right))
    return loss / k


def entropy_loss_high_order(output, input):
    residual = input - output
    variance = torch.var(residual)
    residual = (residual - torch.mean(residual)) / torch.std(residual)
    x_2 = torch.pow(residual, 2)
    x_3 = torch.pow(residual, 3)
    x_4 = torch.pow(residual, 4)
    k_3 = torch.mean(x_3) / torch.pow(torch.mean(x_2), 3. / 2.)
    k_4 = torch.mean(x_4) / torch.pow(torch.mean(x_2), 4. / 2.)

    loss = 1 / 2 * torch.log(variance) - 1 / 12 * torch.pow(k_3, 2) - 1 / 48 * torch.pow(k_4, 2)
    return loss


def entropy_loss_G1(output, input):
    residual = input - output
    residual = (residual - torch.mean(residual)) / torch.std(residual)
    loss = -torch.pow(torch.mean(torch.log(torch.cosh(residual))), 2)
    return loss


def entropy_loss_G1_high_order(output, input):
    residual = input - output
    residual = (residual - torch.mean(residual)) / torch.std(residual)
    x_2 = torch.pow(residual, 2)
    x_3 = torch.pow(residual, 3)
    x_4 = torch.pow(residual, 4)
    k_3 = torch.mean(x_3) / torch.pow(torch.mean(x_2), 3. / 2.)
    k_4 = torch.mean(x_4) / torch.pow(torch.mean(x_2), 4. / 2.)
    variance = torch.var(residual)
    loss = 1 / 2 * torch.log(variance) - 1 / 12 * torch.pow(k_3, 2) - 1 / 48 * torch.pow(k_4, 2) \
           - torch.pow(torch.mean(torch.log(torch.cosh(residual))), 2)
    return loss


# Based on NIPS*97 paper, www.cs.helsinki.fi/u/ahyvarin/papers/NIPS97.pdf
def entropy_loss_mentappr(output, input):
    residual = input - output
    residual_std = torch.std(residual)
    residual = (residual - torch.mean(residual)) / residual_std
    # const
    # gaussianEntropy = math.log(2. * math.pi) / 2. + 1 / 2
    gaussianEntropy = 1.4189385332046727
    # k1 = 36. / (8 * math.sqrt(3.) - 9.)
    k1 = 7.412888581800331
    # gamma = 0.37457
    # k2 = 79.047
    # negentropy
    # negentropy = k2 * torch.pow((torch.mean(torch.log(torch.cosh(residual))) - gamma), 2) \
    #              + k1 * torch.pow(torch.mean(residual * torch.exp(-residual * residual / 2)), 2)
    # k2,con=24./(16.*math.sqrt(3)-27.),math.sqrt(1./2.)
    k2, con = 33.66942333606287, 0.7071067811865476
    negentropy = k2 * torch.pow(torch.mean(torch.exp(-torch.pow(residual, 2) / 2.)) - con, 2) \
                 + k1 * torch.pow(torch.mean(residual * torch.exp(-residual * residual / 2)), 2)
    # entropy
    entropy = gaussianEntropy - negentropy + torch.log(residual_std)
    return entropy



def variance_loss(output, input):
    residual = input - output
    residual_std = torch.std(residual)
    return residual_std


def simulate_Test(n):
    X = np.zeros([n, 3])

    X[:, 0] = np.random.normal(loc=0., scale=1., size=n)
    X[:, 1] = np.random.normal(loc=0., scale=1., size=n)
    X[:, 2] = np.random.normal(loc=0., scale=1., size=n)

    X[:, 1] += 3 * (X[:, 0])
    X[:, 2] += 0.6 * (X[:, 1]) + 1 * (X[:, 0])
    return X


def set_seed(seed=None):
    if seed is None:
        seed = 123
    # for pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for np
    np.random.seed(seed)
    random.seed(seed)
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)

def draw_line_pdf(num_line, data, index_X, data_name, xlabel, ylabel, ylim=(-0.05, 1.09), loc=(0.7,0.7),
                  file_name = None, is_show=False, is_save=False, title=None):
    """
    df = pd.DataFrame(np.array([[0.2       , 0.4       , 0.9       , 0.94      , 0.94      ,
        0.98      , 1.        ],
       [0.33      , 0.39      , 0.82      , 0.92      , 0.92      ,
        0.99      , 1.        ],
       [0.32666667, 0.39333333, 0.78666667, 0.86666667, 0.85333333,
        0.94666667, 0.99333333],
       [0.325     , 0.465     , 0.785     , 0.855     , 0.83      ,
        0.965     , 0.995     ]]),index=['0.5', '1.0', '1.5', '2.0'],columns=['PCMCI', 'NHPC', 'MLE_SGL', 'ADM4', 'TTHP_NT', 'TTHP_S', 'TTHP'])

    :param num_line:
    :param data:
    :param index_X:
    :param data_name:
    :param xlabel:
    :param ylabel:
    :param file_name:
    :return:
    """
    error_config = {'ecolor': '0.3', 'capsize': 2}
    marker = ['-x', '-D', '-*', '-s', '-v', '-o', '-^']
    marker = marker[:num_line]
    df = pd.DataFrame(data, index=index_X, columns=data_name,)
    ax = df.plot(kind='line', style=marker,figsize=(4,2.3), ylim=ylim, rot=0) # 这里面的line可以改成bar之类的
    ax.grid( linestyle="dotted") # 设置背景线
    ax.legend(fontsize=9, loc=loc) # 设置图例位置
    ax.set_xlabel(xlabel, fontsize='13')
    ax.set_ylabel(ylabel, fontsize='13')
    if title is not None:
        plt.title(label=title, fontsize='17', loc='center')
    if is_save and file_name is not None:
        plt.savefig("graph/" + file_name, format='pdf', bbox_inches='tight')
        df.to_excel("graph/" + file_name + '.xlsx')
    if is_show:
        plt.show()

def readcsv(files):
    data = pd.read_csv(files)
    data_np = np.array(data.values)
    return data_np

def savecsv(data,file_name):
    df = pd.DataFrame(data)
    df.to_excel("data/" + file_name + '.xlsx', encoding="UTF-8")


def draw_bar_pdf(num_line, data, data_std, index_X, data_name, xlabel, ylabel, ylim=(-0.05, 1.09), loc=(0.7,0.7),
                  file_name = None, is_show=False, is_save=False, title=None, bar=True):
    """
    df = pd.DataFrame(np.array([[0.2       , 0.4       , 0.9       , 0.94      , 0.94      ,
        0.98      , 1.        ],
       [0.33      , 0.39      , 0.82      , 0.92      , 0.92      ,
        0.99      , 1.        ],
       [0.32666667, 0.39333333, 0.78666667, 0.86666667, 0.85333333,
        0.94666667, 0.99333333],
       [0.325     , 0.465     , 0.785     , 0.855     , 0.83      ,
        0.965     , 0.995     ]]),index=['0.5', '1.0', '1.5', '2.0'],columns=['PCMCI', 'NHPC', 'MLE_SGL', 'ADM4', 'TTHP_NT', 'TTHP_S', 'TTHP'])

    :param num_line:
    :param data:
    :param index_X:
    :param data_name:
    :param xlabel:
    :param ylabel:
    :param file_name:
    :return:
    """
    error_config = {'ecolor': '0.3', 'capsize': 2}
    marker = ['-x', '-D', '-*', '-s', '-v', '-o', '-^']
    marker = marker[:num_line]
    df = pd.DataFrame(data, index=index_X, columns=data_name)
    df_std = pd.DataFrame(data_std,  index=index_X, columns=data_name)
    if bar == True:
        ax = df.plot(kind='bar', style=marker, yerr=df_std, figsize=(4,2.3), ylim=ylim, rot=0, error_kw=error_config)
    else:

        ax = df.plot(kind='line', style=marker, yerr=df_std, capsline='dash' ,capthick=4, capsize=2,figsize=(4,2.3), ylim=ylim, rot=0)

    ax.grid( linestyle="dotted") # 设置背景线
    ax.legend(fontsize=7,loc='best') # 设置图例位置
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title is not None:
        plt.title(label=title, fontsize='17', loc='center')
    if is_save and file_name is not None:
        plt.savefig("graph/" + file_name + '.pdf', format='pdf', bbox_inches='tight')
        df.to_excel("graph/" + file_name + '.xlsx')
        df_std.to_excel("graph/" + file_name + '_std.xlsx')
    if is_show:
        plt.show()
