from scipy.stats import binom
from itertools import product
from copy import deepcopy
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from scipy import stats

# binomial thin operator
def thin_operate(pois_component, coef):
    return np.array(binom.rvs(n=pois_component, p=coef))

def check_orientate(orientate_res):
    seen = {}
    for index, (i, j) in enumerate(orientate_res):
        if j in seen:
            return True, [seen[j], index]
        seen[j] = index
    return False, []


def shuffle_data(index, data, edge_mat):
    ret_data = np.zeros_like(data)
    ret_mat = np.zeros_like(edge_mat)
    for i in range(len(index)):
        ret_data[i] = data[index[i]]

    for i in range(len(index)):
        for j in range(len(index)):
            orgin_i = index[i]
            orgin_j = index[j]
            ret_mat[i, j] = edge_mat[orgin_i, orgin_j]
    return ret_data, ret_mat


def data_generate(n=5, seed=1, in_degree_rate=1.5, sample_size=10000,
                  alpha_range_str="0.2, 1", mu_range_str="1,3", shuffle=False):
    """
    :param n: Number of vertices
    :param seed: Random seed
    :param in_degree_rate: Avg. indegree rate
    :param sample_size: Sample size
    :param alpha_range_str: Range of causal coefficient alpha,
    :param mu_range_str: Range of parameter mu of Poisson noise component
    :param shuffle:
    :param in_degree_rule:
    :return: data, edge_mat, alpha_mat, mu
    """
    n = n  # number of vertex
    in_degree_rate = in_degree_rate  # avg. indegree rate
    sample_size = sample_size  # sample size

    if seed is not None:
        rand_state = np.random.RandomState(seed)
        np.random.seed(seed)
    else:
        rand_state = np.random.RandomState()

    edge_mat = np.zeros([n, n], dtype=int)

    edge_select = list(filter(lambda i: i[0] < i[1], product(range(n), range(n))))
    rand_state.shuffle(edge_select)
    for edge_ind in edge_select:
        if edge_mat.sum() >= round(in_degree_rate * n):
            break
        edge_mat[edge_ind] = 1

    # generating alpha
    alpha_range = tuple([float(i) for i in alpha_range_str.split(',')])
    alpha_mat = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=(n, n)) * edge_mat

    # generating mu
    mu_range = tuple([float(i) for i in mu_range_str.split(',')])
    mu = rand_state.uniform(*mu_range, n)

    data = np.random.poisson(lam=np.ones(n) * mu, size=(sample_size, n)).T

    for row in range(n):
        for col in range(n):
            if edge_mat[row][col] == 1:  # row->col
                data[col] = thin_operate(pois_component=data[row], coef=alpha_mat[row][col]) \
                            + data[col]

    if shuffle:
        shuffle_index = list(range(n))
        rand_state.shuffle(shuffle_index)
        data, edge_mat = shuffle_data(index=shuffle_index, data=data, edge_mat=edge_mat)

    return data.T, edge_mat, alpha_mat, mu


def find_triangle_in_undirected_graph(skeleton_mat):
    """Finds all triangles in an undirected graph using K3 Algorithm."""
    skeleton = np.copy(skeleton_mat)
    n = len(skeleton)
    degree = {i: sum(skeleton[i][j] for j in range(n)) for i in range(n)}
    sorted_degree = sorted(degree, reverse=True)
    mark = [0 for _ in range(n)]
    ret = []
    for u in sorted_degree:
        neigh = [i for i in range(n) if skeleton[u][i] != 0]
        for v in neigh: mark[v] = 1
        for v in neigh:
            for w in range(n):
                if w != u and skeleton[v][w] != 0 and mark[w] and v < w:
                    ret.append((u, v, w))
        for v in neigh: mark[v] = 0
        for v in neigh: skeleton[v][u] = 0  # Remove u from G
    return ret


def diff_get_epgf_gradient_np(data, z, order_list):
    """Compute the gradient of the empirical probability generating function (EPGF) using numpy."""
    z = np.array(z)
    n, m = data.shape
    data_ = deepcopy(data)

    tmp = np.ones_like(data_)
    for i in range(n):
        o = order_list[i]
        for j in range(o):
            tmp[i, :] *= data_[i, :] - j
        data_[i, :] -= o

    power_matrix = np.power(z[:, np.newaxis], data_) * tmp
    product_per_sample = np.prod(power_matrix, axis=0)
    epgf_gradient = np.mean(product_per_sample)
    return epgf_gradient


def get_epgf(data, z):
    """Estimate the empirical probability generating function (EPGF) using numpy."""
    z = np.array(z)
    power_matrix = np.power(z[:, np.newaxis], data)
    product_per_sample = np.prod(power_matrix, axis=0)
    epgf = np.mean(product_per_sample)

    return epgf


def get_epgf_torch(data, z_tensor):
    """Estimate the empirical probability generating function using torch."""
    data_tensor = torch.tensor(data, dtype=torch.float64)

    power_matrix = torch.pow(z_tensor[:, None], data_tensor)
    product_per_sample = torch.prod(power_matrix, axis=0)
    epgf = torch.mean(product_per_sample)

    return epgf


def diff_get_epgf_gradient(data, z_list, order_list):
    """Compute the gradient of the empirical probability generating function using torch."""
    z = torch.tensor(z_list, requires_grad=True, dtype=torch.float64)
    epgf = get_epgf_torch(data, z)

    # Compute higher-order derivatives based on order_list
    for i, order in enumerate(order_list):
        for _ in range(int(order)):
            if epgf.grad_fn is None:  # If epgf does not depend on z
                return 0
            # Compute the gradient for the current variable
            epgf = torch.autograd.grad(epgf, z, create_graph=True, retain_graph=True, allow_unused=True)[0][i]

    return epgf.detach().item() if epgf is not None else 0


def get_mat_S(data, i, j):
    n = data.shape[0]
    order_list = np.zeros(n).astype(int)

    z_list = [0.05] * n
    z_list[i], z_list[j] = 1, 1

    A = get_epgf(data, z_list)
    order_list[i], order_list[j] = 1, 0
    B = diff_get_epgf_gradient_np(data, z_list, order_list=order_list)
    order_list[i], order_list[j] = 0, 1
    C = diff_get_epgf_gradient_np(data, z_list, order_list=order_list)
    order_list[i], order_list[j] = 1, 1
    D = diff_get_epgf_gradient_np(data, z_list, order_list=order_list)
    mat = np.array([[A, B],
                    [C, D]])

    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    res = S[1]

    return res

def subsample_task(data, i, j):
    n, m = data.shape
    random_indices = np.random.choice(m, size=int(0.75 * m), replace=True)
    selected_data = data[:, random_indices]
    return get_mat_S(selected_data, i, j)


def parallel_subsample(data, batch, i, j):
    subsample_res = np.zeros(batch)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(subsample_task, data, i, j) for k in range(batch)]
        for k, future in enumerate(futures):
            subsample_res[k] = future.result()

    return subsample_res

def test_adjacent_rank_bootstrap_gaussian(data, i, j, batch=20, alpha=0.1):
    subsample_res = parallel_subsample(data, batch, i, j)
    original_res = get_mat_S(data, i, j)

    sigma = np.sqrt(np.var(subsample_res))
    Z = stats.norm.ppf(1 - alpha / 2)

    lower_bound = -Z * sigma
    upper_bound = Z * sigma

    interval = [lower_bound, upper_bound, original_res]
    if lower_bound < original_res < upper_bound:
        return False, interval
    else:
        return True, interval