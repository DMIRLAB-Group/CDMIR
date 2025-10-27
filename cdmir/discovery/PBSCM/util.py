import numpy as np
from scipy.stats import binom
from itertools import product
from math import factorial
import copy
from KDEpy import FFTKDE
from tqdm import tqdm


# binomial thin operator
def thin_operate(pois_component, coef):
    return np.array(binom.rvs(n=pois_component, p=coef))


# derivative function of the PGF binomial distribution
def derivative_func_of_B(alpha, X, n, s):
    # X:count, n:order
    if X < n:
        return 0
    else:
        factorial_item = 1
        for i in range(X, X - n, -1):
            factorial_item *= i
        return factorial_item * alpha ** n * (1 - alpha + alpha * s) ** (X - n)


# derivative function of the PGF poisson distribution
def derivative_func_of_P(mu, n, s):
    return mu ** n * np.exp(mu * (s - 1))


def get_sample_likelihood(coeff, sample, parent_list):
    '''
    get the likelihood of a sample

    Parameters:
    ----------
    coeff: Model parameter
    sample: A sample in the dataset
    parent_list: The list of parent vertices

    Returns:
    ----------
    li_for_sample: Likelihood of the sample
    '''

    max_order = sample[-1]
    num_f = len(parent_list) + 1
    # Padding zero for FFT
    interpolated_zero_num = num_f * max_order + 1
    # the matrix of coefficient of polynomial, which is different order of derivative
    derivative_mat = np.zeros([num_f, interpolated_zero_num])
    for i in range(num_f):
        # i is the noise variable
        if i == num_f - 1:
            for k in range(max_order + 1):
                mu = coeff[-1]
                try:
                    derivative_mat[i, k] = derivative_func_of_P(mu=mu, n=k, s=0) / factorial(k)
                except OverflowError:
                    derivative_mat[i, k] = 0
        # i is the parent variable
        else:
            for k in range(max_order + 1):
                alpha = coeff[parent_list[i]]
                X = sample[parent_list[i]]
                try:
                    derivative_mat[i, k] = derivative_func_of_B(alpha=alpha, X=X, n=k, s=0) / factorial(k)
                except OverflowError:
                    derivative_mat[i, k] = 0
    # calculate by FFT
    fft_res = 1
    for derivative_func in derivative_mat:
        fft_res *= np.fft.fft(derivative_func)

    li_for_sample = float(np.fft.ifft(fft_res)[max_order])
    return li_for_sample


def get_symmetric_matrix(A):
    W = np.maximum(A, A.transpose())
    return W


def one_step_change(edge_mat, e):
    '''
    change the edge_mat by adding , deleting or reverse the edge e

    Parameters:
    ----------
    edge_mat: the original edge_mat
    e: the edge to be changed

    Returns:
    ----------
    new_edge_mat: the new edge_mat
    '''

    j, i = e
    if j == i:
        return edge_mat
    new_edge_mat = edge_mat.copy()

    if new_edge_mat[j, i] == 1:
        new_edge_mat[j, i] = 0
        return new_edge_mat
    else:
        new_edge_mat[j, i] = 1
        new_edge_mat[i, j] = 0
        return new_edge_mat


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
    '''
    generate synthetic data with Poisson noise component and binomial thin operator

    Parameters:
    ----------
    n: Number of vertices
    seed: Random seed
    in_degree_rate: Avg. in-degree rate
    sample_size: Sample size
    alpha_range_str: Range of causal coefficient alpha,
    mu_range_str: Range of parameter mu of Poisson noise component

    Returns:
    ----------
    data.T: Generated data, shape (sample_size, n)
    edge_mat: Adjacency matrix, shape (n, n)
    alpha_mat: Causal coefficient matrix, shape (n, n)
    mu: Parameter mu of Poisson noise component, shape (n,)
    '''

    n = n  # number of vertex
    in_degree_rate = in_degree_rate  # avg. indegree rate
    sample_size = sample_size  # sample size

    if seed is not None:
        rand_state = np.random.RandomState(seed)
        np.random.seed(seed)
    else:
        rand_state = np.random.RandomState()

    edge_mat = np.zeros([n, n])
    edge_select = list(filter(lambda i: i[0] < i[1], product(range(n), range(n))))
    rand_state.shuffle(edge_select)
    for edge_ind in edge_select[:round(in_degree_rate * n)]:
        edge_mat[edge_ind] = 1

    # alpha
    alpha_range = tuple([float(i) for i in alpha_range_str.split(',')])
    alpha_mat = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=(n, n)) * edge_mat

    # mu
    mu_range = tuple([float(i) for i in mu_range_str.split(',')])
    mu = rand_state.uniform(*mu_range, n)

    data = np.random.poisson(lam=np.ones(n) * mu, size=(sample_size, n)).T

    for row in range(n):
        for col in range(n):
            if edge_mat[row][col] == 1:
                data[col] = thin_operate(pois_component=data[row], coef=alpha_mat[row][col]) \
                            + data[col]

    if shuffle:
        shuffle_index = list(range(n))
        rand_state.shuffle(shuffle_index)
        data, edge_mat = shuffle_data(index=shuffle_index, data=data, edge_mat=edge_mat)

    return data.T , edge_mat, alpha_mat, mu


def f(n, k, vis, num, ret, up, subset):
    if num == 1:
        for i in range(up, n):
            if not vis[i]:
                ret[k].append(i)
                subset.append(copy.deepcopy(ret))
                ret[k].remove(i)
        return
    for i in range(up, n):
        if vis[i]: continue
        ret[k].append(i)
        vis[i] = True
        f(n, k, vis, num - 1, ret, i + 1, subset)
        f(n, k + 1, vis, num - 1, ret, ret[k][0], subset)
        vis[i] = False
        ret[k].remove(i)
    return subset


# get the partial of random variable
def get_subset_list(variable_num):
    vis = [False for i in range(variable_num)]
    ret = [[] for i in range(variable_num)]
    subset_list = []
    subset_list = f(variable_num, 0, vis, variable_num, ret, 0, subset_list)
    for list_index in range(len(subset_list)):
        pop_num = 0
        for i in range(len(subset_list[list_index])):
            i = i - pop_num
            if len(subset_list[list_index][i]) == 0:
                subset_list[list_index].pop(i)
                pop_num += 1
            else:
                sorted(subset_list[list_index][i])
        subset_list[list_index].sort(key=sum)
        subset_list[list_index].sort(key=len)

    list2 = []
    for l1 in subset_list:
        if l1 not in list2 and l1[::-1] not in list2:
            list2.append(l1)
    list2.sort(key=len)
    return list2


def get_cumulant(joint):
    '''
    calculate the cumulant 

    Parameters:
    ----------
    joint: random vector

    Returns:
    ----------
    expect_f: cumulant of the random vector
    '''

    order = len(joint)
    # get the partial of random variable
    subset_list = get_subset_list(order)
    expect_f = 0
    for subset in subset_list:
        joint_moment = 1
        for variable_list in subset:
            joint_variable = 1
            for variable in variable_list:
                joint_variable *= joint[variable]
            joint_moment *= np.mean(joint_variable)
        coef = factorial(len(subset) - 1) * (-1) ** (len(subset) - 1)
        expect_f += coef * joint_moment

    return expect_f


# Stirling number, an alternative representation of polynomial coefficients
def sterling2(n, k):
    computed = {}
    key = str(n) + "," + str(k)

    if key in computed.keys():
        return computed[key]
    if n == k == 0:
        return 1
    if (n > 0 and k == 0) or (n == 0 and k > 0):
        return 0
    if n == k:
        return 1
    if k > n:
        return 0
    result = k * sterling2(n - 1, k) + sterling2(n - 1, k - 1)
    computed[key] = result
    return result


def get_path_cum_summation(X, Y, max_order=3):
    '''
    calculate the path cumulation summation of the random vector X and Y

    Parameters:
    ----------
    X: random vector X
    Y: random vector Y
    max_order: The max order of Lambda_k.

    Returns:
    ----------
    The list of path cumulation summation of the random vector X and Y
    '''

    path_cum_list = []
    # 2nd order of 2D slice of cumulant
    joint_variable = [X, Y]
    path_cum_list.append(get_cumulant(joint_variable))

    # 3rd order to max order
    for PA_order in range(1, max_order):
        order = PA_order + 1
        joint_variable.append(Y)
        path_information = get_cumulant(joint_variable)
        for i in range(0, order - 1):
            path_information -= sterling2(order, i + 1) * path_cum_list[i]
        path_cum_list.append(path_information)

    return np.array(path_cum_list)


def path_cum_summation_test(x, y, n_samples, batch, alpha, max_order):
    '''
    calculate the path cumulation summation test of the random vector X and Y

    Parameters:
    ----------
    x: random vector X
    y: random vector Y
    n_samples: The number of samples to be used in the bootstrap test.
    batch: The number of bootstrap samples.
    alpha: confidence level
    max_order: The max order of Lambda_k.

    Returns:   
    ----------
    res: The list of confidence interval of the path cumulation summation of the random vector X and Y
    '''
    subsample_res = np.zeros([max_order, batch])
    # subsample and calculate the cumulant
    for i in range(batch):
        random_indices = np.random.choice(len(x), size=n_samples, replace=True)
        subsample_x = x[random_indices[:n_samples]]
        subsample_y = y[random_indices[:n_samples]]
        res = get_path_cum_summation(subsample_x, subsample_y, max_order=max_order)
        subsample_res[:, i] = res

    subsample_res = subsample_res - subsample_res.mean(axis=1, keepdims=True)
    res = []
    # estimate the distribution of Lambda_k and return the confidence interval
    for i in range(max_order):
        try:
            x, y1 = FFTKDE(bw="silverman").fit(subsample_res[i, :]).evaluate(2 ** 10)
            dx = x[1] - x[0]
            ydx = abs(y1 * dx)
            cdf = np.cumsum(ydx)
            cdf /= cdf[-1]
            lower_index = np.argmax(cdf >= alpha)
            upper_index = np.argmax(cdf >= 1 - alpha)

            lower_bound = x[lower_index]
            upper_bound = x[upper_index]
            res.append([lower_bound, upper_bound])
        except ValueError:
            res.append([0, 0])

    return res













