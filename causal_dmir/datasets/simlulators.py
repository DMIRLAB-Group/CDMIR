import networkx as nx
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import List, Tuple, Union, Optional

try:
    from scipy.special import expit as sigmoid
except:
    def sigmoid(inputs):
        return 1 / (1 + np.exp(-inputs))

from ..datasets import utils


def _pnl_func(func_type, d_inputs, d_outputs, neg_slope=0.2):
    if func_type == 'tanh':
        return np.tanh
    elif func_type == 'leaky_relu':
        return lambda x: utils.leaky_relu(x, neg_slope)
    elif func_type == 'sin':
        return lambda x: np.sin(x)
    else:
        W1 = np.random.uniform(low=0.5, high=2.0, size=[d_inputs, d_outputs])
        W1[np.random.rand(*W1.shape) < 0.5] *= -1
        return lambda x: x @ W1


def _pnl_func_sequential(func_type, d_inputs, d_outputs, neg_slope=0.2, hidden=None):
    if isinstance(func_type, str):
        func = _pnl_func(func_type, d_inputs, d_outputs, neg_slope)
        return func
    else:
        if len(func_type) == 0:
            return lambda x: x
        last_linear = (func_type[0] == 'linear' and 'linear' not in func_type[1:])
        hidden = hidden if 'linear' in func_type[1:] else d_outputs
        layer_d_outputs = d_outputs if last_linear else hidden
        func = _pnl_func(func_type[0], d_inputs, layer_d_outputs, neg_slope)
        follow_func = _pnl_func_sequential(func_type[1:], layer_d_outputs, d_outputs, neg_slope, hidden)

        return lambda x: follow_func(func(x))


def _gen_noise(size, noise_type='gauss', noise_scale=1.0):
    if noise_type == 'gauss':
        noise = np.random.normal(scale=1.0, size=size)
    elif noise_type == 'exp':
        noise = np.random.exponential(scale=1.0, size=size)
    elif noise_type == 'gumbel':
        noise = np.random.gumbel(scale=1.0, size=size)
    elif noise_type == 'uniform':
        noise = np.random.uniform(low=-1.0, high=1.0, size=size)
    elif noise_type == 'laplace':
        noise = np.random.laplace(scale=1.0, size=size)
    else:
        raise ValueError('Unknown noise type! The options are follows: gauss, exp, \
                             gumbel, uniform, logistic or poisson(only for a linear model)')
    return noise * np.array(noise_scale)


class IIDSimulator:  # generate the I.I.D. dataset

    def __init__(self):
        pass

    @staticmethod
    def _simulate_single_value_linear_anm(n_samples, parents, weights, noise_type='gauss', noise_scale=1.0):
        if noise_type == 'logistic':
            value = np.random.binomial(1, sigmoid(parents @ weights)) * 1.0
        elif noise_type == 'poisson':
            value = np.random.poisson(np.exp(parents @ weights)) * 1.0
        else:
            noise = _gen_noise(n_samples, noise_type, noise_scale)
            value = parents @ weights + noise
        return value

    @staticmethod
    def simulate_linear_anm(weight_mat: np.ndarray, n_samples: Optional[int] = None,
                            noise_type: Union[str, List[str]] = 'gauss',
                            noise_scale: Union[float, int, np.ndarray, list, None] = None,
                            seed: Optional[int] = None) -> np.array:
        """
        simulate the dataset from linear additive-noise model with specified type of noise
        :param weight_mat: np.ndarray, the weighted adjacent matrix of shape [n_nodes, n_nodes]
        :param n_samples: int, number of samples
        :param noise_type: str or list, type of noise, e.g , 'gauss', 'exp', 'gumbel', 'uniform', 'logistic',
                    'laplace' or 'poisson'
        :param noise_scale: float, np.array or list, scale parameter of noise
        :param seed:
        :return:
            np.ndarray, samples of shape [n_samples, n_nodes]
        """
        assert noise_type in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'laplace', 'poisson']
        weight_mat = utils.check_data(weight_mat)
        if seed is not None:
            utils.set_random_seed(seed)
        n_nodes = weight_mat.shape[0]
        n_samples = np.inf if n_samples is None else n_samples

        noise_scale = 1.0 if noise_scale is None else noise_scale
        if np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(n_nodes)
        elif len(noise_scale) == n_nodes:
            for ns in noise_scale:
                assert isinstance(ns, (int, float))
            scale_vec = noise_scale
        else:
            raise ValueError('noise scale must be a scalar or has length d')

        nx_graph = utils.np2nx(weight_mat, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(nx_graph):
            raise ValueError('the weight matrix must be a DAG')
        if np.isinf(n_samples):
            if noise_type == 'gauss':
                # make 1/d X'X = true cov
                X = np.sqrt(n_nodes) * np.diag(scale_vec) @ np.linalg.inv(np.eye(n_nodes) - weight_mat)
                return X
            else:
                raise ValueError('population risk not available')

        topo_order = list(nx.topological_sort(nx_graph))
        assert len(topo_order) == n_nodes
        X = np.zeros([n_samples, n_nodes])
        for j in topo_order:
            pare_idx = list(nx_graph.predecessors(j))
            X[:, j] = IIDSimulator._simulate_single_value_linear_anm(n_samples=n_samples, parents=X[:, pare_idx],
                                                                     weights=weight_mat[pare_idx, j],
                                                                     noise_type=noise_type, noise_scale=scale_vec[j])
        return X

    @staticmethod
    def _simulate_single_value_nonlinear_anm(
            n_samples, parents, noise_type='gauss', noise_scale=1.0, seed=None, func_type='mlp', hidden=None):
        noise = _gen_noise(n_samples, noise_type, noise_scale)
        pare_size = parents.shape[1]
        if pare_size == 0:
            return noise
        if func_type == 'mlp':
            hidden = 100 if hidden is None else hidden
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pare_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            value = sigmoid(parents @ W1) @ W2 + noise
        elif func_type == 'mim':
            W1 = np.random.uniform(low=0.5, high=2.0, size=pare_size)
            W1[np.random.rand(pare_size) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=pare_size)
            W2[np.random.rand(pare_size) < 0.5] *= -1
            W3 = np.random.uniform(low=0.5, high=2.0, size=pare_size)
            W3[np.random.rand(pare_size) < 0.5] *= -1
            value = np.tanh(parents @ W1) + np.cos(parents @ W2) + np.sin(parents @ W3) + noise
        # elif func_type == 'gp':  # ysa
        #     gp = GaussianProcessRegressor()
        #     value = gp.sample_y(parents, random_state=seed).flatten() + noise
        # elif func_type == 'gp-add':
        #     gp = GaussianProcessRegressor()
        #     value = sum([gp.sample_y(parents[:, i, None], random_state=seed).flatten()
        #                  for i in range(parents.shape[1])]) + noise
        else:
            raise ValueError('Unknown function type. The options are follows: mlp or mim. ')
        return value

    @staticmethod
    def simulate_nonlinear_anm(
            adj_mat: np.ndarray, n_samples: Optional[int] = None, noise_type: Union[str, List[str]] = 'gauss',
            noise_scale: Union[float, int, np.ndarray, list, None] = None, seed: Optional[int] = None,
            func_type: str = 'mlp', hidden: Optional[int] = None):
        """
        simulate the dataset from non-linear additive-noise model with specified type of noise
        :param adj_mat: np.ndarray, the adjacent matrix of shape [n_nodes, n_nodes]
        :param n_samples: int, number of samples
        :param noise_type: str or list, type of noise, e.g , 'gauss', 'exp', 'gumbel', 'uniform', 'laplace'
        :param noise_scale: float, np.array or list, scale parameter of noise
        :param seed:
        :param func_type: str, type of non-linear function, e.g, 'mlp', 'mim'
        :param hidden: int, the numbers of hidden linear layers in non-linear funciton
        :return:
            np.ndarray, samples of shape [n_samples, n_nodes]
        """
        assert noise_type in ['gauss', 'exp', 'gumbel', 'uniform', 'laplace']
        assert func_type in ['mlp', 'mim']  # ['mlp', 'mim', 'gp', 'gp-add']
        if seed is not None:
            utils.set_random_seed(seed)

        n_nodes = adj_mat.shape[0]
        n_samples = np.inf if n_samples is None else n_samples
        adj_mat = (adj_mat != 0).astype(int)

        noise_scale = 1.0 if noise_scale is None else noise_scale
        if np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(n_nodes)
        elif len(noise_scale) == n_nodes:
            for ns in noise_scale:
                assert isinstance(ns, (int, float))
            scale_vec = noise_scale
        else:
            raise ValueError('noise scale must be a scalar or has length d')

        nx_graph = utils.np2nx(adj_mat, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(nx_graph):
            raise ValueError('the weight matrix must be a DAG')

        topo_order = list(nx.topological_sort(nx_graph))
        assert len(topo_order) == n_nodes
        X = np.zeros([n_samples, n_nodes])
        for j in topo_order:
            parents = list(nx_graph.predecessors(j))
            X[:, j] = IIDSimulator._simulate_single_value_nonlinear_anm(n_samples, X[:, parents], noise_type,
                                                                        scale_vec[j], hidden=hidden)
        return X

    @staticmethod
    def _simulate_single_value_pnl(
            n_samples, parents, noise_type='gauss', noise_scale=1.0, func1_type='tanh',
            func2_type='leaky_relu', neg_slope=0.2, hidden=None):
        pare_size = parents.shape[1]
        if hidden is None:
            hidden = pare_size
        elif hidden != pare_size:
            if ('linear' not in func1_type or 'linear' not in func2_type) \
                    and ('linear' != func1_type or 'linear' != func2_type):
                raise ValueError("func1 and func2 must contain linear layer if you want to set the hidden size")
        if pare_size == 0:
            noise = _gen_noise(n_samples, noise_type, noise_scale)
            return noise
        else:
            noise = _gen_noise([n_samples, hidden], noise_type, noise_scale)
        func1 = _pnl_func_sequential(func1_type, pare_size, hidden, neg_slope, hidden)
        func2 = _pnl_func_sequential(func2_type, hidden, pare_size, neg_slope, hidden)
        value = np.sum(func2(func1(parents) + noise))
        return value

    @staticmethod
    def simulate_pnl(
            adj_mat: np.ndarray, n_samples: int, noise_type: Union[str, List[str]] = 'gauss',
            noise_scale: Union[float, int, np.ndarray, list, None] = None, seed: Optional[int] = None,
            func1_type: Union[List[str], str] = 'tanh', func2_type: Union[List[str], str] = 'leaky_relu',
            neg_slope: float = 0.2, hidden: Optional[int] = None):
        """
        simulate the dataset from post non-linear model with specified type of noise
        :param adj_mat: np.ndarray, the adjacent matrix of shape [n_nodes, n_nodes]
        :param n_samples: int, number of samples
        :param noise_type: str or list, type of noise, e.g , 'gauss', 'exp', 'gumbel', 'uniform', 'laplace'
        :param noise_scale: float, np.array or list, scale parameter of noise
        :param seed:
        :param func1_type: str or list, type of non-linear function, e.g, 'tanh', 'leaky_relu', 'cos', 'linear'
                if 'func1_type' is a list, a sequence of non-linear function will be created, e.g, ['linear', 'tanh', 'linear']
        :param func2_type: str or list, type of non-linear function, e.g, 'tanh', 'leaky_relu', 'cos', 'linear'
                if 'func2_type' is a list, a sequence of non-linear function will be created, e.g, ['linear', 'tanh', 'linear']
        :param neg_slope: float, the negative slop of leaky relu function
        :param hidden: int, the numbers of hidden linear layers in non-linear funciton
        :return:
            np.ndarray, samples of shape [n_samples, n_nodes]
        """
        assert noise_type in ['gauss', 'exp', 'gumbel', 'uniform', 'laplace']
        assert isinstance(noise_type, (str, list)), "The type of 'noise_type' must be str or list"
        func_list = ['tanh', 'leaky_relu', 'cos', 'linear']
        for func_type in [func1_type, func2_type]:
            if isinstance(func_type, str):
                assert func_type in func_list
            else:
                assert set(func_type) <= set(func_list)
        if seed is not None:
            utils.set_random_seed(seed)
        n_nodes = adj_mat.shape[0]

        noise_scale = 1.0 if noise_scale is None else noise_scale
        if np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(n_nodes)
        elif len(noise_scale) == n_nodes:
            for ns in noise_scale:
                assert isinstance(ns, (int, float))
            scale_vec = noise_scale
        else:
            raise ValueError('noise scale must be a scalar or has length d')

        nx_graph = utils.np2nx(adj_mat, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(nx_graph):
            raise ValueError('the weight matrix must be a DAG')

        topo_order = list(nx.topological_sort(nx_graph))
        assert len(topo_order) == n_nodes
        X = np.zeros([n_samples, n_nodes])
        for j in topo_order:
            pare_idx = list(nx_graph.predecessors(j))
            X[:, j] = IIDSimulator._simulate_single_value_pnl(n_samples=n_samples,
                                                              parents=X[:, pare_idx],
                                                              noise_type=noise_type,
                                                              noise_scale=scale_vec[j],
                                                              func1_type=func1_type,
                                                              func2_type=func2_type,
                                                              neg_slope=neg_slope,
                                                              hidden=hidden)
        return X


class TimeLagSimulator:  # generate the time series dataset

    def __init__(self):
        pass

    @staticmethod
    def _simulate_single_time_linear_anm(
            n_nodes, value_lags, weight_mats, max_lag, noise_type='gauss', noise_scale=1.0):
        value_t = np.zeros(n_nodes)
        for lag in range(max_lag):
            value_t += value_lags[lag] @ weight_mats[lag]

        if noise_type == 'logistic':
            value_t = np.random.binomial(1, sigmoid(value_t)) * 1.0
        # elif noise_type == 'poisson':
        #     value_t = np.random.poisson(np.exp(value_t)) * 1.0
        else:
            noise = _gen_noise(n_nodes, noise_type, noise_scale)
            value_t += noise
        value_lags = np.concatenate((value_lags, value_t[np.newaxis, :]), axis=0)[1:, :]

        return value_lags, value_t

    @staticmethod
    def simulate_linear_anm(
            weight_mats: np.ndarray, max_lag: Optional[int] = None, length: Optional[int] = None,
            noise_type: Union[str, List[str]] = 'gauss', noise_scale: Union[float, int, np.ndarray, list, None] = None,
            seed: Optional[int] = None):
        """
        simulate the dataset from linear additive-noise model for time lags with specified type of noise
        :param weight_mats: np.ndarray, the weighted adjacent matrix of shape [n_nodes, n_nodes]
        :param max_lag: int, the max lag
        :param length: int, the length of generated data
        :param noise_type: str or list, type of noise, e.g , 'gauss', 'exp', 'gumbel', 'uniform', 'laplace', 'logistic'
        :param noise_scale: float, np.array or list, scale parameter of noise
        :param seed:
        :return:
            np.ndarray, samples of shape [length+max_lag, n_nodes]
        """
        assert noise_type in ['gauss', 'exp', 'gumbel', 'uniform', 'laplace', 'logistic']
        length = 5000 if length is None else length
        n_nodes = weight_mats.shape[1]
        if seed is not None:
            utils.set_random_seed(seed)
        if max_lag is None:
            max_lag = weight_mats.shape[0]
        elif max_lag != weight_mats.shape[0]:
            raise ValueError('The max time lag must be equal to the shape 0 of weight matrix')

        noise_scale = 1.0 if noise_scale is None else noise_scale
        if np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(n_nodes)
        elif len(noise_scale) == n_nodes:
            for ns in noise_scale:
                assert isinstance(ns, (int, float))
            scale_vec = noise_scale
        else:
            raise ValueError('noise scale must be a scalar or has length d')

        Xt_ = []
        X_lags = _gen_noise((max_lag, n_nodes), 'gauss' if noise_type is 'logistic' else noise_type, scale_vec)
        for lag in range(max_lag):
            Xt_.append(X_lags[lag, :])

        for t in range(length):
            X_lags, X_t = TimeLagSimulator._simulate_single_time_linear_anm(
                n_nodes, X_lags, weight_mats, max_lag, noise_type, scale_vec)
            Xt_.append(X_t)
        Xt_ = np.array(Xt_)

        return Xt_

    @staticmethod
    def _simulate_single_value_nonlinear_anm(
            pare_lags, func_type='mlp', hidden=None
    ):
        pare_size = pare_lags.shape[1]
        if pare_size == 0:
            return 0.0
        if func_type == 'mlp':
            hidden = 100 if hidden is None else hidden
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pare_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            value = sigmoid(np.sum(pare_lags @ W1, axis=0)) @ W2
        elif func_type == 'mim':
            W1 = np.random.uniform(low=0.5, high=2.0, size=pare_size)
            W1[np.random.rand(pare_size) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=pare_size)
            W2[np.random.rand(pare_size) < 0.5] *= -1
            W3 = np.random.uniform(low=0.5, high=2.0, size=pare_size)
            W3[np.random.rand(pare_size) < 0.5] *= -1
            value = np.tanh(np.sum(pare_lags @ W1)) \
                    + np.cos(np.sum(pare_lags @ W2)) \
                    + np.sin(np.sum(pare_lags @ W3))
        else:
            raise ValueError('Unknown function type. The options are follows: mlp or mim. ')
        return value

    @staticmethod
    def _simulate_single_time_nonlinear_anm(
            n_nodes, value_lags, adj_mats, noise_type='gauss', noise_scale=1.0, func_type='mlp', hidden=None):
        value_t = np.zeros(n_nodes)
        noise = _gen_noise(n_nodes, noise_type, noise_scale)

        for j in range(n_nodes):
            value_t[j] = TimeLagSimulator._simulate_single_value_nonlinear_anm(
                np.multiply(value_lags, adj_mats[:, :, j]), func_type, hidden)
        value_t += noise
        value_lags = np.concatenate((value_lags, value_t[np.newaxis, :]), axis=0)[1:, :]

        return value_lags, value_t

    @staticmethod
    def simulate_nonlinear_anm(
            adj_mats: np.ndarray, max_lag: Optional[int], length: Optional[int] = None,
            noise_type: Union[str, List[str]] = 'gauss', noise_scale: Union[float, int, np.ndarray, list, None] = None,
            seed: Optional[int] = None, func_type: str = 'mlp',
            hidden: Optional[int] = None):
        """
        simulate the dataset from non-linear additive-noise model for time lags with specified type of noise
        :param adj_mats: np.ndarray, the adjacent matrix of shape
        :param max_lag: int, the max lag
        :param length: int, the length of generated data
        :param noise_type: str or list, type of noise, e.g , 'gauss', 'exp', 'gumbel', 'uniform', 'laplace'
        :param noise_scale: float, np.array or list, scale parameter of noise
        :param seed:
        :param func_type: str, type of non-linear function, e.g, 'mlp', 'mim'
        :param hidden: int, the numbers of hidden linear layers in non-linear funciton
        :return:
            np.ndarray, samples of shape [length+max_lag, n_nodes]
        """
        assert noise_type in ['gauss', 'exp', 'gumbel', 'uniform', 'laplace']
        assert func_type in ['mlp', 'mim']
        length = 5000 if length is None else length
        n_nodes = adj_mats.shape[1]
        if seed is not None:
            utils.set_random_seed(seed)
        if max_lag is None:
            max_lag = adj_mats.shape[0]
        elif max_lag != adj_mats.shape[0]:
            raise ValueError('The max time lag must be equal to the shape 0 of adjacent matrix')

        noise_scale = 1.0 if noise_scale is None else noise_scale
        if np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(n_nodes)
        elif len(noise_scale) == n_nodes:
            for ns in noise_scale:
                assert isinstance(ns, (int, float))
            scale_vec = noise_scale
        else:
            raise ValueError('noise scale must be a scalar or has length d')

        Xt_ = []
        X_lags = _gen_noise((max_lag, n_nodes), noise_type, scale_vec)
        for lag in range(max_lag):
            Xt_.append(X_lags[lag, :])

        for t in range(length):
            X_lags, X_t = TimeLagSimulator._simulate_single_time_nonlinear_anm(
                n_nodes, X_lags, adj_mats, noise_type, noise_scale, func_type, hidden)
            Xt_.append(X_t)
        Xt_ = np.array(Xt_)

        return Xt_

    @staticmethod
    def _simulate_single_time_pnl(
            n_nodes, value_lags, adj_mats, func1_series, func2_series, noise_type='gauss', noise_scale=1.0,
            hidden=None):
        value_t = np.zeros(n_nodes)
        noise = _gen_noise([hidden, n_nodes], noise_type, noise_scale).T

        for j in range(n_nodes):
            func1, func2 = func1_series[j], func2_series[j]
            value_t[j] = np.sum(
                func2(np.sum(func1(np.multiply(value_lags, adj_mats[:, :, j])), axis=0) + noise[j]))  # ysa
        value_lags = np.concatenate((value_lags, value_t[np.newaxis, :]), axis=0)[1:, :]
        return value_lags, value_t

    @staticmethod
    def simulate_pnl(
            adj_mats: np.ndarray, max_lag: Optional[int], length: Optional[int] = None,
            noise_type: Union[str, List[str]] = 'gauss', noise_scale: Union[float, int, np.ndarray, list, None] = None,
            seed: Optional[int] = None, func1_type: Union[List[str], str] = 'tanh',
            func2_type: Union[List[str], str] = 'leaky_relu',
            neg_slope: float = 0.2,
            hidden: Optional[int] = None):
        """
        simulate the dataset from post non-linear model for time lags with specified type of noise
        :param adj_mats: np.ndarray, the adjacent matrix of shape [n_nodes, n_nodes]
        :param max_lag: int, the max lag
        :param length: int, the length of generated data
        :param noise_type: str or list, type of noise, e.g , 'gauss', 'exp', 'gumbel', 'uniform', 'laplace'
        :param noise_scale: float, np.array or list, scale parameter of noise
        :param seed:
        :param func1_type: str or list, type of non-linear function, e.g, 'tanh', 'leaky_relu', 'cos', 'linear'
                if 'func1_type' is a list, a sequence of non-linear function will be created, e.g, ['linear', 'tanh', 'linear']
        :param func2_type: str or list, type of non-linear function, e.g, 'tanh', 'leaky_relu', 'cos', 'linear'
                if 'func2_type' is a list, a sequence of non-linear function will be created, e.g, ['linear', 'tanh', 'linear']
        :param neg_slope: float, the negative slop of leaky relu function
        :param hidden: int, the numbers of hidden linear layers in non-linear funciton
        :return:
            np.ndarray, samples of shape [length+max_lag, n_nodes]
        """
        assert noise_type in ['gauss', 'exp', 'gumbel', 'uniform', 'laplace']
        assert isinstance(noise_type, (str, list))
        func_list = ['tanh', 'leaky_relu', 'cos', 'linear']
        for func_type in [func1_type, func2_type]:
            if isinstance(func_type, str):
                assert func_type in func_list
            else:
                assert set(func_type) <= set(func_list)
        if seed is not None:
            utils.set_random_seed(seed)
        n_nodes = adj_mats.shape[1]
        if max_lag is None:
            max_lag = adj_mats.shape[0]
        elif max_lag != adj_mats.shape[0]:
            raise ValueError('The max time lag must be equal to the shape 0 of weight matrix')

        noise_scale = 1.0 if noise_scale is None else noise_scale
        if np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(n_nodes)
        elif len(noise_scale) == n_nodes:
            for ns in noise_scale:
                assert isinstance(ns, (int, float))
            scale_vec = noise_scale
        else:
            raise ValueError('noise scale must be a scalar or has length d')

        Xt_ = []
        X_lags = _gen_noise((max_lag, n_nodes), noise_type, scale_vec)
        for lag in range(max_lag):
            Xt_.append(X_lags[lag, :])

        if hidden is None:
            hidden = n_nodes
        elif hidden != n_nodes:
            if ('linear' not in func1_type or 'linear' not in func2_type) \
                    and ('linear' != func1_type or 'linear' != func2_type):
                raise ValueError("func1 and func2 must contain linear layer if you want to set the hidden size")
        func1_series, func2_series = [], []
        for j in range(n_nodes):
            func1_series.append(_pnl_func_sequential(func1_type, n_nodes, hidden, neg_slope, hidden))
            func2_series.append(_pnl_func_sequential(func2_type, hidden, 1, neg_slope, hidden))

        for t in range(length):
            X_lags, X_t = TimeLagSimulator._simulate_single_time_pnl(
                n_nodes, X_lags, adj_mats, func1_series, func2_series, noise_type, noise_scale, hidden)
            Xt_.append(X_t)
        Xt_ = np.array(Xt_)

        return Xt_


class RepresentationSimulator:  # generate the dataset for representation learning
    pass


class HawkesSimulator:  # generate the dataset of the hawkes process
    pass
