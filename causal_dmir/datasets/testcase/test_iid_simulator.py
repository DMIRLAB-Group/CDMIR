import numpy as np

from causal_dmir.datasets.simlulators import IIDSimulator
from ..utils import erdos_renyi


class TestIIDSimulator:

    pass

    def test_simulate_linear_anm_gauss(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'gauss'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_linear_anm_gauss2(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = None
        noise_type = 'gauss'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape[1] == n_nodes

    def test_simulate_linear_anm_exp(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'exp'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_linear_anm_gumbel(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'gumbel'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_linear_anm_uniform(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'uniform'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_linear_anm_logistic(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'logistic'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_linear_anm_laplace(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'laplace'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_linear_anm_noise_scale(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        noise_scale = 0.9
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'gauss'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_linear_anm_noise_scale2(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        noise_scale = [0.9] * n_nodes
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'gauss'
        X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_linear_anm_noise_scale3(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        noise_scale = ['0.9'] * n_nodes
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'gauss'
        try:
            X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, noise_scale, seed=seed)
        except Exception as e:
            assert isinstance(e, AssertionError)

    def test_simulate_linear_anm_noise_scale4(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        weight_range = (0.5, 2.0)
        noise_scale = [0.9] * (n_nodes - 1)
        weight_mat = erdos_renyi(n_nodes, n_edges, weight_range, seed)
        n_samples = 5000
        noise_type = 'gauss'
        try:
            X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, noise_scale, seed=seed)
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_simulate_linear_anm_weight_mat(self):
        seed = 10
        weight_mat = [1, 2, 3]
        n_samples = 5000
        noise_type = 'gauss'
        try:
            X = IIDSimulator.simulate_linear_anm(weight_mat, n_samples, noise_type, seed=seed)
        except Exception as e:
            assert isinstance(e, AssertionError)

    def test_simulate_nonlinear_anm_gauss(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_type = 'gauss'
        func_type = 'mlp'
        X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, seed=seed, func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_nonlinear_anm_exp(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_type = 'exp'
        func_type = 'mlp'
        X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, seed=seed, func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_nonlinear_anm_gumbel(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_type = 'gumbel'
        func_type = 'mlp'
        X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, seed=seed, func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_nonlinear_anm_uniform(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_type = 'uniform'
        func_type = 'mlp'
        X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, seed=seed, func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_nonlinear_anm_laplace(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_type = 'laplace'
        func_type = 'mlp'
        X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, seed=seed, func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_nonlinear_anm_mim(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_type = 'gauss'
        func_type = 'mim'
        X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, seed=seed, func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_nonlinear_anm_noise_scale(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = 0.9
        noise_type = 'gauss'
        func_type = 'mlp'
        X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_nonlinear_anm_noise_scale2(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = [0.9] * n_nodes
        noise_type = 'gauss'
        func_type = 'mlp'
        X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_nonlinear_anm_noise_scale3(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = ['0.9'] * n_nodes
        noise_type = 'gauss'
        func_type = 'mlp'
        try:
            X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func_type=func_type)
        except Exception as e:
            assert isinstance(e, AssertionError)

    def test_simulate_nonlinear_anm_noise_scale4(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = [0.9] * (n_nodes - 1)
        noise_type = 'gauss'
        func_type = 'mlp'
        try:
            X = IIDSimulator.simulate_nonlinear_anm(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func_type=func_type)
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_simulate_pnl_gauss_gauss(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = [0.9] * n_nodes
        noise_type = 'gauss'
        func1_type = 'leaky_relu'
        neg_slope = 0.3
        func2_type = 'tanh'
        X = IIDSimulator.simulate_pnl(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func1_type=func1_type,
                                      func2_type=func2_type, neg_slope=neg_slope)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_pnl_gauss_uniform(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = [0.9] * n_nodes
        noise_type = 'uniform'
        func1_type = 'leaky_relu'
        neg_slope = 0.3
        func2_type = 'tanh'
        X = IIDSimulator.simulate_pnl(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func1_type=func1_type,
                                      func2_type=func2_type, neg_slope=neg_slope)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_pnl_gauss_funcseq1(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = [0.9] * n_nodes
        noise_type = 'uniform'
        func1_type = 'leaky_relu'
        neg_slope = 0.3
        func2_type = 'tanh'
        X = IIDSimulator.simulate_pnl(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func1_type=func1_type,
                                      func2_type=func2_type, neg_slope=neg_slope)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_pnl_gauss_funcseq2(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = [0.9] * n_nodes
        noise_type = 'uniform'
        func1_type = ['linear', 'leaky_relu']
        neg_slope = 0.3
        func2_type = 'tanh'
        X = IIDSimulator.simulate_pnl(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func1_type=func1_type,
                                      func2_type=func2_type, neg_slope=neg_slope)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_pnl_gauss_funcseq3(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = [0.9] * n_nodes
        noise_type = 'uniform'
        func1_type = ['linear', 'leaky_relu']
        neg_slope = 0.3
        func2_type = ['linear', 'tanh']
        X = IIDSimulator.simulate_pnl(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func1_type=func1_type,
                                      func2_type=func2_type, neg_slope=neg_slope)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)

    def test_simulate_pnl_gauss_funcseq4(self):
        n_nodes = 10
        n_edges = 20
        seed = 10
        adj_mat = erdos_renyi(n_nodes, n_edges, seed=seed)
        n_samples = 5000
        noise_scale = [0.9] * n_nodes
        noise_type = 'uniform'
        func1_type = ['linear', 'leaky_relu']
        neg_slope = 0.3
        func2_type = ['linear', 'tanh']
        hidden = 100
        X = IIDSimulator.simulate_pnl(adj_mat, n_samples, noise_type, noise_scale, seed=seed, func1_type=func1_type,
                                      func2_type=func2_type, neg_slope=neg_slope, hidden=hidden)
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_samples, n_nodes)
