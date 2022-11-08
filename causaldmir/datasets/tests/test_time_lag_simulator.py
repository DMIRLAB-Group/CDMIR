from unittest import TestCase
import numpy as np

from ..simlulators import TimeLagSimulator
from ..utils import erdos_renyi, generate_lag_transitions


class TestTimeLagSimulator(TestCase):

    def test_simulate_linear_anm_gauss(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        weight_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'gauss'
        X = TimeLagSimulator.simulate_linear_anm(weight_mats, max_lag, length, noise_type, noise_scale, seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length+max_lag, n_nodes)

    def test_simulate_linear_anm_exp(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        weight_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'exp'
        X = TimeLagSimulator.simulate_linear_anm(weight_mats, max_lag, length, noise_type, noise_scale, seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length+max_lag, n_nodes)

    def test_simulate_linear_anm_gumbel(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        weight_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'gumbel'
        X = TimeLagSimulator.simulate_linear_anm(weight_mats, max_lag, length, noise_type, noise_scale, seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length+max_lag, n_nodes)

    def test_simulate_linear_anm_uniform(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        weight_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'uniform'
        X = TimeLagSimulator.simulate_linear_anm(weight_mats, max_lag, length, noise_type, noise_scale, seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length+max_lag, n_nodes)

    def test_simulate_linear_anm_laplace(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        weight_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'laplace'
        X = TimeLagSimulator.simulate_linear_anm(weight_mats, max_lag, length, noise_type, noise_scale, seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length+max_lag, n_nodes)

    def test_simulate_linear_anm_logistic(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        weight_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'logistic'
        X = TimeLagSimulator.simulate_linear_anm(weight_mats, max_lag, length, noise_type, noise_scale, seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length+max_lag, n_nodes)

    def test_simulate_nonlinear_anm_gauss(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'gauss'
        X = TimeLagSimulator.simulate_nonlinear_anm(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_nonlinear_anm_exp(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'exp'
        X = TimeLagSimulator.simulate_nonlinear_anm(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_nonlinear_anm_gumbel(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'gumbel'
        X = TimeLagSimulator.simulate_nonlinear_anm(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_nonlinear_anm_uniform(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'uniform'
        X = TimeLagSimulator.simulate_nonlinear_anm(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_nonlinear_anm_laplace(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'laplace'
        X = TimeLagSimulator.simulate_nonlinear_anm(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_nonlinear_anm_mim(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'gauss'
        func_type = 'mim'
        X = TimeLagSimulator.simulate_nonlinear_anm(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed,
                                                    func_type=func_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_nonlinear_anm_hidden(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'gauss'
        hidden = 100
        X = TimeLagSimulator.simulate_nonlinear_anm(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed,
                                                    hidden=hidden)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_gauss(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'gauss'
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_exp(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'exp'
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_gumbel(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'gumbel'
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_uniform(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'uniform'
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_laplace(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'laplace'
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_funcseq1(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'laplace'
        func1_type = 'leaky_relu'
        func2_type = 'tanh'
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed,
                                          func1_type=func1_type, func2_type=func2_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_funcseq2(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'laplace'
        func1_type = ['linear', 'leaky_relu']
        func2_type = 'tanh'
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed,
                                          func1_type=func1_type, func2_type=func2_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_funcseq3(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'laplace'
        func1_type = ['linear', 'leaky_relu', 'linear', 'tanh']
        func2_type = ['linear', 'tanh']
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed,
                                          func1_type=func1_type, func2_type=func2_type)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)

    def test_simulate_pnl_funcseq4(self):
        n_nodes = 10
        seed = 10
        max_lag = 3
        noise_scale = [0.9] * n_nodes
        adj_mats = generate_lag_transitions(n_nodes, max_lag, seed)
        length = 5000
        noise_type = 'laplace'
        func1_type = ['linear', 'leaky_relu', 'linear', 'tanh']
        func2_type = ['linear', 'tanh']
        hidden = 100
        X = TimeLagSimulator.simulate_pnl(adj_mats, max_lag, length, noise_type, noise_scale, seed=seed,
                                          func1_type=func1_type, func2_type=func2_type, hidden=hidden)
        assert isinstance(X, np.ndarray)
        assert X.shape == (length + max_lag, n_nodes)
