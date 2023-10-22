from locally_connected import LocallyConnected
from lbfgsb_scipy import LBFGSBScipy
import torch
import torch.nn as nn
import numpy as np
import math
import utils as utils


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        M = torch.eye(d) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class NotearsSobolev(nn.Module):
    def __init__(self, d, k):
        """d: num variables k: num expansion of each variable"""
        super(NotearsSobolev, self).__init__()
        self.d, self.k = d, k
        self.fc1_pos = nn.Linear(d * k, d, bias=False)  # ik -> j
        self.fc1_neg = nn.Linear(d * k, d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                for _ in range(self.k):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x):  # [n, d] -> [n, dk]
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / math.pi  # sobolev basis
            psi = mu * torch.sin(x / mu)
            seq.append(psi)  # [n, d] * k
        bases = torch.stack(seq, dim=2)  # [n, d, k]
        bases = bases.view(-1, self.d * self.k)  # [n, dk]
        return bases

    def forward(self, x):  # [n, d] -> [n, d]
        bases = self.sobolev_basis(x)  # [n, dk]
        x = self.fc1_pos(bases) - self.fc1_neg(bases)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        M = torch.eye(self.d) + A / self.d  # (Yu et al. 2019)
        E = torch.matrix_power(M, self.d - 1)
        h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:  # 0.25 in original paper
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 10,  # 100 in original paper
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      loss_type='mse'):  # 0.3 in original paper
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)  # not use adam in original paper
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import utils as ut
    # ut.set_random_seed(123)

    """
    n : sample size
    d : num of nodes
    s0: num of edges
    """
    # 200 5 9 er mim  in original paper
    n, d, s0, graph_type, noise_dist, sem_type = 400, 10, 20, 'ER', 'uniform', 'mim'
    lambda1, lambda2 = 0.01, 0.001
    low, high = 3., 3.
    print("n={}, d={}, s0={}, l1={}, l2={}, graph_type={}, sem_type={}, noise={}, low={}, high={}".
          format(n, d, s0, lambda1, lambda2, graph_type, sem_type, noise_dist, low, high))

    noise_scale = np.random.uniform(low=low, high=high, size=d)
    noise_scale = noise_scale * np.ones(d)
    B_true = ut.simulate_dag(d, s0, graph_type)
    X = ut.simulate_nonlinear_sem(B_true, n, sem_type=sem_type, noise_dist=noise_dist, noise_scale=noise_scale)

    # B_true = ut.simulate_dag(d, s0, graph_type)
    # X = ut.simulate_nonlinear_sem(B_true, n, sem_type=sem_type, noise_dist=noise_dist)

    # normalize
    # for i in range(d):
    #     X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=lambda1, lambda2=lambda2)
    # print(W_est)

    assert ut.is_dag(W_est)
    acc = ut.count_accuracy(B_true, W_est != 0)
    # ut.drawGraph(B_true)
    # ut.drawGraph(W_est != 0)
    print(acc)

    # 测试pairwise
    # n, d = 200, 2
    # B_true = np.array([[0, 1],
    #                    [0, 0]])  # x1 -> x2
    # X = simulate_pariwise(5, n, linear=False, distribution='uniform', direct=0)
    # # # normalize
    # # for i in range(d):
    # #     X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    #
    # model = NotearsMLP(dims=[d, 10, 1], bias=True)
    # W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
    # assert ut.is_dag(W_est)
    # acc = ut.count_accuracy(B_true, W_est != 0)
    # drawGraph(B_true)
    # drawGraph(W_est != 0)
    # print(acc)

    # print(B_true)
    # print(W_est)
    return acc


"""
n : sample size
d : num of nodes
s0: num of edges
"""


def run(n=200, d=5, s0=10, graph_type='ER', noise_dist='uniform', sem_type='mim',
        lambda1=0.01, lambda2=0.001, low=3., high=3.):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import utils as ut

    print("n={}, d={}, s0={}, l1={}, l2={}, graph_type={}, sem_type={}, noise={}, low={}, high={}".
          format(n, d, s0, lambda1, lambda2, graph_type, sem_type, noise_dist, low, high))
    noise_scale = np.random.uniform(low=low, high=high, size=d)
    noise_scale = noise_scale * np.ones(d)
    B_true = ut.simulate_dag(d, s0, graph_type)
    X = ut.simulate_nonlinear_sem(B_true, n, sem_type=sem_type, noise_dist=noise_dist, noise_scale=noise_scale)

    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=lambda1, lambda2=lambda2)

    assert ut.is_dag(W_est)
    acc = ut.count_accuracy(B_true, W_est != 0)

    return acc


def run_real():
    import utils

    B_true = utils.readcsv('data/true.csv')
    X = utils.readcsv('data/sachs_unstand.csv')
    d = 11
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=0.001, lambda2=0.001)
    if not utils.is_dag(W_est):
        print('not a dag')
        print(B_true)
        print(W_est)
        utils.drawGraph(B_true)
        utils.drawGraph(W_est != 0)

    acc = utils.count_accuracy(B_true, W_est != 0)
    print(B_true)
    print(W_est)
    return acc


def run_real_853(lambda1=1e-1, lambda2=1e-1):
    import utils
    print(lambda1, lambda2)
    path = "sachs/continuous/data1.npy"
    X = np.load(path)
    path = "sachs/continuous/DAG1.npy"
    B_true = np.load(path)

    d = 11
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=lambda1, lambda2=lambda2)
    if not utils.is_dag(W_est):
        print('not a dag')
        print(B_true)
        print(W_est)
        utils.drawGraph(B_true)
        utils.drawGraph(W_est != 0)

    acc = utils.count_accuracy(B_true, W_est != 0)
    # print(B_true)
    # print(W_est)
    return acc

