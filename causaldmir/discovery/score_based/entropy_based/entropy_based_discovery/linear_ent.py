import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.nn.parameter import Parameter
from torch.optim import LBFGS

from causaldmir.discovery.score_based.entropy_based.entropy_based_discovery import utils
from causaldmir.discovery.score_based.entropy_based.entropy_based_discovery.lbfgsb_scipy import LBFGSBScipy

debug = False
is_cuda = False
num_device = 2
pos_neg = True


class linear_nonGauss(nn.Module):

    def __init__(self, dims, bias=False):
        super(linear_nonGauss, self).__init__()
        self.l2_reg_store = None
        self.d = dims

        if pos_neg:
            self.linear_pos = nn.Linear(dims, dims, bias=bias)
            self.linear_neg = nn.Linear(dims, dims, bias=bias)
            self.linear_pos.weight.bounds = self._bounds()
            self.linear_neg.weight.bounds = self._bounds()
            nn.init.zeros_(self.linear_pos.weight)
            nn.init.zeros_(self.linear_neg.weight)
        else:
            self.linear = nn.Linear(dims, dims, bias=bias)

        if is_cuda:
            self.eye = torch.eye(dims).cuda()
        else:
            self.eye = torch.eye(dims)

    def _bounds(self):
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                if i == j:
                    bound = (0, 0)
                else:
                    bound = (0, None)
                bounds.append(bound)
        return bounds

    @torch.no_grad()
    def weight_test(self):
        if pos_neg:
            weight = self.linear_pos.weight - self.linear_neg.weight
        else:
            weight = self.linear.weight
        weight = weight.cpu().detach().numpy().T
        return weight

    def weight_train(self):
        if pos_neg:
            weight = self.linear_pos.weight - self.linear_neg.weight
        else:
            weight = self.linear.weight
        return weight

    def abs_L1(self):
        # reg = torch.sum(torch.abs(self.linear.weight))
        reg = torch.norm(self.weight_train(),p=1)
        return reg

    def L1(self):
        reg = torch.sum(self.weight_train())
        # reg = torch.sum(torch.abs(self.linear.weight))
        return reg

    def L2(self):
        reg = self.l2_reg_store
        return reg

    def h_func(self):
        W = self.weight_train().t()
        A = W * W
        M = self.eye + A / self.d
        # M = torch.eye(self.d) + A / self.d
        E = torch.matrix_power(M, self.d)
        h_A = torch.trace(E) - self.d
        return h_A

    def _h(self):
        """Evaluate value and gradient of acyclicity constraint."""
        W = self.weight_train().t()
        # E = slin.expm(W * W)  # (Zheng et al. 2018)
        # h = np.trace(E) - d
        # G_h = (E @ W).T
        M = torch.eye(self.d).cuda() + W * W / self.d  # (Yu et al. 2019)
        E = torch.matrix_power(M, self.d - 1)
        h = (E.t() * M).sum() - self.d
        return h

    def forward(self, x):
        if pos_neg:
            x_hat = self.linear_pos(x) - self.linear_neg(x)
        else:
            x_hat = self.linear(x)
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x_hat


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


# torch_LBFGS, train on GPU
def LBFGS_torch(model, X_torch, lambda1, lambda2, rho, alpha, h, rho_max, beta, gama, BATCH_SIZE=64):
    """Perform one step of  ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGS(model.parameters(), lr=1)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            mse_loss = squared_loss(X_hat, X_torch)
            entloss = utils.entropy_loss_mentappr(X_hat, X_torch)

            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * model.L2()
            l1_reg = model.abs_L1()
            primal_obj = beta * entloss + gama * mse_loss + penalty + lambda2 * l2_reg + lambda1 * l1_reg
            if debug:
                print('loss={}'.format(primal_obj))
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

# scipy_LBFGSB, train on cpu
def LBFGSB(model, X_torch, lambda1, lambda2, rho, alpha, h, rho_max, beta, gama):
    """Perform one step of  ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            mse_loss = squared_loss(X_hat, X_torch)
            entloss = utils.entropy_loss_mentappr(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.L2()
            l1_reg = lambda1 * model.abs_L1()
            primal_obj = beta * entloss + gama * mse_loss + penalty + l2_reg + l1_reg
            primal_obj.backward()

            # print(primal_obj)
            if debug:
                print('entloss={}, mseloss={}'.format(entloss, mse_loss))
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place

        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:  # 0.25 in original paper
            rho *= 10
        else:
            break
    if debug:
        print(alpha, rho, h_new)
    alpha += rho * h_new
    return rho, alpha, h_new


def train_LBFGSB(model: nn.Module,
                 X: np.ndarray,
                 lambda1: float = 0.,
                 lambda2: float = 0.,
                 max_iter: int = 100,  # 100 in original paper
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_threshold: float = 0.3,  # 0.3 in original paper
                 beta: float = 1.,
                 gama: float = 1.,  # balance weight between entloss and mseloss, obj = beta*ent + gama * mse
                 anneal=False):
    rho, alpha, h = 1.0, 0.0, np.inf
    if is_cuda:
        X_torch = torch.from_numpy(X).cuda()
    else:
        X_torch = torch.from_numpy(X)

    beta_anneal = 0.1
    for _ in range(max_iter):
        # anneal beta from initial value to zero
        if anneal:
            if beta > beta_anneal:
                beta -= beta_anneal
                if debug:
                    print(beta)

        if is_cuda:
            rho, alpha, h = LBFGS_torch(model, X_torch, lambda1, lambda2,
                                        rho, alpha, h, rho_max, beta=beta, gama=gama)  # not use adam in original paper
        else:
            rho, alpha, h = LBFGSB(model, X_torch, lambda1, lambda2,
                                   rho, alpha, h, rho_max, beta=beta, gama=gama)  # not use adam in original paper
        if h <= h_tol or rho >= rho_max:
            break
    if debug:
        print('rho={}, alpha={}, h={}'.format(rho, alpha, h))
    # print(model.linear_pos.weight)
    # print(model.linear_neg.weight)

    W_est = model.weight_test()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def train_dam(model, X, optim_type='adam',
              max_iter=100,
              rho=1.0, alpha=0.0, h=np.inf,
              lambda1=0.01, lambda2=0.00,
              rho_max=1e+20, h_tol=1e-8,
              w_threshold=0.3, beta=1., gama=1.,
              epoch=8,
              BATCH_SIZE=64):
    global optimizer
    X_torch_all = torch.from_numpy(X).cuda()

    deal_Data = Data.TensorDataset(X_torch_all, torch.zeros_like(X_torch_all))
    train_data = Data.DataLoader(dataset=deal_Data, batch_size=BATCH_SIZE, shuffle=True)

    if optim_type == 'adam':
        optimizer = optim.Adam(list(model.parameters()), lr=1e-2)
    elif optim_type == 'adadelta':
        optimizer = optim.Adadelta(list(model.parameters()), lr=1e-2)
    elif optim_type == 'adamax':
        optimizer = optim.Adamax(list(model.parameters()), lr=1e-2)
    elif optim_type == 'asgd':
        optimizer = optim.ASGD(list(model.parameters()), lr=1e-2)
    h_old = np.inf
    for _ in range(max_iter):
        h_new = None
        while rho <= rho_max:
            for i in range(epoch):

                for step, (X_torch, y) in enumerate(train_data):
                    optimizer.zero_grad()
                    X_hat = model(X_torch)
                    residual_loss = utils.entropy_loss_mentappr(X_hat, X_torch)

                    mse_loss = squared_loss(X_hat, X_torch)
                    L1_loss = lambda1 * model.L1()
                    L2_loss = 2 * lambda2 * model.L2()
                    h_val = model.h_func()
                    # h_val = model._h()
                    acy_loss = 0.5 * rho * h_val * h_val + alpha * h_val
                    loss = beta * residual_loss + gama * mse_loss + L1_loss + L2_loss + acy_loss
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                # h_new = model._h().item()
                h_new = model.h_func().item()
            if h_new > 0.25 * h_old:  # 0.25 in original paper
                rho *= 10
            else:
                break
        h_old = h_new
        alpha += rho * h_new
        if h_old <= h_tol or rho >= rho_max:
            if debug:
                print(h_old)
            break
    W_est = model.weight()
    W_est[np.abs(W_est) < w_threshold] = 0
    if debug:
        print('h_new={},rho={},alpha={}'.format(h_old, rho, alpha))

    return W_est



# original in paper
def original():
    n, d, s0, graph_type, sem_type = 500, 10, 20, 'ER', 'uniform'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)

    X = utils.simulate_linear_sem(W_true, n, sem_type)

    for i in range(d):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

    model = linear_nonGauss(dims=d, bias=False)
    W_est = train_LBFGSB(model, X, lambda1=0.001, lambda2=0.00, beta=1., gama=0., max_iter=2000, anneal=False)
    assert utils.is_dag(W_est)
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
    return acc


def run_real_853(lamda1=1e-3, lamda2=0):
    print('run 853')

    lambda1 = lamda1    # 0.00001
    lambda2 = lamda2     # 0.00001
    print(lambda1, lambda2)
    path = "sachs/continuous/data1.npy"
    X = np.load(path)
    path = "sachs/continuous/DAG1.npy"
    B_true = np.load(path)

    d = 11
    model = linear_nonGauss(dims=d, bias=False)
    W_est = train_LBFGSB(model, X, lambda1=lamda1, lambda2=lamda2, beta=1., gama=0., max_iter=2000, anneal=False)
    if not utils.is_dag(W_est):
        print('not a dag')
        print(B_true)
        print(W_est)
        utils.drawGraph(B_true)
        utils.drawGraph(W_est != 0)

    acc = utils.count_accuracy(B_true, W_est != 0)
    # print(B_true)
    # print(W_est)
    print(acc)
    return acc

def main():
    # n = sample size
    # d = num of nodes
    # s0 = num of edges
    # balance weight between entloss and mseloss, obj = beta*ent + gama * mse
    # uniform :5,8-0.  200,10,20-1.0   1000,10,20-1.2(ent 0.1)  200.15.20-1.5
    # gumbel  :5,8-1.0(only ent and anneal 0.0)  200,10,20-8.0   1000,10,20-1.2(ent 0.1)  200.15.20-1.5
    n, d, s0, graph_type, sem_type = 600, 5, 8, 'ER', 'uniform'
    beta, gama, lambda1, anneal = 1., 0., 0.005, False
    print("n={}, d={}, s0={}, graph_type={}, sem_type={}, beta={}, gama={}, lambda1={}, anneal={}, kkt={}".
          format(n, d, s0, graph_type, sem_type, beta, gama, lambda1, anneal, pos_neg))

    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true, ((-0.8, -0.4), (0.4, 0.8)))
    # W_true = utils.simulate_parameter(B_true)

    # X = utils.simulate_linear_sem(W_true, n, sem_type)
    X = utils.simulate_linear_sem_variable_scale(W_true, n, sem_type, scale_low=0.5, scale_high=1.)
    # normalize
    # for i in range(d):、
    #     X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

    if is_cuda:  # still bug
        model = linear_nonGauss(dims=d, bias=False).cuda()
        W_est = train_LBFGSB(model, X, lambda1=lambda1, lambda2=0.00, beta=beta, gama=gama, max_iter=2000)
        # W_est = train_dam(model, X, lambda1=lambda1, lambda2=0.00, beta=beta, gama=gama, max_iter=100)
    else:
        model = linear_nonGauss(dims=d, bias=False)
        W_est = train_LBFGSB(model, X, lambda1=lambda1, lambda2=0.00, beta=beta, gama=gama, max_iter=4000, anneal=anneal)

    if not utils.is_dag(W_est):
        print('not a dag')
        # print(W_true)
        print(W_est)

    # utils.drawGraph(B_true)
    # utils.drawGraph(W_est != 0)
    # print(W_true)
    # print(W_est)

    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

    return acc

def run(n=400, d=5, s0=8, graph_type='ER', sem_type='gauss', lamda1 = 0.001,
        w_ranges=((-2.0, -0.5), (0.5, 2.0)), scale_low=1., scale_high=1.):
    torch.set_default_dtype(torch.double)


    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true, w_ranges)

    X = utils.simulate_linear_sem_variable_scale(W_true, n, sem_type, scale_low=scale_low, scale_high=scale_high)

    model = linear_nonGauss(dims=d, bias=False)
    W_est = train_LBFGSB(model, X, lambda1=lamda1, lambda2=0.00, beta=1., gama=0., max_iter=2000, anneal=False)

    if not utils.is_dag(W_est):
        print('not a dag')
        print(W_true)
        print(W_est)
        utils.drawGraph(B_true)
        utils.drawGraph(W_est != 0)

    acc = utils.count_accuracy(B_true, W_est != 0)
    # print(acc)
    return acc


if __name__ == '__main__':
    torch.set_default_dtype(torch.double)
    main()