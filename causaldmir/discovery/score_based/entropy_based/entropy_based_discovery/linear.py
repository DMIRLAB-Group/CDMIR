import numpy as np
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import scipy.sparse.linalg as slin
import utils as ut



def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):

    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.


    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """

    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        # E = slin.expm(W * W)  # (Zheng et al. 2018)
        # h = np.trace(E) - d
        # G_h = (E @ W).T
        M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        E = np.linalg.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)

        # log
        # print('mse={},h={}'.format(loss,h)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg) 
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    # print('rho={},alpha={}'.format(rho,alpha))
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

# original in paper
def original():
    n, d, s0, graph_type, sem_type = 500, 10, 20, 'ER', 'uniform'
    B_true = ut.simulate_dag(d, s0, graph_type)
    W_true = ut.simulate_parameter(B_true)

    X = ut.simulate_linear_sem(W_true, n, sem_type)

    for i in range(d):
        X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert ut.is_dag(W_est)
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)
    return acc


def main():
    from notears import utils

    n, d, s0, graph_type, sem_type = 600, 20, 40, 'ER', 'uniform'
    lamda1 = 0.001  #  gumbel:变量多的时候1.效果好  10,20-1.7  15,30-3.0
    print("n={}, d={}, s0={}, graph_type={}, sem_type={}，lamda={}".
          format(n, d, s0, graph_type, sem_type, lamda1))

    a, b = 0.4, 0.8
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true, w_ranges=((-0.8, -0.4), (0.4, 0.8)))
    # W_true = utils.simulate_parameter(B_true)
    # W_true = B_true * 0.5

    X = utils.simulate_linear_sem_variable_scale(W_true, n, sem_type, scale_low=0.5, scale_high=1.)

    W_est = notears_linear(X, lambda1=lamda1, loss_type='l2', w_threshold=0.3, max_iter=2000)
    if not utils.is_dag(W_est):
        print('not a dag')
        print(W_true)
        print(W_est)
        utils.drawGraph(B_true)
        utils.drawGraph(W_est != 0)

    # print(W_true)
    # print(W_est)
    # utils.drawGraph(B_true)
    # utils.drawGraph(W_est != 0)

    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

    return acc

def run(n=400, d=5, s0=8, graph_type='ER', sem_type='gauss', lamda1 = 0.001,
        w_ranges=((-2.0, -0.5), (0.5, 2.0)), scale_low=1., scale_high=1.):
    import utils

    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true, w_ranges)

    X = utils.simulate_linear_sem_variable_scale(W_true, n, sem_type, scale_low=scale_low, scale_high=scale_high)

    W_est = notears_linear(X, lambda1=lamda1, loss_type='l2', w_threshold=0.3, max_iter=2000)
    if not utils.is_dag(W_est):
        print('not a dag')
        print(W_true)
        print(W_est)
        utils.drawGraph(B_true)
        utils.drawGraph(W_est != 0)

    acc = utils.count_accuracy(B_true, W_est != 0)
    # print(acc)
    return acc

def run_pairwise(n=400, graph_type='ER', sem_type='gauss', lamda1 = 0.001,
        weight=1. , var_cause=1., var_effect=1.):
    import utils
    d, s0 = 2, 1
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = B_true * weight
    X = utils.simulate_pariwise(weight=weight, distribution=sem_type,
                                direct= 0 if B_true[0,1]==1 else 1, n=n,
                                var_cause=var_cause, var_effect=var_effect)
    # X = utils.simulate_linear_sem_variable_scale(W_true, n, sem_type, scale_low=scale_low, scale_high=scale_high)

    W_est = notears_linear(X, lambda1=lamda1, loss_type='l2', w_threshold=0.3, max_iter=2000)
    if not utils.is_dag(W_est):
        print('not a dag')
        print(W_true)
        print(W_est)
        utils.drawGraph(B_true)
        utils.drawGraph(W_est != 0)

    acc = utils.count_accuracy(B_true, W_est != 0)

    return acc

def run_real(lambda1=1e-3):
    import utils

    # B_true = utils.readcsv('data/true.csv')
    # X = utils.readcsv('data/sachs_unstand.csv')
    path = "sachs/continuous/data1.npy"
    X = np.load(path)
    path = "sachs/continuous/DAG1.npy"
    B_true = np.load(path)

    W_est = notears_linear(X, lambda1=lambda1, loss_type='l2', w_threshold=0.3, max_iter=2000)
    if not utils.is_dag(W_est):
        print('not a dag')
        print(B_true)
        print(W_est)
        utils.drawGraph(B_true)
        utils.drawGraph(W_est != 0)

    print(B_true)
    print(W_est)
    acc = utils.count_accuracy(B_true, W_est != 0)

    return acc

def run_data(data, B_true, lamda1):
    import utils
    W_est = notears_linear(data, lambda1=lamda1, loss_type='l2', w_threshold=0.3, max_iter=2000)
    acc = utils.count_accuracy(B_true, W_est != 0)
    return acc

if __name__ == '__main__':
    np.set_printoptions(precision=1)
    np.set_printoptions(precision=1)

    ut.set_seed()
    acc = run_real()
    print(acc)

    #
    # repeat_exp = 30
    # fdr, tpr, fpr, shd, nnz = [], [], [], [], []
    # for _ in range(repeat_exp):
    #     acc = original()
    #     fdr.append(acc.get('fdr'))
    #     tpr.append(acc.get('tpr'))
    #     fpr.append(acc.get('fpr'))
    #     shd.append(acc.get('shd'))
    #     nnz.append(acc.get('nnz'))
    #
    # print('fdr:{}'.format(np.array(fdr).mean()))
    # print('tpr:{}'.format(np.array(tpr).mean()))
    # print('fpr:{}'.format(np.array(fpr).mean()))
    # print('shd:{}'.format(np.array(shd).mean()))
    # print('nnz:{}'.format(np.array(nnz).mean()))
    #
