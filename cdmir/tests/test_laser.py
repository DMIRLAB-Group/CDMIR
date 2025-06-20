from cdmir.effect.LASER.laser import IVAE_tx_wrapper
import numpy as np
import torch
import numpy.random as random
from cdmir.effect.ate_estimator import ipw_estimator
from cdmir.datasets.utils import set_random_seed


def data_generator_np_wo(size=2000, obs_size=1200, dim_x=10, dim_so=2, dim_sl=2, dim_proxy=2, wo=1, seednum=0):
    """Generate experimental data based on wo.
    wo=0, no lack
    wo=1, lack so
    wo=0, lack sl

    Parameters
    ----------
    size : int
        The total number of samples in the dataset.
    obs_size : int
        The number of observed samples in the dataset.
    dim_x : int
        The dimension of the covariates.
    dim_so : int
        The dimension of the observable surrogate.
    dim_sl : int
        The dimension of the latent surrogate
    dim_proxy : int
        The dimension of the proxy of the latent surrogate.
    wo : int
        The form of the surrogate is determined by the value of wo.
    seednum : int
        Random seed.
    Returns
    -------
    Obs : array-like, shape (obs_size, dim(x)+dim(t)+dim(y)+dim(s))
        Observed dataset.
    Exp : array-like, shape (size-obs_size, dim(x)+dim(t)+dim(y)+dim(s))
        Experimental dataset.
    tau_real : float
        True causal effect.
    """
    set_random_seed(seednum, True)
    w1_x2so = np.random.normal(1, 1, size=[dim_x, int(dim_so)])
    w0_x2so = np.random.normal(1, 1, size=[dim_x, int(dim_so)])
    w1_x2sl = np.random.normal(1, 1, size=[dim_x, int(dim_sl)])
    w0_x2sl = np.random.normal(1, 1, size=[dim_x, int(dim_sl)])

    w_sl2p = np.random.normal(1, 1, size=[dim_sl, int(dim_proxy)])

    w_s2y = np.random.normal(1, 1, size=[int((dim_so + dim_sl)), 1])
    w_xy = np.random.normal(1, 1, size=[dim_x, 1])

    # data
    x0 = np.random.normal(0, 1, size=[size, int(dim_x - 3)])
    x1 = np.random.normal(1, 1, size=[size, int(3)])
    x = np.concatenate((x0, x1), axis=1)

    t_obs = []
    for i in range(obs_size):
        p = 1 / (1 + np.exp(-np.mean(x[i, :])))
        # print(p)
        t = np.random.binomial(1, p=p)
        t_obs.append(t)
    t_obs = np.array(t_obs)
    t_exp = np.random.binomial(1, 0.6, size=[int(size - obs_size), 1])
    t = np.concatenate((t_obs[:, None], t_exp), axis=0)

    e_so, e_sl, e_p, e_y = np.random.normal(0, 1, size=[size, dim_so]), \
                           np.random.normal(0, 1, size=[size, dim_sl]), \
                           np.random.normal(0, 1, size=[size, dim_proxy]), np.random.normal(0, 1, size=[size, 1])

    so1 = np.matmul(x, w1_x2so + 1) + e_so
    so0 = np.matmul(x, w0_x2so - 1) + e_so
    sl1 = np.matmul(x, w1_x2sl + 1) + e_sl
    sl0 = np.matmul(x, w0_x2sl - 1) + e_sl

    so = np.where(t == 1, so1, so0)
    sl = np.where(t == 1, sl1, sl0)
    surrogate1 = np.concatenate((so1, sl1), axis=1)
    surrogate0 = np.concatenate((so0, sl0), axis=1)

    p = np.matmul(sl, w_sl2p) + e_sl

    y1 = np.matmul(surrogate1, w_s2y) + np.matmul(x, w_xy) + e_y
    y0 = np.matmul(surrogate0, w_s2y) + np.matmul(x, w_xy) + e_y

    if wo == 0:
        s = np.concatenate((so, p), axis=1)
    if wo == 1:
        s = p
    if wo == 2:
        s = so

    # s = np.where(t==1, so1, so0)
    y = np.where(t == 1, y1, y0)
    if obs_size >= size:
        return
    xo, xe = x[:obs_size, :], x[obs_size:, :]
    so, se = s[:obs_size, :], s[obs_size:, :]
    yo, ye = y[:obs_size, :], y[obs_size:, :]
    to, te = t[:obs_size, :], t[obs_size:, :]
    Obs = xo, to, so, yo
    Exp = xe, te, se, ye
    tau_real = np.mean(y1[obs_size:, :] - y0[obs_size:, :])
    print(tau_real)

    return Obs, Exp, tau_real


def test_laser(data, seed_num=0, is_rct=1):
    set_random_seed(seed_num)
    if torch.cuda.is_available():
        cuda = True
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_dtype(torch.double)

    # numpy to tensor
    Obs, Exp, tau_real = data
    xo, to, so, yo = Obs
    xe, te, se, ye = Exp
    xo, xe, so, yo, te, se, ye, to = torch.tensor(xo), torch.tensor(xe), torch.tensor(so), torch.tensor(
        yo), torch.tensor(
        te), torch.tensor(se), torch.tensor(ye), torch.tensor(to)
    Obs = xo, to, so, yo
    Exp = xe, te, se, ye
    data = Obs, Exp, tau_real

    # use wrapper to train
    losses, model = IVAE_tx_wrapper(data=data, batch_size=100, max_epoch=1000, n_layers=3, hidden_dim=200,
                                    learn_rate=1e-4, weight_decay=1e-4, activation='lrelu', inference_dim=2,
                                    optm='Adam', base_eopch=100, anneal=False, print_log=True, is_rct=True,
                                    cuda=True, normalization=True, beta=1, theta=1, early_stop=True,
                                    treatment_dim=1, treated=1, control=0)

    ye = model.test(covariate=xe.cuda(), s=se.cuda(), treatment=te.cuda())
    y1_index = (te[:, 0] == 1).unsqueeze(1)
    y0_index = (te[:, 0] == 0).unsqueeze(1)

    # effect ate
    if is_rct:
        E_y1 = torch.mean(ye[y1_index])
        E_y0 = torch.mean(ye[y0_index])
        tau_ivae = E_y1 - E_y0
        E_y1 = E_y1.cpu().detach().numpy()
        E_y0 = E_y0.cpu().detach().numpy()
        tau_ivae = tau_ivae.cpu().detach().numpy()
    else:
        tau_ivae = ipw_estimator(x=xe.cpu().detach().numpy(), t=te.cpu().detach().numpy(), y=ye.cpu().detach().numpy())

    print('In RCT: E[y_1]={}, E[y_0]={}, ivae tau={}'.format(E_y1, E_y0, tau_ivae))

    error = np.abs(tau_real-tau_ivae)
    mape = error / tau_real
    print(f'The mean absolute percentage error is {100*mape:.2f}%')

if __name__ == '__main__':
    data = data_generator_np_wo(wo=2, seednum=0)
    test_laser(data)
# In RCT: E[y_1]=54.72615579973221, E[y_0]=6.444974099706979, ivae tau=48.28118170002523
# The mean absolute percentage error is 5.33%
