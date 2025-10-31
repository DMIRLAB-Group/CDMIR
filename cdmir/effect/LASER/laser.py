from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch import optim
import random
from cdmir.effect.ate_estimator import ipw_estimator
from torch.utils.data import DataLoader, TensorDataset, random_split

grad_clip = 0.5

def weights_init(m):
    """Apply Xavier initialization to each submodule of the model.

    Parameters
    ----------
    m : module
        Submodule of the model.

    Returns
    -------

    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    """Normal distribution.

    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        """Sample from a normal distribution.

        Parameters
        ----------
        mu : Number
            The mean of the normal distribution.
        v : Number
            The variance of the normal distribution.

        Returns
        -------
        sample : Number
            Samples obtained from a normal distribution using the reparameterization trick.
        """
        # eps = self._dist.sample(mu.size()).squeeze()
        eps = self._dist.sample(mu.size()).squeeze(dim=-1)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """Compute the log-pdf of a normal distribution with diagonal covariance.

        Parameters
        ----------
        x : array-like
            Input data.
        mu : Number
            The mean of the normal distribution.
        v : Number
            The variance of the normal distribution.
        reduce : bool
            Whether to return a scalar value.
        param_shape : array-like
            The shape of parameters.
        Returns
        -------
        log_pdf : array-like
            The log probability density function of the data.
        """
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        """A multilayer perceptron (MLP) is used to learn the parameters.

        Parameters
        ----------
        input_dim : int
            The dimension of input data.
        output_dim : int
            The expected output dimension.
        hidden_dim : Number or list
            The dimension of the hidden layers.
        n_layers : int
            The number of hidden layers.
        activation : str or list
            The activation function of the hidden layers.
        slope : float
            The slope of the LeakyReLU activation function.
        device : str
            The device used for training.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        """Fit the data using an MLP.

        Parameters
        ----------
        x : array-like
            Input data.
        Returns
        -------
        h : array-like
            Output data.
        """
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class iVAE_tx(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None, y_recon=None,
                 t_recon=None, t_temperature=1e-2,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False,
                 treatment_dim=1):
        """Initialize the parameters of the iVAE model.

        Parameters
        ----------
        latent_dim : int
            The dimension of latent distribution.
        data_dim : int
            The dimension of surrogates.
        aux_dim : int
            The dimension of data.
        prior : class
            The prior distribution.
        decoder : class
            The distribution of decoder.
        encoder : class
            The distribution of encoder.
        y_recon : class
            The distribution used to reconstruct the long-term outcome y.
        t_recon : class
            The distribution used to reconstruct the treatment t.
        t_temperature : float
            The parameter used to control the smoothness of the Bernoulli sampling process
        n_layers : int
            The number of hidden layers.
        hidden_dim : Number or list
            The dimension of the hidden layers.
        activation : str or list
            The activation function of the hidden layers.
        slope : float
            The slope of the LeakyReLU activation function.
        device : str
            The device used for training.
        anneal : bool
            Whether to perform annealing.
        treatment_dim : int
            The dimension of treatment vector.
        """
        super().__init__()

        self.data_dim = data_dim  # s.shape[1]
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim  # x.shape[1]
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal
        self.t_temperature = t_temperature
        self.treatment_dim = treatment_dim

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        if y_recon is None:
            self.y_recon_dist = Normal(device=device)
        else:
            self.y_recon_dist = y_recon

        # if t_recon is None:
        #     self.t_recon_dist = RelaxedBernoulli(device=device)
        #     # self.t_recon_dist = Bernoulli(device=device)
        # else:
        #     self.t_recon_dist = t_recon
        #
        # self.x_recon_dist = Normal(device=device)

        # prior_params
        # p(s|x,t)
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim + treatment_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        # p(m|s)
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        # encoder params
        # q(s|m,x,t)
        self.g = MLP(data_dim + aux_dim +treatment_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim +treatment_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)
        # reconstruct long-term y
        # p(y|s,x)
        self.meany = MLP(aux_dim + latent_dim, 1, hidden_dim, n_layers, activation=activation, slope=slope,
                         device=device)
        self.logvy = .01 * torch.ones(1).to(device)
        # reconstruct treatment t, in fact pro_t have to be input sigmoid to get true probability
        self.pro_t = MLP(aux_dim + latent_dim, 1, hidden_dim, n_layers, activation=activation, slope=slope,
                         device=device)
        self.sigmoid = nn.Sigmoid()

        # balanced x
        self.ph_x = MLP(aux_dim, hidden_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                         device=device)

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

        # normalizing y param
        self.pre_process_y = False
        self.y_mu = None
        self.y_std = None

    def encoder_params(self, s, covariate, treatment):
        """Approximate the parameters of the encoder.

        Parameters
        ----------
        s : array-like
            The data of surrogates.
        covariate : array-like
            The data of covariates.
        treatment : array-like
            The data of the treatment vector.
        Returns
        -------
        mean : array-like
            The mean of the distribution corresponding to the encoder.
        variance : array-like
            The variance of the distribution corresponding to the encoder
        """
        s_covariate_treatment = torch.cat((s, covariate, treatment), 1)
        g = self.g(s_covariate_treatment)
        logv = self.logv(s_covariate_treatment)
        return g, logv.exp()

    def decoder_params(self, s):
        """Approximate the parameters of the decoder.

        Parameters
        ----------
        s : array-like
            Samples drawn from the latent distribution.
        Returns
        -------
        mean : array-like
            The mean of the distribution corresponding to the decoder.
        variance : Number
            Fixed variance of distribution corresponding to the decoder.
        """
        f = self.f(s)
        return f, self.decoder_var

    def prior_params(self, covariate, treatment):
        """Approximate the parameters of the prior distribution.

        Parameters
        ----------
        covariate : array-like
            The data of covariates.
        treatment : array-like
            Treatment vector

        Returns
        -------
        mean : array-like
            Fixed mean of prior distribution.
        variance : array-like
            The variance of prior distribution.
        """
        covariate_treatment = torch.cat((covariate, treatment), 1)
        logl = self.logl(covariate_treatment)
        return self.prior_mean, logl.exp()

    # def sl2x(self, s_latent):
    #     meanx = self.meanx(s_latent)
    #     return meanx, self.logvy

    def sl2y(self, s_latent, covariate):
        """Approximate the parameters of the distribution used to reconstruct the long-term outcome y.

        Parameters
        ----------
        s_latent : array-like
            Data sampled from the latent distribution.
        covariate : array-like
            The data of covariates.

        Returns
        -------
        mean : array-like
            The mean of distribution.
        variance : array-like
            Fixed variance of distribution.
        """
        meany = self.meany(torch.cat((s_latent, covariate), dim=1))
        return meany, self.logvy

    # def sl2t(self, s_latent, covariate):
    #     t = self.sigmoid(self.pro_t(torch.cat((s_latent, covariate), dim=1)))
    #     return t

    def forward(self, s, covariate, treatment):
        """

        Parameters
        ----------
        s : array-like
            The data of surrogates.
        covariate : array-like
            The data of covariates.
        treatment : array-like
            The data of the treatment vector.

        Returns
        -------
        decoder_params : array-like
            The parameters of the decoder.
        encoder_params : array-like
            The parameters of the encoder.
        s_latent : array-like
            Samples drawn from the latent distribution.
        prior_params : array-like
            The parameters of the prior distribution.
        y_params : array-like
            The parameters of the reconstruct distribution.
        y_hat : array-like
            Samples drawn from the reconstruction distribution.
        """
        treatment = 1. * treatment
        # encoder
        prior_params = self.prior_params(covariate, treatment)
        encoder_params = self.encoder_params(s, covariate, treatment)
        # s_latent sample from latent variable
        s_latent = self.encoder_dist.sample(*encoder_params)
        # decoder
        decoder_params = self.decoder_params(s_latent)

        # recon_x
        # x_params = self.sl2x(s_latent)
        # x_hat = self.x_recon_dist.sample(*x_params)

        # auxiliary distribution
        y_params = self.sl2y(s_latent, covariate=covariate)
        y_hat = self.y_recon_dist.sample(*y_params)
        # t_params = self.sl2t(s_latent, covariate=covariate)
        # t_hat = self.t_recon_dist.sample(t_params, temperature=self.t_temperature)
        return decoder_params, encoder_params, s_latent, prior_params, y_params, y_hat

    def test(self, s, covariate, treatment):
        """The input data enters the encoder to generate the parameters of the latent distribution,
        which are then used to reconstruct the long-term outcome y.

        Parameters
        ----------
        s : array-like
            The data of surrogates.
        covariate : array-like
            The data of covariates.
        treatment : array-like
            The data of the treatment vector.

        Returns
        -------
        meany : array-like
            Long-term outcome y.
        """
        treatment = 1. * treatment
        encoder_params = self.encoder_params(s, covariate, treatment)
        sl_mean, _ = encoder_params
        meany, variance = self.sl2y(sl_mean, covariate=covariate)
        if self.pre_process_y:
            meany = meany * self.y_std + self.y_mu
        # print(meany)
        return meany

    def elbo(self, s, decoder_params, g, v, s_latent, prior_params, theta=1):
        """Calculate the Evidence Lower Bound (ELBO).

        Parameters
        ----------
        s : array-like
            The data of surrogates.
        decoder_params : array-like
            The parameters of the decoder.
        g : array-like
            The mean of the distribution corresponding to the encoder.
        v : array-like
            The variance of the distribution corresponding to the encoder.
        s_latent : array-like
            Samples drawn from the latent distribution.
        prior_params : array-like
            The parameters of the prior distribution.
        theta : float
            The coefficient of the ELBO reconstruction term.

        Returns
        -------
        ELBO : array-like
            Evidence Lower Bound (ELBO).
        """
        # p(m|s)
        log_ps_sl = self.decoder_dist.log_pdf(s, *decoder_params)
        # q(s|x, m, t)
        log_qsl_scovariate = self.encoder_dist.log_pdf(s_latent, g, v)
        # p(s|x, t)
        log_psl_covariate = self.prior_dist.log_pdf(s_latent, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = s_latent.size(0)
            log_qsl_tmp = self.encoder_dist.log_pdf(s_latent.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                    v.view(1, M, self.latent_dim), reduce=False)
            log_qsl = torch.logsumexp(log_qsl_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qsl_i = (torch.logsumexp(log_qsl_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_ps_sl - b * (log_qsl_scovariate - log_qsl) - c * (log_qsl - log_qsl_i) - d * (
                    log_qsl_i - log_psl_covariate)).mean(), s_latent

        else:
            return (log_ps_sl * theta + log_psl_covariate - log_qsl_scovariate).mean(), s_latent

    def log_qy_xsl(self, y, y_params, o_index):
        """Calculate the negative marginal log-likelihood of y.

        Parameters
        ----------
        y : array-like
            Long-term outcome y.
        y_params : array-like
            The parameters of the reconstruct distribution.
        o_index : array-like
            The indicator is used to select the observed data.
        Returns
        -------
        Ly : array-like
            E[log(p(y|s, x))] in observed data.
        """
        # log(p(y|s, x))
        mu, var = y_params
        mu = mu[o_index].unsqueeze(dim=1)
        y_params = mu, var
        y = y[o_index].unsqueeze(dim=1)
        log_qy_xsl_mean = self.y_recon_dist.log_pdf(y, *y_params).mean()
        return log_qy_xsl_mean

    def loss_tatol(self, covariate, treatment, s, o_index, y, cuda=True, beta=1, theta=1.):
        """The loss function of model.

        Parameters
        ----------
        covariate : array-like
            The data of covariates.
        treatment : array-like
            The data of the treatment vector.
         s : array-like
            The data of surrogates.
        o_index : array-like
            The indicator is used to select the observed data.
        y : array-like
            Long-term outcome y.
        cuda : bool
            Whether to use CUDA.
        beta : float
        The coefficient of the negative marginal log-likelihood of y.
        theta : float
            The coefficient of the ELBO reconstruction term.

        Returns
        -------
        loss : array-like
            The loss of objective function.
        ELBO : array-like
            Negative evidence Lower Bound (ELBO).
        Ly : array-like
            Negative marginal log-likelihood of y.
        """
        decoder_params, (g, v), s_latent, prior_params, y_params, y_hat= self.forward(s=s,covariate=covariate,
                                                                                      treatment=treatment)

        elbo, sl_est = self.elbo(s=s, decoder_params=decoder_params, g=g, v=v, s_latent=s_latent,
                                 prior_params=prior_params, theta=theta)
        l_yo = self.log_qy_xsl(y=y, y_params=y_params, o_index=o_index) * beta

        loss = elbo.mul(-1) + l_yo.mul(-1)
        return loss, elbo.mul(-1), l_yo.mul(-1)

    def anneal(self, N, max_iter, it):
        """Tune the training hyperparameters.

        Parameters
        ----------
        N : int
            The number of covariates.
        max_iter : int
            The maximum number of epochs.
        it : int
            The current training epoch.

        Returns
        -------
        hyperparams
        """
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False


def IVAE_tx_wrapper(data, batch_size=256, max_epoch=2000, n_layers=3, hidden_dim=200, learn_rate=1e-3, weight_decay=1e-4,
                 activation='lrelu', slope=.1, inference_dim=None, optm='Adam', min_lr=1e-6, base_eopch=200,
                 anneal=False, print_log=True, is_rct=True, cuda=True, normalization=True, beta=1, theta=1,
                 early_stop=True, early_stop_epoch=100, valid_rate=0.2,
                 treatment_dim=1,treated=0.7, control=0.97):
    """Wrap the training function to train the iVAE.

    Parameters
    ----------
    data : array-like
        Input data.
    batch_size : int
        The size of a batch.
    max_epoch : int
        The max number of epochs.
    n_layers : int
        The number of layers in the MLP.
    hidden_dim : array-like
        The dimension of the hidden layers in the MLP.
    learn_rate : float
        Learning rate.
    weight_decay : float
        The weight decay coefficient of the optimizer.
    activation : str or list
        The activation function of the hidden layers.
    slope : float
            The slope of the LeakyReLU activation function.
    inference_dim : int
        The output dimension of the inference network.
    optm : str
        The optimizer.
    min_lr : float
        The minimum lower bound of the learning rate.
    base_eopch : int
        Adjust the optimizer parameters after the specified epoch.
    anneal : bool
        Whether to perform annealing.
    print_log : bool
        Whether to output logs.
    is_rct : bool
        Whether it is a randomized controlled experiment.
    cuda : bool
        Whether to use CUDA.
    normalization : bool
        Whether to perform normalization.
    beta : float
        The coefficient of the negative marginal log-likelihood of y.
    theta : float
        The coefficient of the ELBO reconstruction term.
    early_stop : bool
        Whether early stopping is performed.
    early_stop_epoch : int
        Early stop if performance does not improve at the specified epoch.
    valid_rate : float
        The proportion used to split the validation set.
    treatment_dim : int
        The dimension of the treatment vector.
    treated : Number
        Screen out the treatment group data based on this value.
    control : Number
        Screen out the control group data based on this value.

    Returns
    -------
    losses : array-like
        The recorded loss.
    model : iVAE
        The trained iVAE.
    """
    device = torch.device('cuda' if cuda else 'cpu')
    if print_log:
        print('training on {}'.format(torch.cuda.get_device_name(device) if cuda else 'cpu'))

    # load data
    Obs, Exp, tau_real = data
    xo, to, so, yo = Obs
    xe, te, se, ye = Exp

    y1_index = (te[:,0] == treated).unsqueeze(1)
    y0_index = (te[:,0] == control).unsqueeze(1)

    if normalization:
        # pre_process
        yo_mu = torch.mean(yo)
        yo_std = torch.std(yo)
        yo = (yo - yo_mu) / yo_std

    # o group first, e group second
    o_indicator = torch.cat((1. * torch.ones_like(yo), 0. * torch.zeros_like(ye)), dim=0)
    x = torch.cat((xo, xe), dim=0)
    t = torch.cat((1. * to, 1. * te), dim=0)
    s = torch.cat((so, se), dim=0)
    y = torch.cat((yo, 0. * torch.zeros_like(ye)), dim=0)
    if torch.cuda.is_available() and cuda:
        xe, se, te = xe.cuda(), se.cuda(), te.cuda()
        x, t, s, y, o_indicator = x.cuda(), t.cuda(), s.cuda(), y.cuda(), o_indicator.cuda()

    data_dim = s.shape[1]
    aux_dim = x.shape[1]
    N = x.shape[0]
    if inference_dim is not None:
        latent_dim = inference_dim
    else:
        latent_dim = 10

    # if print_log:
    #     print('Creating shuffled dataset..')
    dataset = TensorDataset(x, t, s, y, o_indicator)
    if early_stop:
        N = x.shape[0]
        train, valid = random_split(dataset, [N - int(valid_rate * N), int(valid_rate * N)])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)
        for x,t,s,y,o in valid_loader:
            x_valid, t_valid, s_valid, y_valid, o_indicator_valid = x,t,s,y,o
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # define model and optimizer
    # if print_log:
    #     print('Defining model and optimizer..')
    model = iVAE_tx(latent_dim, data_dim, aux_dim, activation=activation, device=device,
                 n_layers=n_layers, hidden_dim=hidden_dim, slope=slope, anneal=anneal,
                 treatment_dim=treatment_dim)
    if normalization:
        model.y_mu, model.y_std, model.pre_process_y = yo_mu, yo_std, True

    if torch.cuda.is_available() and cuda:
        model = model.cuda()

    if optm == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    else:  # default: sgd
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=3)

    # training loop
    # if print_log:
    #     print("Training..")
    losses, elbos, l_yos, l_tes, l_inds = [], [], [], [], []
    global_valid_loss, valid_epoch = np.inf, 0

    iter = 0  # 为了计数

    for it in range(max_epoch):
        iter = iter
        for iter, (x, t, s, y, o_indicator) in enumerate(train_loader, iter):
            o_index = o_indicator == 1
            e_index = o_indicator == 0

            # print(s)

            if anneal:
                model.anneal(N, max_epoch, it)
            optimizer.zero_grad()
            loss, elbo, l_yo = model.loss_tatol(covariate=x, treatment=t, s=s, o_index=o_index, y=y,
                                                cuda=True, beta=beta, theta=theta)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            optimizer.step()

            iter += 1

            with torch.no_grad():
                if print_log and iter % 1 == 0:
                    # xe_test = torch.cat((xe, 1. * te),dim=1)
                    ye = model.test(covariate=xe, treatment=1. * te, s=se)
                    # ye = ye * yo_std + yo_mu
                    if is_rct:
                        Ey1 = torch.mean(ye[y1_index])
                        Ey0 = torch.mean(ye[y0_index])
                        ate = Ey1 - Ey0
                        print('naive:epoch={%d}, iter={%d}, loss={%.4f}, ate={%.4f}, real_ate={%.4f}' % (it, iter, loss, ate, tau_real))
                    else:
                        x, t, y = xe.cpu().detach().numpy(), te.cpu().detach().numpy(), ye.cpu().detach().numpy()
                        ate = ipw_estimator(x=x, t=t, y=y)
                        print('ipw:epoch={}, iter={}, loss={}, ate={}'.format(it, len(losses), losses[-1], ate))
        with torch.no_grad():
            # valid loss
            if early_stop:
                o_index_valid = o_indicator_valid == 1
                valid_loss, elbo_, l_yo_ = model.loss_tatol(covariate=x_valid, s=s_valid, treatment=t_valid,
                                                                         o_index=o_index_valid, y=y_valid, cuda=True)
                if valid_loss < global_valid_loss:
                    if print_log:
                        print('update valid loss from {} to {}'.format(global_valid_loss, valid_loss))
                    global_valid_loss = valid_loss
                    model_dist = model.state_dict()
                    valid_epoch = 0
                else:
                    valid_epoch += 1
                    if valid_epoch >= early_stop_epoch:
                        break

        if it > base_eopch:
            scheduler.step(valid_loss)
        # print('current lr={}'.format(optimizer.param_groups[-1]['lr']))
        if optimizer.param_groups[-1]['lr'] < min_lr:
            break
    if early_stop:
        model.load_state_dict(model_dist)
    return losses, model