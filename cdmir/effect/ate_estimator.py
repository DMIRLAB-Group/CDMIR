import sys

import numpy

sys.path.append('..')

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from scipy.special import expit, logit

def e_x_estimator(x, w):
    """estimate P(W_i=1|X_i=x)"""
    log_reg = LogisticRegression().fit(x, w)
    return log_reg


def naive_estimator(t, y):
    """
    naive estimator of ATE
    estimate E[Y|T=1] - E[Y|T=0]

    Parameters:
    ---------
    t: treatment indicator (n_samples, 1)
    y: outcome (n_samples, 1)

    Returns:
    ---------
    tau: estimated ATE
    """

    # Get index of treatment = 1 and treatment = 0
    index_t1 = np.squeeze(t == 1)
    index_t0 = np.squeeze(t == 0)

    # Extract Y|T=1 and Y|T=0
    y1 = y[index_t1,]
    y0 = y[index_t0,]

    # Calculate ATE
    tau = np.mean(y1) - np.mean(y0)
    return tau


def ipw_estimator(x, t, y):
    """estimate ATE using ipw method"""
    propensity_socre_reg = e_x_estimator(x, t)
    propensity_socre = propensity_socre_reg.predict_proba(x)
    propensity_socre = propensity_socre[:, 1][:, None]  # prob of treatment=1

    ps1 = 1. / np.sum(t / propensity_socre)
    y1 = ps1 * np.sum(y * t / propensity_socre)
    ps0 = 1. / np.sum((1. - t) / (1. - propensity_socre))
    y0 = ps0 * np.sum(y * ((1. - t) / (1 - propensity_socre)))
    # print((1. - t).sum())
    # print(t.sum())

    tau = y1 - y0
    return tau


def s_learner_estimator(x, t, y, regression=LinearRegression()):
    """ 
    s_learner: naive estimator using same regression function
    estimate E(Y|X,T=1)-E(Y|X,T=0)

    Parameters:
    ---------
    x: input data (n_samples, n_features)
    t: treatment indicator (n_samples, 1)
    y: outcome data (n_samples, 1)
    regression: regression function, default = LinearRegression()

    Returns:   
    ---------
    tau: estimated ATE 
    """

    # Concatenate covariates (X) with treatment indicator (T)
    x_t = np.concatenate((x, t), axis=1)

    # Fit regression model by combined data (X, T) and outcome (Y)
    regression.fit(X=x_t, y=y)

    # Create data for two hypothetical scenarios:
    # All samples receive treatment (t=1)
    x_t1 = np.concatenate((x, numpy.ones_like(t)), axis=1)
    # All samples receive treatment (t=0)
    x_t0 = np.concatenate((x, numpy.zeros_like(t)), axis=1)

    # Predict the results under two scenarios
    y1 = regression.predict(X=x_t1)
    y0 = regression.predict(X=x_t0)

    # Calculate ATE
    tau = np.mean(y1 - y0)
    return tau


def t_learner_estimator(x, t, y, regression_1=LinearRegression(), regression_0=LinearRegression()):
    """ estimate E(Y|X,T=1)-E(Y|X,T=0)
        t_learner: naive estimator using different regression function
    """
    index_t1 = np.squeeze(t == 1)
    index_t0 = np.squeeze(t == 0)
    x_t1 = np.concatenate((x[index_t1,], t[index_t1,]), axis=1)
    x_t0 = np.concatenate((x[index_t0,], t[index_t0,]), axis=1)
    regression_1.fit(X=x_t1, y=y[index_t1,])
    regression_0.fit(X=x_t0, y=y[index_t0,])
    x_t1 = np.concatenate((x, numpy.ones_like(t)), axis=1)
    x_t0 = np.concatenate((x, numpy.zeros_like(t)), axis=1)
    y1 = regression_1.predict(X=x_t1)
    y0 = regression_0.predict(X=x_t0)

    tau = np.mean(y1 - y0)
    return tau


def x_learner_estimator(X, T, Y,  model_y=LinearRegression(), model_tau=LinearRegression(), prop_model=LogisticRegression()):
    if model_y is None:
        model_y = RandomForestRegressor(n_estimators=100)
    if model_tau is None:
        model_tau = RandomForestRegressor(n_estimators=100)
    if prop_model is None:
        prop_model = LogisticRegression()

    # Step 1: fit Y(0)  Y(1)
    model_y0 = clone(model_y)
    model_y1 = clone(model_y)
    model_y0.fit(X[T == 0], Y[T == 0])
    model_y1.fit(X[T == 1], Y[T == 1])

    # Step 2: obtain psd-effect
    D1 = Y[T == 1] - model_y0.predict(X[T == 1])  # 处理组的伪效应
    D0 = model_y1.predict(X[T == 0]) - Y[T == 0]  # 对照组的伪效应

    # Step 3: fit tau using psd-effect
    tau1_model = clone(model_tau)
    tau0_model = clone(model_tau)
    tau1_model.fit(X[T == 1], D1)
    tau0_model.fit(X[T == 0], D0)

    # Step 4: propensity
    prop_model.fit(X, T)
    p = prop_model.predict_proba(X)[:, 1]  # P(T=1|X)

    # Step 5: reweight by propensity
    tau1 = tau1_model.predict(X)
    tau0 = tau0_model.predict(X)
    tau_x = (1 - p) * tau1 + p * tau0

    return np.mean(tau_x)

def double_robust_estimator(X, T, Y, outcome_model=LinearRegression()):
    if outcome_model is None:
        outcome_model = RandomForestRegressor(n_estimators=100)

    T = T.reshape(-1)
    # 1. fit μ0(x)  μ1(x)
    mu0_model = clone(outcome_model)
    mu1_model = clone(outcome_model)
    mu0_model.fit(X[T == 0], Y[T == 0])
    mu1_model.fit(X[T == 1], Y[T == 1])
    y0 = mu0_model.predict(X)
    y1 = mu1_model.predict(X)

    # logistic regression (generalized linear)
    propensity_socre_reg = e_x_estimator(X, T)
    propensity_socre = propensity_socre_reg.predict_proba(X)
    propensity_socre = propensity_socre[:, 1][:, None]  # prob of treatment=1

    T = T[:,None]

    tau = np.mean(T * (Y - y1) / propensity_socre + y1) - \
          np.mean((1-T) * (Y - y0) / (1-propensity_socre) + y0)

    return tau


if __name__ == '__main__':
    # test

    n, p = 1000, 5
    np.random.seed(42)
    X = np.random.normal(1, 1, (n, p))
    T = np.random.binomial(1, 0.5, n)
    tau = X[:, 0]
    y0 = X @ np.array([1, 2, 0, 0, 0]) + np.random.normal(0, 1, n)
    y1 = y0 + tau
    Y = T * y1 + (1 - T) * y0
    T,Y = T[:,None],Y[:,None]

    print('groud true  ' + str(np.mean(tau)))
    print(naive_estimator(T,Y))
    print(s_learner_estimator(X,T,Y))
    print(t_learner_estimator(X,T,Y))
    print(ipw_estimator(X,T,Y))
    print(double_robust_estimator(X,T,Y))
