from itertools import combinations, permutations

import networkx as nx
import numpy as np
from networkx.algorithms.clique import find_cliques
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

from cdmir.utils.independence.functional.kci import KCI


def normalize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def xcum4(X):
    if X.shape[1] != 4:
        raise NotImplementedError

    X = X - np.mean(X, axis=0)
    n = X.shape[0]

    M0123 = np.sum(np.prod(X, axis=1))
    M01 = np.sum(np.prod(X, axis=1, where=[True, True, False, False]))
    M02 = np.sum(np.prod(X, axis=1, where=[True, False, True, False]))
    M03 = np.sum(np.prod(X, axis=1, where=[True, False, False, True]))
    M12 = np.sum(np.prod(X, axis=1, where=[False, True, True, False]))
    M13 = np.sum(np.prod(X, axis=1, where=[False, True, False, True]))
    M23 = np.sum(np.prod(X, axis=1, where=[False, False, True, True]))
    return ((n + 1) * n * M0123 - (n - 1) * (M01 * M23 + M02 * M13 + M03 * M12)) * np.true_divide(1,
                                                                                                  (n - 1) * (n - 2) * (
                                                                                                          n - 3))


def fisher_test(pvals):
    pvals = [pval if pval >= 1e-5 else 1e-5 for pval in pvals]
    fisher_stat = -2.0 * np.sum(np.log(pvals))
    return 1 - chi2.cdf(fisher_stat, 2 * len(pvals))


def surrogate_regression(data, A, X, common_exogenous, mod_omega=False, z=None):
    if mod_omega:
        X = X + [z]
    sub_A = A[np.ix_(X, common_exogenous)]
    _, _, v = np.linalg.svd(sub_A.T)
    omega = v.T[:, -1]
    if mod_omega:
        omega /= omega[-1]
    e = np.dot(omega, data[np.ix_(range(data.shape[0]), X)].T)
    return e


def olc(data, alpha=0.05, beta=0.01, verbose=False):
    n = data.shape[1]
    UDG = nx.Graph()
    CG = nx.DiGraph()

    kci = KCI()
    def cal_pval(x, y):
        return kci.test(x.reshape((-1, 1)), y.reshape((-1, 1)))[0]
 
    for (x, y) in combinations(range(n), 2):
        UDG.add_edge(x, y)

    reg = LinearRegression()

    changed = True
    while changed:
        changed = False
        for x, y in permutations(range(n), 2):
            if not UDG.has_edge(x, y):
                continue
            reg.fit(data[:, [y]], data[:, x])
            res_x = data[:, x] - reg.intercept_
            pval = cal_pval(res_x, data[:, y])
            if verbose:
                print(f"test orient {y} --> {x} pval: {pval}")
            if pval >= alpha:
                changed = True
                UDG.remove_edge(x, y)
                CG.add_edge(y, x)
                data[:, x] = normalize(res_x)
                if verbose:
                    print(f"orient {y} --> {x} pval: {pval}")

    changed = True
    while changed:
        changed = False
        for x in range(n):
            adj_x = list(UDG.neighbors(x))
            if len(adj_x) <= 0:
                continue
            reg.fit(data[:, adj_x], data[:, x])
            e = data[:, x] - reg.intercept_
            pvals = []
            for y in adj_x:
                pvals.append(cal_pval(e, data[:, y]))
            fisher_pval = fisher_test(pvals)
            if fisher_pval >= alpha:
                changed = True
                for y in adj_x:
                    UDG.remove_edge(x, y)
                    CG.add_edge(y, x)
                    if verbose:
                        print(f"parce orient {y} --> {x} pval: {fisher_pval}")

    cov = np.cov(data.T)

    cliques = list(find_cliques(UDG))

    latent_id = 0
    latent_size = n

    surrogate = {x: set() for x in range(n)}
    exogenous = {x: set() for x in range(n)}

    A = np.zeros((latent_size + n, latent_size + n), dtype=float)

    EPS = 1e-5
    partial_observed = set()
    for clique in cliques:
        partial_observed |= set(clique)
    partial_observed = list(partial_observed)

    changed = True
    changed_latent = False
    while changed:
        changed = False
        changed_orient = False
        if changed_latent:
            latent_id += 1
            changed_latent = False

        for (x, y) in combinations(partial_observed, 2):
            if not UDG.has_edge(x, y):
                continue
            common_surrogate = list(surrogate[x] & surrogate[y])
            common_exogenous = list(exogenous[x] & exogenous[y])
            try:
                e_x = surrogate_regression(data=data, A=A, X=[x] + common_surrogate, common_exogenous=common_exogenous)
                pval = cal_pval(e_x, data[:, y])
            except np.linalg.LinAlgError:
                pval = 0

            if pval >= beta:
                UDG.remove_edge(x, y)
                if verbose:
                    print(f"remove edge {x} --- {y}")
            if UDG.has_edge(x, y) and len(common_surrogate) > 0:
                for i in range(2):
                    changedx = False
                    changedy = False
                    var_x = cov[x, x]
                    if np.abs(A[x, x]) < EPS:
                        for i in common_exogenous:
                            var_x -= A[x, i] * A[x, i]
                        A[x, x] = var_x
                        changedx = True

                    ce = cov[x, y]
                    if np.abs(A[y, x]) < EPS:
                        for i in common_exogenous:
                            ce -= A[x, i] * A[y, i]
                        A[y, x] = ce
                        changedy = True

                    try:
                        e_y = surrogate_regression(data=data, A=A, X=[x, y] + common_surrogate,
                                    common_exogenous=common_exogenous + [x])
                        pval = cal_pval(e_y, data[:, x])
                    except np.linalg.LinAlgError:
                        pval = 0

                    if verbose:
                        print(f"test {x} --> {y} pval: {pval}")
                    if pval >= beta:
                        surrogate[y] |= {x}
                        exogenous[x] |= {x}
                        exogenous[y] |= {y}
                        changed_orient = True
                        if UDG.has_edge(x, y):
                            UDG.remove_edge(x, y)
                        CG.add_edge(x, y)
                        if verbose:
                            print(f"{x} --> {y} pval: {pval}")
                        break
                    else:
                        if changedx:
                            A[x, x] = 0.0
                        if changedy:
                            A[y, x] = 0.0
                    x, y = y, x

            # detect latent
            for z in partial_observed:
                if z == x or z == y:
                    continue
                if (not UDG.has_edge(x, z)) and (not UDG.has_edge(y, z)):
                    continue

                changedx = False
                changedy = False
                changedz = False

                common_surrogate = list(
                    (surrogate[x] & surrogate[y]) | (surrogate[x] & surrogate[z]) | (surrogate[y] & surrogate[z]))
                common_exogenous = list(
                    (exogenous[x] & exogenous[y]) | (exogenous[x] & exogenous[z]) | (exogenous[y] & exogenous[z]))

                e_z = None
                if len(common_surrogate) > 0:
                    try:
                        e_z = surrogate_regression(data=data, A=A, X=common_surrogate, common_exogenous=common_exogenous,
                                    mod_omega=True, z=z)

                    except np.linalg.LinAlgError:
                        e_z = data[:, z]

                else:
                    e_z = data[:, z]

                if np.abs(A[z, latent_id + n]) < EPS:
                    if np.abs(A[x, latent_id + n]) >= EPS:
                        temp_cov = np.cov(np.array([e_z, data[:, x]]))
                        A[z, latent_id + n] = temp_cov[0, 1] / A[x, latent_id + n]
                    elif np.abs(A[y, latent_id + n]) >= EPS:
                        temp_cov = np.cov(np.array([e_z, data[:, y]]))
                        A[z, latent_id + n] = temp_cov[0, 1] / A[y, latent_id + n]
                    else:
                        temp_cov = np.cov(np.array([e_z, data[:, x]]))
                        # print(f"cov : {temp_cov}")
                        A[z, latent_id + n] = np.sqrt(np.abs(temp_cov[0, 1] * xcum4(
                            np.vstack((data[:, x], e_z, e_z, e_z)).T) / xcum4(
                            np.vstack((data[:, x], data[:, x], e_z, e_z)).T)))
                    changedz = True


                if np.abs(A[x, latent_id + n]) < EPS:
                    temp_cov = np.cov(np.array([e_z, data[:, x]]))
                    ce = temp_cov[0, 1]
                    A[x, latent_id + n] = ce / A[z, latent_id + n]
                    changedx = True

                if np.abs(A[y, latent_id + n]) < EPS:
                    temp_cov = np.cov(np.array([e_z, data[:, y]]))
                    ce = temp_cov[0, 1]
                    A[y, latent_id + n] = ce / A[z, latent_id + n]
                    changedy = True
                try:
                    e = surrogate_regression(data=data, A=A, X=common_surrogate + [x, y],
                              common_exogenous=common_exogenous + [latent_id + n])
                    pval = cal_pval(e, data[:, z])
                except np.linalg.LinAlgError:
                    pval = 0
                if verbose:
                    print(f"test pure({x},{y}|{z}) pval : {pval}")
                if pval >= beta:
                    surrogate[x] |= {z}
                    surrogate[y] |= {z}
                    exogenous[x] |= {latent_id + n}
                    exogenous[y] |= {latent_id + n}
                    exogenous[z] |= {latent_id + n}
                    CG.add_edge(latent_id + n, x)
                    CG.add_edge(latent_id + n, y)
                    CG.add_edge(latent_id + n, z)
                    if UDG.has_edge(x, z):
                        UDG.remove_edge(x, z)
                    if UDG.has_edge(y, z):
                        UDG.remove_edge(y, z)

                    changed_latent = True
                    if verbose:
                        print(f"remove edge {x} --- {z} with {common_exogenous}, {latent_id + n}")
                        print(f"remove edge {y} --- {z} with {common_exogenous}, {latent_id + n}")
                    break
                else:
                    if changedx:
                        A[x, latent_id + n] = 0
                    if changedy:
                        A[y, latent_id + n] = 0
                    if changedz:
                        A[z, latent_id + n] = 0
        if changed_latent or changed_orient:
            changed = True


    adjmat = np.zeros(shape=(latent_id + n, latent_id + n), dtype=int)
    for (x, y) in UDG.edges():
        adjmat[x, y] = adjmat[y, x] = 2

    for (x, y) in CG.edges():
        adjmat[y, x] = 1

    coef = np.zeros(shape=(latent_id + n, latent_id + n), dtype=float)

    for (x, y) in CG.edges():
        if x < n and y < n:
            coef[y, x] = A[y, x] / A[x, x]
        else:
            coef[y, x] = A[y, x]

    return adjmat, coef
