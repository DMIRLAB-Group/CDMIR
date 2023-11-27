from __future__ import annotations

import itertools
import warnings
from copy import deepcopy

import numpy as np

from causaldmir.graph import dag2cpdag, Edge, Mark
from causaldmir.graph.pdag2dag import pdag2dag
from causaldmir.utils.local_score import *
from causaldmir.graph.pdag import PDAG


def kernel(x, xKern, theta):
    # KERNEL Compute the rbf kernel
    n2 = dist2(x, xKern)
    if (theta[0] == 0):
        theta[0] = 2 / np.median(n2[np.where(np.tril(n2) > 0)])
        theta_new = theta[0]
    wi2 = theta[0] / 2
    kx = theta[1] * np.exp(-n2 * wi2)
    bw_new = 1 / theta[0]
    return kx, bw_new


def Combinatorial(T0):
    # sub = Combinatorial (T0); % find all the sbusets of T0
    sub = []
    count = 0
    if (len(T0) == 0):
        sub.append(())  # a 1x0 empty matrix
    else:
        if (len(T0) == 1):
            sub.append(())
            sub.append(T0)  # when T0 is a scale, it is a special case!!
        else:
            for n in range(len(T0) + 1):
                for S in list(itertools.combinations(T0, n)):
                    sub.append(S)
    return sub


def Score_G(G, score_func):  # calculate the score for the current G
    # here G is a DAG
    score = 0
    for i, node in enumerate(G.node_list):
        PA = []
        for node_v in G.get_neighbours(node):
            if G.is_arrow(node_v, node):
                PA.append(node_v)

        delta_score = score_func(i, PA)
        score = score + delta_score
    return score


def Insert_validity_test1(G, i, j, T):
    # V=Insert_validity_test1(G, X, Y, T,1); % do validity test for the operator Insert; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0

    # condition 1
    Tj = np.intersect1d(np.where([G.is_tail(G.node_list[j], G.node_list[j_tail]) for j_tail in range(G.number_of_nodes())])[0], np.where([G.is_tail(G.node_list[j_tail], G.node_list[j]) for j_tail in range(G.number_of_nodes())])[0])  # neighbors of Xj
    Ti = np.where([G.is_connected(G.node_list[i], G.node_list[i_nei]) for i_nei in range(G.number_of_nodes())])[0]  # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti)  # find the neighbours of Xj and are adjacent to Xi
    V = check_clique(G, list(np.union1d(NA, T).astype(int)))  # check whether it is a clique
    return V


def check_clique(G, subnode):  # check whether node subnode is a clique in G
    # here G is a CPDAG
    # the definition of clique here: a clique is defined in an undirected graph
    # when you ignore the directionality of any directed edges

    Gs = np.zeros((G.number_of_nodes(), G.number_of_nodes()), dtype=int)
    for edge in G.edges:
        Gs[G.node_list.index(edge.node_u), G.node_list.index(edge.node_v)] = edge.mark_u.value
        Gs[G.node_list.index(edge.node_v), G.node_list.index(edge.node_u)] = edge.mark_v.value
    Gs = deepcopy(Gs[np.ix_(subnode, subnode)])  # extract the subgraph
    ns = len(subnode)

    if ns == 0:
        s = 1
    else:
        row, col = np.where(Gs == 1)
        Gs[row, col] = -1
        Gs[col, row] = -1
        if np.all((np.eye(ns) - np.ones((ns, ns))) == Gs):  # check whether it is a clique
            s = 1
        else:
            s = 0
    return s


def Insert_validity_test2(G, i, j, T):
    # V=Insert_validity_test(G, X, Y, T,1); % do validity test for the operator Insert; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0
    Tj = np.intersect1d(np.where([G.is_tail(G.node_list[j], G.node_list[j_tail]) for j_tail in range(G.number_of_nodes())])[0], np.where([G.is_tail(G.node_list[j_tail], G.node_list[j]) for j_tail in range(G.number_of_nodes())])[0])  # neighbors of Xj
    Ti = np.where([G.is_connected(G.node_list[i], G.node_list[i_nei]) for i_nei in range(G.number_of_nodes())])[0]  # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti)  # find the neighbours of Xj and are adjacent to Xi

    # condition 2: every semi-directed path from Xj to Xi contains a node in union(NA,T)
    # Note: EVERY!!
    s2 = Insert_vC2_new(G, j, i, np.union1d(NA, T))
    if s2:
        V = 1

    return V


def Insert_vC2_new(G, j, i, NAT):  # validity test for condition 2 of Insert operator
    # here G is CPDAG
    # Use Depth-first-Search
    start = j
    target = i
    # stack(1)=start; % initialize the stack
    stack = [{'value': start, 'pa': {}}]
    sign = 1  # If every semi-pathway contains a node in NAT, than sign=1;

    while len(stack):
        top = stack[0]
        stack = stack[1:]  # pop
        if top['value'] == target:  # if find the target, search that pathway to see whether NAT is in that pathway
            curr = top
            ss = 0
            while True:
                if len(curr['pa']):
                    if curr['pa']['value'] in NAT:  # contains a node in NAT
                        ss = 1
                        break
                else:
                    break
                curr = curr['pa']
            if not ss:  # do not include NAT
                sign = 0
                break
        else:
            child = np.concatenate((np.where([G.is_arrow(G.node_list[top['value']], G.node_list[other]) for other in range(G.number_of_nodes())])[0],
                                    np.intersect1d(np.where([G.is_tail(G.node_list[top['value']], G.node_list[other]) for other in range(G.number_of_nodes())])[0], np.where([G.is_tail(G.node_list[other], G.node_list[top['value']]) for other in range(G.number_of_nodes())])[0])))
            sign_child = np.ones(len(child))
            # check each child, whether it has appeared before in the same pathway
            for k in range(len(child)):
                curr = top
                while True:
                    if len(curr['pa']):
                        if curr['pa']['value'] == child[k]:
                            sign_child[k] = 0  # has appeared in that path before
                            break
                    else:
                        break
                    curr = curr['pa']

            for k in range(len(sign_child)):
                if sign_child[k]:
                    stack.insert(0, {'value': child[k], 'pa': top})  # push
    return sign


def find_subset_include(s0, sub):
    # S = find_subset_include(sub(k),sub); %  find those subsets that include sub(k)
    if len(s0) == 0 or len(sub) == 0:
        Idx = np.ones(len(sub))
    else:
        Idx = np.zeros(len(sub))
        for i in range(len(sub)):
            tmp = set(s0).intersection(set(sub[i]))
            if len(tmp):
                if tmp == set(s0):
                    Idx[i] = 1
    return Idx


def Insert_changed_score(G, i, j, T, record_local_score, score_func):
    # calculate the changed score after the insert operator: i->j
    Tj = np.intersect1d(np.where([G.is_tail(G.node_list[j], G.node_list[j_tail]) for j_tail in range(G.number_of_nodes())])[0], np.where([G.is_tail(G.node_list[j_tail], G.node_list[j]) for j_tail in range(G.number_of_nodes())])[0])  # neighbors of Xj
    Ti = np.where([G.is_connected(G.node_list[i], G.node_list[i_nei]) for i_nei in range(G.number_of_nodes())])[0]  # adjacent to Xi;
    NA = np.intersect1d(Tj, Ti)  # find the neighbours of Xj and are adjacent to Xi
    Paj = np.where([G.is_arrow(G.node_list[j_parent], G.node_list[j]) for j_parent in range(G.number_of_nodes())])[0]  # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = np.union1d(NA, T).astype(int)
    tmp2 = np.union1d(tmp1, Paj)
    tmp3 = np.union1d(tmp2, [i]).astype(int)

    # before you calculate the local score, firstly you search in the
    # "record_local_score", to see whether you have calculated it before
    r = len(record_local_score[j])
    s1 = 0
    s2 = 0

    for r0 in range(r):
        if not np.setxor1d(record_local_score[j][r0][0:-1], tmp3).size:
            score1 = record_local_score[j][r0][-1]
            s1 = 1

        if (not np.setxor1d(record_local_score[j][r0][0:-1],
                            tmp2).size):  # notice the differnece between 0*0 empty matrix and 1*0 empty matrix
            score2 = record_local_score[j][r0][-1]
            s2 = 1
        else:
            if (not np.setxor1d(record_local_score[j][r0][0:-1], [-1]).size) and (not tmp2.size):
                score2 = record_local_score[j][r0][-1]
                s2 = 1

        if s1 and s2:
            break

    if not s1:
        score1 = score_func(j, tmp3)
        temp = list(tmp3)
        temp.append(score1)
        record_local_score[j].append(temp)

    if not s2:
        score2 = score_func(j, tmp2)
        # r = len(record_local_score[j])
        if len(tmp2) != 0:
            temp = list(tmp2)
            temp.append(score2)
            record_local_score[j].append(temp)
        else:
            record_local_score[j].append([-1, score2])

    chscore = score1 - score2
    desc = [i, j, T]
    return chscore, desc, record_local_score


def Insert(G, i, j, T):
    # Insert operator
    # insert the directed edge Xi->Xj
    nodes = G.node_list
    G.add_edge(Edge(nodes[i], nodes[j], Mark.Tail, Mark.ARROW), overwrite=True)

    for k in range(len(T)):  # directing the previous undirected edge between T and Xj as T->Xj
        if G.get_edge(nodes[T[k]], nodes[j]) is not None:
            G.remove_edge(nodes[T[k]], nodes[j])
        G.add_edge(Edge(nodes[T[k]], nodes[j], Mark.Tail, Mark.ARROW), overwrite=True)

    return G


def Delete_validity_test(G, i, j, H):
    # V=Delete_validity_test(G, X, Y, H); % do validity test for the operator Delete; V=1 means valid, V=0 mean invalid;
    # here G is CPDAG
    V = 0

    # condition 1
    Hj = np.intersect1d(np.where([G.is_tail(G.node_list[j], G.node_list[j_tail]) for j_tail in range(G.number_of_nodes())])[0], np.where([G.is_tail(G.node_list[j_tail], G.node_list[j]) for j_tail in range(G.number_of_nodes())])[0])  # neighbors of Xj
    Hi = np.where([G.is_connected(G.node_list[i], G.node_list[i_nei]) for i_nei in range(G.number_of_nodes())])[0]  # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi)  # find the neighbours of Xj and are adjacent to Xi
    s1 = check_clique(G, list(set(NA) - set(H)))  # check whether it is a clique

    if s1:
        V = 1

    return V


def Delete_changed_score(G, i, j, H, record_local_score, score_func):
    # calculate the changed score after the Delete operator
    Hj = np.intersect1d(np.where([G.is_tail(G.node_list[j], G.node_list[j_tail]) for j_tail in range(G.number_of_nodes())])[0], np.where([G.is_tail(G.node_list[j_tail], G.node_list[j]) for j_tail in range(G.number_of_nodes())])[0])  # neighbors of Xj
    Hi = np.where([G.is_connected(G.node_list[i], G.node_list[i_nei]) for i_nei in range(G.number_of_nodes())])[0]  # adjacent to Xi;
    NA = np.intersect1d(Hj, Hi)  # find the neighbours of Xj and are adjacent to Xi
    Paj = np.union1d(np.where([G.is_arrow(G.node_list[j_parent], G.node_list[j]) for j_parent in range(G.number_of_nodes())])[0], [i])  # find the parents of Xj
    # the function local_score() calculates the local score
    tmp1 = set(NA) - set(H)
    tmp2 = set.union(tmp1, set(Paj))
    tmp3 = tmp2 - {i}

    # before you calculate the local score, firstly you search in the
    # "record_local_score", to see whether you have calculated it before
    r = len(record_local_score[j])
    s1 = 0
    s2 = 0

    for r0 in range(r):
        if set(record_local_score[j][r0][0:-1]) == tmp3:
            score1 = record_local_score[j][r0][-1]
            s1 = 1

        if set(record_local_score[j][r0][0:-1]) == tmp2:  # notice the differnece between 0*0 empty matrix and 1*0 empty matrix
            score2 = record_local_score[j][r0][-1]
            s2 = 1
        else:
            if (set(record_local_score[j][r0][0:-1]) == {-1}) and len(tmp2) == 0:
                score2 = record_local_score[j][r0][-1]
                s2 = 1

        if s1 and s2:
            break

    if not s1:
        score1 = score_func(j, list(tmp3))
        temp = list(tmp3)
        temp.append(score1)
        record_local_score[j].append(temp)

    if not s2:
        score2 = score_func(j, list(tmp2))
        if len(tmp2) != 0:
            temp = list(tmp2)
            temp.append(score2)
            record_local_score[j].append(temp)
        else:
            record_local_score[j].append([-1, score2])

    chscore = score1 - score2
    desc = [i, j, H]
    return chscore, desc, record_local_score


def Delete(G, i, j, H):
    # Delete operator
    nodes = G.node_list
    if G.get_edge(nodes[i], nodes[j]) is not None:
        # delete the edge between Xi and Xj
        G.remove_edge(nodes[i], nodes[j])
    for k in range(len(H)):  # directing the previous undirected edge
        if G.get_edge(nodes[j], nodes[H[k]]) is not None:
            G.remove_edge(nodes[j], nodes[H[k]])
        if G.get_edge(nodes[i], nodes[H[k]]) is not None:
            G.remove_edge(nodes[i], nodes[H[k]])
        G.add_edge(Edge(nodes[j], nodes[H[k]], Mark.Tail, Mark.ARROW), overwrite=True)
        G.add_edge(Edge(nodes[i], nodes[H[k]], Mark.Tail, Mark.ARROW), overwrite=True)
    return G


def dist2(x, c):
    # DIST2	Calculates squared distance between two sets of points.
    #
    # Description
    # D = DIST2(X, C) takes two matrices of vectors and calculates the
    # squared Euclidean distance between them.  Both matrices must be of
    # the same column dimension.  If X has M rows and N columns, and C has
    # L rows and N columns, then the result has M rows and L columns.  The
    # I, Jth entry is the  squared distance from the Ith row of X to the
    # Jth row of C.
    #
    # See also
    # GMMACTIV, KMEANS, RBFFWD
    #

    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if (dimx != dimc):
        raise Exception('Data dimension does not match dimension of centres')

    n2 = (np.matlib.ones((ncentres, 1)) * np.sum(np.multiply(x, x).T, axis=0)).T + \
         np.matlib.ones((ndata, 1)) * np.sum(np.multiply(c, c).T, axis=0) - \
         2 * (x * c.T)

    # Rounding errors occasionally cause negative entries in n2
    n2[np.where(n2 < 0)] = 0
    return n2


def pdinv(A):
    # PDINV Computes the inverse of a positive definite matrix
    numData = A.shape[0]
    try:
        U = np.linalg.cholesky(A).T
        invU = np.eye(numData).dot(np.linalg.inv(U))
        Ainv = invU.dot(invU.T)
    except np.linalg.LinAlgError as e:
        warnings.warn('Matrix is not positive definite in pdinv, inverting using svd')
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        Ainv = vh.T.dot(np.diag(1 / s)).dot(u.T)
    except Exception as e:
        raise e
    return np.matlib.asmatrix(Ainv)



class GES(object):

    def __init__(self, score_function, max_p: int | None = None):
        if not issubclass(type(score_function), BaseLocalScoreFunction):
            raise Exception("'score_function' must be subclass of 'BaseLocalScoreFunction'!")
        self.score_function = score_function

        if type(score_function) == MultiCVScore or type(score_function) == MultiMarginalScore:
            self.var_count = len(score_function.d_label)
        else:
            self.var_count = score_function.data.shape[1]
        if max_p is None:
            max_p = self.var_count / 2
        self.max_p = max_p
        self.__causal_graph = None
        self.__score = None

    def get_causal_graph(self):
        if self.__causal_graph is None:
            raise Exception("please fit some data with this algorithm before get causal graph!")
        return self.__causal_graph

    def get_score(self):
        if self.__score is None:
            raise Exception("please fit some data with this algorithm before get score!")
        return self.__score

    def fit(self):
        # Greedy equivalence search
        # INPUT:
        # X: Data with T*D dimensions
        # score_func: the score function you want to use
        # maxP: allowed maximum number of parents when searching the graph
        # parameters: when using CV likelihood,
        #               parameters['kfold']: k-fold cross validation
        #               parameters['lambda']: regularization parameter
        #               parameters['dlabel']: for variables with multi-dimensions,
        #                            indicate which dimensions belong to the i-th variable.


        node_names = [("x%d" % i) for i in range(self.var_count)]

        G = PDAG(node_names)
        # G = np.matlib.zeros((N, N)) # initialize the graph structure
        score = Score_G(G, score_func=self.score_function)  # initialize the score

        G = pdag2dag(G)
        G = dag2cpdag(G)

        ## --------------------------------------------------------------------
        ## forward greedy search
        record_local_score = [[] for _ in range(self.var_count)]  # record the local score calculated each time. Thus when we transition to the second phase, many of the operators can be scored without an explicit call the the scoring function
        # record_local_score{trial}{j} record the local scores when Xj as a parent
        score_new = score
        count = 0

        while True:
            count += 1
            score = score_new
            max_chscore = -1e7
            max_ac = []
            for i in range(self.var_count):
                for j in range(self.var_count):
                    if not G.is_connected(G.node_list[i], G.node_list[j]) and i != j and len(np.where([G.is_arrow(G.node_list[j_parent], G.node_list[j]) for j_parent in range(G.number_of_nodes())])[0]) <= self.max_p:  # find a pair (Xi, Xj) that is not adjacent in the current graph , and restrict the number of parents
                        Tj = np.intersect1d(np.where([G.is_tail(G.node_list[j], G.node_list[j_tail]) for j_tail in range(G.number_of_nodes())])[0], np.where([G.is_tail(G.node_list[j_tail], G.node_list[j]) for j_tail in range(G.number_of_nodes())])[0])  # neighbors of Xj
                        Ti = np.where([G.is_connected(G.node_list[i], G.node_list[i_nei]) for i_nei in range(G.number_of_nodes())])[0]  # adjacent to Xi
                        NTi = np.setdiff1d(np.arange(self.var_count), Ti)
                        T0 = np.intersect1d(Tj, NTi)  # find the neighbours of Xj that are not adjacent to Xi
                        # for any subset of T0
                        sub = Combinatorial(T0.tolist())  # find all the subsets for T0
                        S = np.zeros(len(sub))
                        # S indicate whether we need to check sub{k}.
                        # 0: check both conditions.
                        # 1: only check the first condition
                        # 2: check nothing and is not valid.
                        for k in range(len(sub)):
                            if S[k] < 2:  # S indicate whether we need to check subset(k)
                                V1 = Insert_validity_test1(G, i, j, sub[k])  # Insert operator validation test:condition 1
                                if V1:
                                    if not S[k]:
                                        V2 = Insert_validity_test2(G, i, j, sub[k])  # Insert operator validation test:condition 2
                                    else:
                                        V2 = 1
                                    if V2:
                                        Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
                                        S[np.where(Idx == 1)] = 1
                                        chscore, ac, record_local_score = Insert_changed_score(G, i, j, sub[k], record_local_score, self.score_function)  # calculate the changed score after Insert operator
                                        # desc{count} saves the corresponding (i,j,sub{k})
                                        if chscore > max_chscore:
                                            max_chscore = chscore
                                            max_ac = ac
                                else:
                                    Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
                                    S[np.where(Idx == 1)] = 2

            if len(max_ac) != 0:
                score_new = score + max_chscore
                if score_new - score <= 0:
                    break
                G = Insert(G, max_ac[0], max_ac[1], max_ac[2])
                G = pdag2dag(G)
                G = dag2cpdag(G)
            else:
                score_new = score
                break

        ## --------------------------------------------------------------------
        # backward greedy search
        score_new = score
        while True:
            score = score_new
            max_chscore = -1e7
            max_ac = []
            for i in range(self.var_count):
                for j in range(self.var_count):
                    if (G.is_tail(G.node_list[j], G.node_list[i]) and G.is_tail(G.node_list[i], G.node_list[j])) or G.is_arrow(G.node_list[i], G.node_list[j]):  # if Xi - Xj or Xi -> Xj
                        Hj = np.intersect1d(np.where([G.is_tail(G.node_list[j], G.node_list[j_tail]) for j_tail in range(G.number_of_nodes())])[0], np.where([G.is_tail(G.node_list[j_tail], G.node_list[j]) for j_tail in range(G.number_of_nodes())])[0])  # neighbors of Xj
                        Hi = np.where([G.is_connected(G.node_list[i], G.node_list[i_nei]) for i_nei in range(G.number_of_nodes())])[0]  # adjacent to Xi
                        H0 = np.intersect1d(Hj, Hi)  # find the neighbours of Xj that are adjacent to Xi
                        # for any subset of H0
                        sub = Combinatorial(H0.tolist())  # find all the subsets for H0
                        S = np.ones(len(sub))  # S indicate whether we need to check sub{k}.
                        # 1: check the condition,
                        # 2: check nothing and is valid;
                        for k in range(len(sub)):
                            if S[k] == 1:
                                V = Delete_validity_test(G, i, j, sub[k])  # Delete operator validation test
                                if V:
                                    # find those subsets that include sub(k)
                                    Idx = find_subset_include(sub[k], sub)
                                    S[np.where(Idx == 1)] = 2  # and set their S to 2
                            else:
                                V = 1

                            if V:
                                chscore, ac, record_local_score = Delete_changed_score(G, i, j, sub[k], record_local_score, self.score_function)  # calculate the changed score after Insert operator
                                # desc{count} saves the corresponding (i,j,sub{k})
                                if chscore > max_chscore:
                                    max_chscore = chscore
                                    max_ac = ac

            if len(max_ac) != 0:
                score_new = score + max_chscore
                if score_new - score <= 0:
                    break
                G = Delete(G, max_ac[0], max_ac[1], max_ac[2])
                G = pdag2dag(G)
                G = dag2cpdag(G)

            else:
                score_new = score
                break

        self.__causal_graph = G
        self.__score = score
