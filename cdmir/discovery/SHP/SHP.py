from __future__ import print_function
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from itertools import count, product
from Generate_Hawkes_data_from_tick import generate_data
from functools import partialmethod


#tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

__MIN__ = -np.inf


def get_interval_events(event_table, time_interval=20):
    """
    Discretize event timestamps into time intervals and organize events by sequence ID and event type

    :param event_table : pandas.DataFrame
        Event data table containing columns 'time_stamp', 'event_type', and 'seq_id'
    :param time_interval : int, default=20
        Time interval size for discretizing timestamps


    :return: list
        3D list where the first dimension is sequence ID,
        second dimension is event type,
        and third dimension is array of timestamps for that sequence and event type
    """

    # Discretize timestamps to specified time intervals
    event_table['time_stamp'] = (event_table['time_stamp'] / time_interval).astype('int') * time_interval

    # Group and organize timestamps by sequence ID and event type
    events = [[event_table[(event_table['event_type'] == i) & (event_table['seq_id'] == j)][
                   'time_stamp'].values.astype('float') for i in
               np.sort((event_table['event_type']).unique())] for j in tqdm((event_table['seq_id']).unique())]
    return events


def check_DAG(edge_mat):
    c_g = nx.from_numpy_array(edge_mat - np.diag(np.diag(edge_mat)), create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(c_g)


class SHP(object):
    """
    Python implementation of SHP algorithm

    Reference:
    ----------
    [1] Qiao J, Cai R, Wu S, et al. Structural hawkes processes for learning causal structure from discrete-time event sequences[J]. arXiv preprint arXiv:2305.05986, 2023.
    """
    def __init__(self, event_table: pd.DataFrame, decay, time_interval=None,
                 init_structure: np.array = None,
                 penalty='BIC', seed=None, reg=3.0):
        """
        Construct the SHP model
        :param event_table: A pandas.DataFrame of events with columns  ['seq_id', 'time_stamp', 'event_type']
        :param decay: The decay used in the exponential kernel
        :param init_structure: adj of causal structure of prior knowledge
        :param penalty: 'BIC' or 'AIC' penalty
        """

        self.random_state = np.random.RandomState(seed)
        self.reg = reg
        self.time_interval = time_interval  # Delta t

        # create the event_table
        self.event_table, self.event_names = self.get_event_table(event_table)

        # set the start timestamp to zero
        for seq_id in np.unique(self.event_table["seq_id"].values):
            seq_index = self.event_table[self.event_table["seq_id"] == seq_id].index.tolist()
            self.event_table.loc[seq_index, "max_time_stamp"] = \
                self.event_table.loc[seq_index]["max_time_stamp"] - self.event_table.loc[seq_index]["time_stamp"].min()
            self.event_table.loc[seq_index, "time_stamp"] = \
                self.event_table.loc[seq_index]["time_stamp"] - self.event_table.loc[seq_index]["time_stamp"].min()

        # store the calculated likelihood
        self.hist_likelihood = dict()
        for i in range(len(self.event_names)):
            self.hist_likelihood[i] = dict()

        if penalty not in {'BIC', 'AIC'}:
            raise Exception('Penalty is not supported')
        self.penalty = penalty

        self.decay = decay  # the decay coefficient of kappa function
        self.n = len(self.event_names)  # num of event type
        self.T = self.event_table.groupby('seq_id').apply(lambda i: (i['time_stamp'].max())).sum()  # total time span
        self.T_each_seq = self.event_table.groupby('seq_id'). \
            apply(lambda i: (i['time_stamp'].max()))  # the last moment of each event sequence

        # Initializing Structs
        if init_structure is None:
            self.init_structure = np.zeros([self.n, self.n])
        elif not ((init_structure == 1) | (init_structure == 0)).all():
            raise ValueError('Elements of the adjacency matrix need to be 0 or 1')
        else:
            self.init_structure = np.array(init_structure)

        X_dict = dict()
        for seq_id, time_stamp, _, times, type_ind, _ in self.event_table.values:
            if (seq_id, time_stamp) not in X_dict:
                X_dict[(seq_id, time_stamp)] = [0] * self.n
            X_dict[(seq_id, time_stamp)][type_ind] = times
        self.X_df = pd.DataFrame(X_dict).T

        self.X = self.X_df.values
        self.sum_t_X_kappa, self.decay_effect_integral_to_T = self.calculate_influence_of_each_event()

    def calculate_influence_of_each_event(self):
        """
        calculate the influence of each event

        :return:
        """
        sum_t_X_kappa = np.zeros_like(self.X, dtype='float64')
        decay_effect_integral_to_T = self.X_df.copy()

        for ind, (seq_id, time_stamp) in tqdm(enumerate(self.X_df.index)):
            # calculate the integral of decay function on time
            decay_effect_integral_to_T.iloc[ind] = \
                self.X_df.iloc[ind] * ((1 - np.exp(-self.decay * (self.T_each_seq[seq_id] - time_stamp))) / self.decay)

            start_ind = ind
            start_seq_id, start_time_stamp = self.X_df.index[start_ind]
            # the influence on subsequent timestamp when the event occurs
            next_ind = start_ind
            while start_seq_id == seq_id:  # the influence only spread on the same sequence
                kap = self.kappa(start_time_stamp - time_stamp)
                if kap < 0.0001:
                    break

                X_kappa = self.X[ind] * kap
                sum_t_X_kappa[next_ind] += X_kappa  # record the influence
                next_ind += 1
                if next_ind >= len(self.X):
                    break
                start_seq_id, start_time_stamp = self.X_df.index[next_ind]
        return sum_t_X_kappa, decay_effect_integral_to_T

    # decay function
    def kappa(self, t):
        y = np.exp(-self.decay * t)
        return y

    # transfer event table from continuous time domain to the discrete time domain
    def get_event_table(self, event_table: pd.DataFrame):
        event_table = event_table.copy()
        event_table.columns = ['seq_id', 'time_stamp', 'event_type']
        if self.time_interval is not None:
            event_table['time_stamp'] = (event_table['time_stamp'] / self.time_interval).astype(
                'int') * self.time_interval

        event_table['times'] = np.zeros(len(event_table))
        event_table = event_table.groupby(['seq_id', 'time_stamp', 'event_type']).count().reset_index()

        event_ind = event_table['event_type'].astype('category')
        event_table['type_ind'] = event_ind.cat.codes
        event_names = event_ind.cat.categories

        max_time = event_table.groupby('seq_id').apply(lambda i: i['time_stamp'].max())
        event_table = pd.merge(event_table, pd.DataFrame(max_time, columns=['max_time_stamp']).reset_index())

        event_table.sort_values(['seq_id', 'time_stamp', 'type_ind'])

        return event_table, event_names

    def EM(self, edge_mat):
        """
        EM (Expectation-Maximization) module with using hill climb
        :param edge_mat:    Adjacency matrix
        :return:            (likelihood, alpha matrix, mu vector)
        """

        if not check_DAG(edge_mat):
            return __MIN__, np.zeros([len(self.event_names), len(self.event_names)]), np.zeros(
                len(self.event_names))

        # Initialize alpha and mu parameters
        alpha = self.random_state.uniform(0, 1, [len(self.event_names), len(self.event_names)])
        alpha = alpha * edge_mat
        mu = np.ones(len(self.event_names))
        L = 0

        # calculate the likelihood for each event type i
        for i in (range(len(self.event_names))):
            # Get the parent node set of event i (i.e., all events pointing to i
            Pa_i = tuple(np.where(edge_mat[:, i] == 1)[0])

            try:
                # Try to retrieve previously calculated results from cache
                Li = self.hist_likelihood[i][tuple(Pa_i)][0]
                mu[i] = self.hist_likelihood[i][tuple(Pa_i)][2]
                for j in Pa_i:
                    alpha[j, i] = self.hist_likelihood[i][tuple(Pa_i)][1][j]
                L += Li
            except Exception as e:
                # If not cached, optimize parameters iteratively using EM algorithm
                Li = __MIN__

                while 1:
                    # the first term of likelihood function
                    lambda_for_i = (self.sum_t_X_kappa * alpha[:, i]).sum(1) + mu[i]
                    # the second term of likelihood function
                    X_log_lambda = (self.X[:, i] * np.log(lambda_for_i)).sum()
                    lambda_i_sum = (((1 / self.decay) * self.X).sum(0) * alpha[:, i].T).sum() + mu[i] * self.T

                    # calculate the likelihood
                    new_Li = -lambda_i_sum + X_log_lambda

                    # Iteration termination condition
                    gain = new_Li - Li
                    if gain < 0.0085:
                        Li = new_Li
                        L += Li
                        Pa_i_alpha = dict()
                        for j in Pa_i:
                            Pa_i_alpha[j] = alpha[j, i]
                        self.hist_likelihood[i][tuple(Pa_i)] = (Li, Pa_i_alpha, mu[i])
                        break
                    Li = new_Li

                    # update mu
                    mu[i] = ((mu[i] / lambda_for_i) * self.X[:, i]).sum() / self.T
                    # update alpha
                    for j in Pa_i:
                        q_alpha = alpha[j, i] * self.sum_t_X_kappa[:, j] / lambda_for_i
                        upper = (q_alpha * self.X[:, i]).sum()

                        lower = self.decay_effect_integral_to_T.sum(0)[j] * self.time_interval
                        if lower == 0:
                            alpha[j, i] = 0
                            continue
                        alpha[j, i] = upper / lower

                i += 1
        # Adjust likelihood value according to specified penalty method (AIC or BIC
        if self.penalty == 'AIC':
            return L - (len(self.event_names) + edge_mat.sum()), alpha, mu
        if self.penalty == 'BIC':
            return L - (len(self.event_names) + edge_mat.sum()) * np.log(
                self.event_table['times'].sum()) * self.reg, alpha, mu


    def EM_not_HC(self, edge_mat):
        """
        EM (Expectation-Maximization) module without using hill climb
        :param edge_mat:    Adjacency matrix

        :return:            (likelihood, alpha matrix, mu vector)
        """

        # Initialize alpha and mu parameters
        alpha = self.random_state.uniform(0, 1, [len(self.event_names), len(self.event_names)])
        alpha = alpha * edge_mat
        mu = np.ones(len(self.event_names))
        L = 0

        # Perform parameter estimation for each event i
        for i in (range(len(self.event_names))):
            # Get the parent node set of event i (i.e., all events pointing to i)
            Pa_i = set(np.where(edge_mat[:, i] == 1)[0])

            try:
                # Try to retrieve previously calculated results from cache
                Li = self.hist_likelihood[i][tuple(Pa_i)][0]
                mu[i] = self.hist_likelihood[i][tuple(Pa_i)][2]
                for j in Pa_i:
                    alpha[j, i] = self.hist_likelihood[i][tuple(Pa_i)][1][j]
                L += Li
            except Exception as e:
                # If not cached, optimize parameters iteratively using EM algorithm
                Li = __MIN__

                while 1:
                    # the first term of likelihood function
                    lambda_for_i = (self.sum_t_X_kappa * alpha[:, i]).sum(1) + mu[i]
                    # the second term of likelihood function
                    X_log_lambda = (self.X[:, i] * np.log(lambda_for_i)).sum()
                    lambda_i_sum = (((1 / self.decay) * self.X).sum(0) * alpha[:, i].T).sum() + mu[i] * self.T

                    # Current likelihood value
                    new_Li = -lambda_i_sum + X_log_lambda
                    # Iteration termination condition
                    gain = new_Li - Li

                    # Check if termination condition is met
                    if gain <= 0.0085:
                        Li = new_Li
                        L += Li
                        # Cache current parameter results
                        Pa_i_alpha = dict()
                        for j in Pa_i:
                            Pa_i_alpha[j] = alpha[j, i]
                        self.hist_likelihood[i][tuple(Pa_i)] = (Li, Pa_i_alpha, mu[i])
                        break
                    Li = new_Li

                    # update mu
                    mu[i] = ((mu[i] / lambda_for_i) * self.X[:, i]).sum() / self.T
                    # update alpha
                    for j in Pa_i:
                        q_alpha = alpha[j, i] * self.sum_t_X_kappa[:, j] / lambda_for_i
                        upper = (q_alpha * self.X[:, i]).sum()

                        lower = self.decay_effect_integral_to_T.sum(0)[j] * self.time_interval
                        if lower == 0:
                            alpha[j, i] = 0
                            continue
                        alpha[j, i] = upper / lower

                i += 1

        # Adjust likelihood value according to specified penalty method (AIC or BIC)
        if self.penalty == 'AIC':
            return L - (len(self.event_names) + edge_mat.sum())* self.reg, alpha, mu
        if self.penalty == 'BIC':
            return L - (len(self.event_names) + edge_mat.sum()) * np.log(
                self.event_table['times'].sum()) * self.reg, alpha, mu

    # the searching module for new edges
    def one_step_change_iterator(self, edge_mat):
        return map(lambda e: self.one_step_change(edge_mat, e),
                   product(range(len(self.event_names)),
                           range(len(self.event_names)), range(3)))

    def one_step_change(self, edge_mat, e):
        j, i = e[0], e[1]
        if j == i:
            return edge_mat
        new_graph = edge_mat.copy()
        if e[2] == 0:
            new_graph[j, i] = 0
            new_graph[i, j] = 0
            return new_graph
        elif e[2] == 1:
            new_graph[j, i] = 1
            new_graph[i, j] = 0
            return new_graph
        else:
            new_graph[j, i] = 0
            new_graph[i, j] = 1
            return new_graph

    def Hill_Climb(self):

        # Initialize the adjacency matrix
        edge_mat = self.init_structure
        result = self.EM(edge_mat)

        L = result[0]
        while 1:
            stop_tag = True
            for new_edge_mat in tqdm(list(self.one_step_change_iterator(edge_mat)), mininterval=5):
                new_result = self.EM(new_edge_mat)
                new_L = new_result[0]
                # Termination condition: no adjacency matrix with higher likelihood appears
                if new_L > L:
                    result = new_result
                    L = new_L
                    # if there is a new edge can be added, then set the stop_tag=False and continue searching
                    stop_tag = False
                    edge_mat = new_edge_mat

            if stop_tag:
                return result

    def train_model(self,hill_climb=True):
        """
        Train the model and return the result
        :param hill_climb: use hill_climb or not
        :return:(likelihood,alpha,mu)
        """
        if hill_climb:
            res = self.Hill_Climb()
        else:
            res = self.EM_not_HC(np.ones([self.n, self.n]) - np.eye(self.n, self.n))
        return res


def SHP_exp(n, sample_size, out_degree_rate, mu_range_str, alpha_range_str, decay, penalty='BIC',
            NE_num=40, model_decay=0.35, seed=0,
            time_interval=5, hill_climb=True, reg=0.85):

    """
    Execute SHP model experiment, generate simulated data and perform parameter estimation

    :param n: Number of nodes
    :param sample_size: Sample size
    :param out_degree_rate: Average out-degree ratio of nodes
    :param mu_range_str: String representing the range of mu parameter values
    :param alpha_range_str: String representing the range of alpha parameter values
    :param decay: Decay parameter
    :param penalty: Type of penalty term, default is 'BIC'
    :param NE_num: Number of non-events, default is 40
    :param model_decay: Model decay parameter, default is 0.35
    :param seed: Random seed, default is 0
    :param time_interval: Time interval, default is 5
    :param hill_climb: Whether to use hill climbing algorithm, default is True
    :param reg: Regularization parameter, default is 0.85

    :return: Tuple containing:
        - likelihood (float): Likelihood value
        - fited_alpha (array): Fitted alpha parameters
        - fited_mu (array): Fitted mu parameters
        - real_edge_mat (array): Real adjacency matrix
        - real_alpha (array): Real alpha parameters
        - real_mu (array): Real mu parameters
    """


    # Generate simulated data, including event table, real adjacency matrix and parameters
    event_table, real_edge_mat, real_alpha, real_mu, events = generate_data(n=n, sample_size=sample_size,
                                                                            out_degree_rate=out_degree_rate,
                                                                            mu_range_str=mu_range_str,
                                                                            alpha_range_str=alpha_range_str,
                                                                            NE_num=NE_num, decay=decay,
                                                                            seed=seed)

    # Build model parameter dictionary
    param_dict = {
        "decay": model_decay,
        "reg": reg,
        "time_interval": time_interval,
        "penalty": penalty
    }
    # Initialize the SHP model instance
    self = SHP(event_table, **param_dict)

    # Select optimization algorithm for parameter estimation based on parameters
    res1=self.train()

    # Extract estimation results
    likelihood = res1[0]
    fited_alpha = res1[1]
    fited_mu = res1[2]
    return likelihood, fited_alpha, fited_mu, real_edge_mat, real_alpha, real_mu


if __name__=='__main__':
    from utils import get_performance

    likelihood, fited_alpha, fited_mu, real_edge_mat, real_alpha, real_mu = SHP_exp(n=20, sample_size=20000,
                                                                                    out_degree_rate=1.5,
                                                                                    mu_range_str="0.00005,0.0001",
                                                                                    alpha_range_str="0.5,0.7",
                                                                                    decay=5, model_decay=0.35, seed=0,
                                                                                    time_interval=5, penalty='BIC',
                                                                                    hill_climb=False, reg=0.85)
    res = get_performance(fited_alpha, real_edge_mat)
    print(res)
