import pandas as pd
from itertools import product
import numpy as np
from tick.hawkes import SimuHawkes, HawkesKernelExp
import argparse
import os

print(os.getcwd())


def check_and_create(path: str):
    if os.path.exists(path):
        return
    os.mkdir(path)


parser = argparse.ArgumentParser(description="data")
parser.add_argument('-sd', '--save_dir', type=str, help='save dir', default='./data/Hawkes_data_from_tick')
parser.add_argument('-et', '--exp_tag', type=int, help='data type', default=0)


def get_artificial_data(mu_range, alpha_range, n, sample_size=30000, out_degree_rate=1.5, NE_num=40, decay=0.1,
                        time_interval=None, seed=None):
    print(
        f'n={n},mu_range={mu_range},alpha_range={alpha_range},edge_num={round(out_degree_rate * n)},sample_size={sample_size}')
    rand_state = np.random.RandomState(seed=seed)

    edge_mat = np.zeros([n, n])
    edge_select = list(filter(lambda i: i[0] < i[1], product(range(n), range(n))))
    rand_state.shuffle(edge_select)
    for edge_ind in edge_select[:round(out_degree_rate * n)]:
        edge_mat[edge_ind] = 1
    mu = rand_state.uniform(*mu_range, n)

    alpha = rand_state.uniform(*alpha_range, [n, n])
    alpha = edge_mat * alpha

    hawkes = SimuHawkes(baseline=mu, max_jumps=sample_size / NE_num, verbose=False, seed=seed)
    for i in range(n):
        for j in range(n):
            if (alpha[i, j] == 0):
                continue
            hawkes.set_kernel(j, i, HawkesKernelExp(alpha[i, j], decay))

    event_dict = dict()
    for node in range(NE_num):
        hawkes.reset()
        hawkes.simulate()
        event_dict[node] = hawkes.timestamps

    event_list = []


    for node in event_dict:
        for event_name in range(n):
            for timestamp in event_dict[node][event_name]:
                event_list.append([node, timestamp, event_name])
    columns = ['seq_id', 'time_stamp', 'event_type']
    event_table = pd.DataFrame(event_list, columns=columns)

    if time_interval is not None:
        event_table['time_stamp'] = (event_table['time_stamp'] / time_interval).astype('int') * time_interval

    events = [[event_table[(event_table['event_type'] == i) & (event_table['seq_id'] == j)][
                   'time_stamp'].values.astype('float') for i in
               np.sort((event_table['event_type']).unique())] for j in (event_table['seq_id']).unique()]
    return event_table, edge_mat, alpha, mu, events


def INSEM_data(sample_size=10000, lambda_x=1, theta=0.5, lambda_e=1, seed=None):
    rand_state = np.random.RandomState(seed=seed)
    X = rand_state.poisson(lambda_x, sample_size)
    Y = np.zeros(sample_size, dtype='int')

    def operator(X):
        return sum(rand_state.binomial(1, theta, X))

    for i in range(sample_size):
        Y[i] = operator(X[i])
    Y = Y + rand_state.poisson(lambda_e, sample_size)
    t = 0
    df = pd.DataFrame()
    for i, n in enumerate(X):
        term = []
        for j in range(n):
            term.append([0, t, 0])
        t = t + 1
        df = pd.concat([df, pd.DataFrame(term)])
    t = 0
    for i, n in enumerate(Y):
        term = []
        for j in range(n):
            term.append([0, t, 1])
        t = t + 1
        df = pd.concat([df, pd.DataFrame(term)])
    df.columns = ['seq_id', 'time_stamp', 'event_type']
    return df


def generate_data(n, mu_range_str, alpha_range_str, sample_size, out_degree_rate, NE_num, decay, seed=None):
    alpha_range = tuple([float(i) for i in alpha_range_str.split(',')])
    mu_range = tuple([float(i) for i in mu_range_str.split(',')])
    event_table, edge_mat, alpha, mu, events = get_artificial_data(mu_range, alpha_range, n, sample_size=sample_size,
                                                                   out_degree_rate=out_degree_rate, NE_num=NE_num,
                                                                   decay=decay, seed=seed)

    return event_table, edge_mat, alpha, mu, events


