from causaldmir.discovery.point_process.SHP import SHP
from causaldmir.datasets.simlulators import HawkesSimulator
import numpy as np
from causaldmir.utils.metrics.graph_evaluation import get_performance
def SHP_exp(n, sample_size, out_degree_rate, mu_range_str, alpha_range_str, decay, penalty='BIC',
            NE_num=40, model_decay=0.35, seed=0,
            time_interval=5, hill_climb=True, reg=0.85):
    event_table, real_edge_mat, real_alpha, real_mu, events = HawkesSimulator.generate_data(n=n, sample_size=sample_size,
                                                                            out_degree_rate=out_degree_rate,
                                                                            mu_range_str=mu_range_str,
                                                                            alpha_range_str=alpha_range_str,
                                                                            NE_num=NE_num, decay=decay,
                                                                            seed=seed)

    param_dict = {
        "decay": model_decay,
        "reg": reg,
        "time_interval": time_interval,
        "penalty": penalty
    }

    self = SHP(event_table, **param_dict)
    if hill_climb:
        res1 = self.Hill_Climb()
    else:
        res1 = self.EM_not_HC(np.ones([self.n, self.n]) - np.eye(self.n, self.n))

    likelihood = res1[0]
    fited_alpha = res1[1]
    fited_mu = res1[2]

    return likelihood, fited_alpha, fited_mu, real_edge_mat, real_alpha, real_mu


if __name__=='__main__':
    likelihood, fited_alpha, fited_mu, real_edge_mat, real_alpha, real_mu = SHP_exp(n=20, sample_size=20000,
                                                                                    out_degree_rate=1.5,
                                                                                    mu_range_str="0.00005,0.0001",
                                                                                    alpha_range_str="0.5,0.7",
                                                                                    decay=5, model_decay=0.35, seed=0,
                                                                                    time_interval=5, penalty='BIC',
                                                                                    hill_climb=True, reg=0.85)
    res = get_performance(fited_alpha, real_edge_mat)
    print(res)