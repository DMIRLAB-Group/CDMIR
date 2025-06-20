from cdmir.discovery.PBSCM.util import *
from tqdm import tqdm
from copy import deepcopy


class PB_SCM(object):
    '''
    Python implementation of PB_SCM algorithm.
    Reference:
    -
    [1] Qiao J, Xiang Y, Chen Z, et al. Causal discovery from poisson branching structural causal model using high-order cumulant with path analysis[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(18): 20524-20531.
    '''
    @staticmethod
    def determine_pair_direction(x, y, max_order, alpha=0.04, threshold=0):
        '''
        determine the direction of the causal relationship between x and y using the path cumulation summation test

            Parameters:
            ----------
            x: random variable x
            y: random variable y
            max_order: The max order of Lambda_k.
            alpha: confidence level
            threshold: threshold when bootstrap test fail.

            Returns:
            ----------
            "x->y": x causes y
            "y->x": y causes x
            '''
            
        n_samples = int(0.75 * len(x))
        confidence_iv = path_cum_summation_test(x=x, y=y, alpha=alpha,
                                                n_samples=n_samples, batch=50, max_order=max_order)
        reverse_confidence_iv = path_cum_summation_test(x=y, y=x, alpha=alpha,
                                                        n_samples=n_samples, batch=50, max_order=max_order)

        path_cum_i_to_j_list = get_path_cum_summation(x, y, max_order=max_order + 1)
        path_cum_j_to_i_list = get_path_cum_summation(y, x, max_order=max_order + 1)

        i_to_j_order = 1
        j_to_i_order = 1

        for order in range(1, max_order):
            if confidence_iv[order][0] < path_cum_i_to_j_list[order] < confidence_iv[order][1]:
                break
            else:
                i_to_j_order = order + 1

        for order in range(1, max_order):
            if reverse_confidence_iv[order][0] < path_cum_j_to_i_list[order] < reverse_confidence_iv[order][1]:
                break
            else:
                j_to_i_order = order + 1

        if i_to_j_order == j_to_i_order and i_to_j_order != 1:
            if path_cum_i_to_j_list[i_to_j_order] - path_cum_j_to_i_list[i_to_j_order] > threshold:
                return "x->y"
            else:
                return "y->x"

        if i_to_j_order > j_to_i_order:
            return "x->y"
        elif j_to_i_order > i_to_j_order:
            return "y->x"
        else:
            return "x-y"
    def __init__(self, data, seed=1):
        '''
        Construct the PB_SCM model

        Parameters
        ----------
        data: input data (n_sample,n_features)
        seed: random seed (default 1)
        '''
        self.data=data.T
        self.n = len(self.data)
        self.sample_size = len(self.data[0])
        self.skeleton_mat = None
        self.coeff_list = None
        self.record = [{} for i in range(self.n)]  # record the likelihood

        if seed is not None:
            self.rand_state = np.random.RandomState(seed)
            np.random.seed(seed)
        else:
            self.rand_state = np.random.RandomState()

    def dfs(self, graph, x, vis):
        '''
        Depth-first search in order to check if there is a circle in the graph

        Parameters:
        ----------
        graph: adjacency matrix (n,n)
        x: current node
        vis: visited node (n,)

        Returns:
        ----------
        ret: True if there is a circle, False otherwise
        '''
        if vis[x]:
            return True
        vis[x] = 1
        ret = False
        for i in range(self.n):
            if graph[x][i] == 1:
                ret = ret or self.dfs(graph, i, vis)
        vis[x] = 0
        return ret

    def check_circle(self, graph):
        '''
        Check whether there is a circle in the graph

        Parameters:
        ----------
        graph: adjacency matrix (n,n)

        Returns:
        ----------
        ret: True if there is a circle, False otherwise
        '''
        n = self.n
        ret = False
        vis = np.zeros(n)
        for i in range(n):
            ret = ret or self.dfs(graph, i, vis)
        return ret

    def one_step_change_iterator(self, edge_mat):
        '''
        Generate all possible one-step changes of the graph

        Parameters:
        ----------
        edge_mat: adjacency matrix (n,n)

        Returns:
        ----------
        all possible one-step changes of the graph
        '''
        return map(lambda e: one_step_change(edge_mat, e),
                   product(range(self.n), range(self.n)))

    def get_parent(self, graph, x):
        '''
        Get the list of parents of node x

        Parameters:
        ----------
        graph: adjacency matrix (n,n)
        x: current node

        Returns:
        ----------
        parent_list: list of parents of node x
        '''
        parent_list = []
        for i in range(self.n):
            if graph[i, x]:
                parent_list.append(i)
            else:
                continue
        return np.array(parent_list)

    # estimate the coefficient
    def get_coefficient(self, parent_index, target_index):
        '''
        Estimate the coefficient of the target node given the parents of the target node

        Parameters:
        ----------
        parent_index: list of parents of the target node
        target_index: target node

        Returns:
        ----------
        coeff: coefficient of the target node given the parents of the target node
        '''

        coeff = np.zeros((self.n+1))
        target_data = self.data[target_index]

        # if target node is the root node,calculate the λ_noise=E(X_target)
        if len(parent_index) == 0:
            lam_for_noise = np.mean(target_data)
            coeff[-1] = lam_for_noise
            return coeff

        # if target node has parents,calculate the coefficients of the parents of the target node
        parent_data_list = self.data[parent_index]
        n = len(parent_data_list)

        # if the number of parents is 1
        if n == 1:
            parent_coeff = np.cov(parent_data_list[0], target_data)[0, 1] / np.mean(parent_data_list[0])
            parent_coeff = np.array([parent_coeff])
        else:
            # if the number of parents is more than 1
            cov_mat = np.cov(parent_data_list)
            cov_vec = np.zeros([n, 1])
            for i in range(n):
                cov_vec[i] = np.cov([parent_data_list[i], target_data])[0, 1]
            parent_coeff = np.dot(np.linalg.inv(cov_mat), cov_vec).ravel()

        # calculate the λ_noise=E(X_target)-E(X_target|X_parent)
        E_of_preant = 0
        for i in range(n):
            E_of_preant += parent_coeff[i] * np.mean(parent_data_list[i])
        lam_for_noise = np.mean(target_data) - E_of_preant
        coeff[parent_index] = parent_coeff
        coeff[-1] = lam_for_noise
        return coeff

    def get_node_likelihood(self, coeff_list, parent_list, target_node):
        '''
        Calculate the likelihood of the target node given the parents of the target node

        Parameters:
        ----------
        coeff_list: list of coefficients of the target node given the parents of the target node
        parent_list: list of parents of the target node
        target_node: target node

        Returns:
        ----------
        likelihood: likelihood of the target node given the parents of the target node
        '''
        not_parent = np.delete(np.array([i for i in range(self.n)]), list(parent_list))
        samples = deepcopy(self.data)
        target_sample = deepcopy(samples[target_node, :])
        samples[not_parent, :] = 0
        samples = np.concatenate([samples, [target_sample]]).T

        sample_unique, unique_count = np.unique(samples, axis=0, return_counts=True)

        likelihood = 0.
        # calculate the likelihood of each sample
        for i in range(len(sample_unique)):
            Li_for_one_sample = get_sample_likelihood(coeff=coeff_list, parent_list=parent_list, sample=sample_unique[i])
            if Li_for_one_sample <= 0:
                likelihood += 0
            else:
                likelihood += np.log(Li_for_one_sample) * unique_count[i]
        return likelihood

    def get_total_likelihood(self, graph):
        '''
        Calculate the total likelihood of the graph

        Parameters:
        ----------
        graph: adjacency matrix (n,n)
        
        '''
        likelihood = 0
        n, m = self.n, self.sample_size
        coeff = np.zeros((n, n + 1))  # coeff set
        if self.check_circle(graph):  # if not DAG
            return -np.inf, coeff

        for i in range(n):
            # get the parents of vertex i
            parent_list = self.get_parent(graph, i)
            parent_list_str = np.array2string(parent_list)
            if parent_list_str in self.record[i]:
                likelihood += self.record[i][parent_list_str][0]
                coeff[i] = self.record[i][parent_list_str][1]
                continue
            # Estimating coefficients
            coeff[i] = self.get_coefficient(parent_list, i)

            if len(parent_list) != 0 and \
                    ((coeff[i][parent_list] < 0).any() or
                     (coeff[i][parent_list] > 1).any() or
                     (coeff[i][-1] < 0).any()
                    ):
                likelihood = -np.inf
                return likelihood, coeff

            # calculate the likelihood of nore i
            L_for_node_i = self.get_node_likelihood(coeff[i], parent_list, i)

            # record the likelihood
            self.record[i][parent_list_str] = [L_for_node_i, coeff[i]]

            likelihood += L_for_node_i
        # BIC penalty
        likelihood -= np.count_nonzero(graph) * np.log(m) / 2

        return likelihood, coeff

    def Hill_Climb_search(self):
        '''
        Hill Climb search for the best graph structure

        Returns:
        ----------
        best_edge_mat: adjacency matrix of the best graph structure
        '''
        n = self.n
        L = -np.inf
        best_edge_mat = np.zeros([n, n])
        step_mat = np.zeros([n, n])
        while True:
            stop_tag = True
            for new_edge_mat in tqdm(list(self.one_step_change_iterator(step_mat))):

                new_L, coeff = self.get_total_likelihood(new_edge_mat)

                if new_L - L > 0:
                    L = new_L
                    stop_tag = False
                    best_edge_mat = new_edge_mat
            step_mat = best_edge_mat

            if stop_tag:
                print("HC_best_likelihood:", L)
                self.skeleton_mat = best_edge_mat
                return get_symmetric_matrix(self.skeleton_mat)



    def learning_causal_direction(self,skeleton_mat, alpha=0.04, max_order=4, threshold=0):
        '''
        learning causal direction of a causal graph using the path cumulation summation test

        Parameters:
        ----------
        skeleton_mat: Skeleton of causal graph.
        alpha: confidence level
        max_order: The max order of Lambda_k.
        threshold: threshold when bootstrap test fail.

        Returns:
        ----------
        Causal graph
        '''

        DAG = copy.copy(skeleton_mat)
        pbar = tqdm(total=skeleton_mat.sum()/2)
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                if skeleton_mat[i, j] == 1:
                    direction = self.determine_pair_direction(x=self.data[i], y=self.data[j],
                                                        threshold=threshold,
                                                        max_order=max_order,
                                                        alpha=alpha)
                    pbar.update(1)

                    if direction == "x->y":
                        DAG[j, i] = 0
                    elif direction == "y->x":
                        DAG[i, j] = 0
                    else:
                        continue

        return np.array(DAG)

    def get_causal_graph(self, alpha=0.04, max_order=4, threshold=0):
        '''
        Get the causal graph of the data

        Returns:
        ----------
        causal_graph: causal graph of the data
        '''
        
        # learn the skeleton of the graph
        skeleton=self.Hill_Climb_search()
        # learn the causal direction of the graph
        causal_graph = self.learning_causal_direction(skeleton,alpha,max_order,threshold)
        return causal_graph