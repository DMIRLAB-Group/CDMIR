from cdmir.effect.PBSCM_PGF.util import *
from tqdm import tqdm
from copy import deepcopy
from cdmir.effect.PBSCM_PGF.CCARankTest import CCARankTester


np.set_printoptions(linewidth=np.inf, precision=3)

class PBSCM_PGF:
    '''
    Python implementation of PB_SCM_PGF algorithm
    Reference:
    [1] Xiang Y, Qiao J, Liang Z, et al.On the identifiability of poisson branching structural causal model using probability generating function[J]. Advances in Neural Information Processing Systems, 2024, 37: 11664-11699.
    '''
    def __init__(self, data):
        '''
        Constructor of PBSCM_PGF model

        Parameters:
        ----------
        data: Input data matrix with shape (n_samples, n_features)
        '''

        self.dag = None
        self.skeleton = None
        self.data = data
        self.n, self.m = data.shape
        self.log_pgf = 0
        self.epgf = 0
        self.epgf_gradient = 0
        self.directed_edges = set()

    def get_causal_graph(self):
        '''
        Learn the skeleton and direction of the causal graph using PGF model.

        Returns:
        ----------
        dag: causal graph of the data
        '''
        # learn the skeleton of the causal graph using PGF model.
        self.learn_skeleton()
        # learn the direction of the triangle structure using PGF model.
        self.learn_direction_for_triangle()
        # learn the direction of the chain structure using PGF model.
        self.learn_direction_for_chain()

        return self.dag

    def learn_skeleton(self):
        '''
        learn the skeleton of the causal graph using PGF model.
        '''

        self.skeleton = np.zeros([self.n, self.n])
        pbar = tqdm(total=self.n * (self.n - 1) / 2)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # test the independence between i and j using bootstrap test.
                res = test_adjacent_rank_bootstrap_gaussian(self.data, i, j, batch=20, alpha=0.05)
                if res[0]:
                    # if i and j are independent, add a undirected edge between i and j
                    self.skeleton[i][j] = 1
                pbar.update(1)

        self.skeleton = np.maximum(self.skeleton, self.skeleton.transpose())
        self.dag = deepcopy(self.skeleton)
        return self.skeleton

    def learn_direction_for_triangle(self):
        '''
        learn the direction of the triangle structure using PGF model.
        '''

        # need to learn the skeleton first.
        if self.skeleton is None:
            raise ValueError("Please learn the skeleton first")
        else:
            self.dag = deepcopy(self.skeleton)

        # find all the triangles in the undirected graph.
        triangles = find_triangle_in_undirected_graph(self.skeleton)

        for (i, j, k) in triangles:
            # orientate the directed edge in the triangle structure.
            self.orientate_in_triangle(i=i, j=j, k=k)

    
    def learn_direction_for_chain(self):
        '''
        After orienting the triangles, orient the remaining undirected chain structures
        learn the direction of the chain structure using PGF model.
        '''

        while True:
            num_of_edge = self.dag.sum()  # Track if any edge is updated

            undirected_edges = []
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if self.dag[i, j] == 1 and self.dag[j, i] == 1:
                        undirected_edges.append((i, j))

            for edge in undirected_edges:
                a, b = edge
                # Check if there is a directed edge pointing to a or b, Additionally, ensure no triangle is formed
                to_a_list = [i for i in range(self.n) if (i != b
                                                          and (self.dag[i, a] == 1 and self.dag[a, i] != 1)
                                                          and (self.dag[b, i] == 0 and self.dag[i, b] == 0))]
                to_b_list = [i for i in range(self.n) if (i != a
                                                          and (self.dag[i, b] == 1 and self.dag[b, i] != 1)
                                                          and (self.dag[a, i] == 0 and self.dag[i, a] == 0))]

                # Prioritize directed edges, then proceed with orientation
                if to_a_list:
                    self.orientate_in_chain_rank(i=to_a_list[0], j=a, k=b, i_to_j=True)
                elif to_b_list:
                    self.orientate_in_chain_rank(i=to_b_list[0], j=b, k=a, i_to_j=True)

                # If no directed edge is found, attempt to match any adjacent undirected edge node
                else:
                    adj_with_a_list = [i for i in range(self.n) if (self.dag[i, a] == 1 and self.dag[a, i] == 1
                                                                    and (self.dag[b, i] == 0 and self.dag[i, b] == 0)
                                                                    and i != b
                                                                    )]
                    adj_with_b_list = [i for i in range(self.n) if (self.dag[i, b] == 1 and self.dag[b, i] == 1
                                                                    and (self.dag[a, i] == 0 and self.dag[i, a] == 0)
                                                                    and i != a
                                                                    )]
                    tag = False
                    # Iterate through nodes adjacent to a, exit if successfully oriented
                    for node in adj_with_a_list:
                        # node-a-b
                        if self.orientate_in_chain_rank(i=node, j=a, k=b, i_to_j=False):
                            tag = True
                            break
                    if not tag:
                        # Iterate through nodes adjacent to b, exit if successfully oriented
                        for node in adj_with_b_list:
                            # node-b-a
                            if self.orientate_in_chain_rank(i=node, j=b, k=a, i_to_j=False):
                                break

            if num_of_edge == self.dag.sum():  # No new orientations in this iteration
                break


    def orientate_in_chain_rank(self, i, j, k, i_to_j):
        """
        Input three adjacent node: i-j-k.
        If i_to_j is True, i and j are already directed; otherwise, both edges are undirected.

        Parameters
        ----------
        i: the index of node i.
        j: the index of node j.
        k: the index of node k.
        i_to_j: i->j is already directed or not

        Returns
        -------
        tag_for_orientated: Has the causal direction of the i-j-k triplet been successfully determined
        """
        tag_for_orientated = True
        z_list = [0.05] * self.n
        z_list[i], z_list[j], z_list[k] = 1, 1, 1

        order_list = np.zeros(self.n).astype(int)

        # Calculate the basic PGF
        A = get_epgf(self.data, z_list)

        # Calulate B
        order_list[i], order_list[k] = 1, 0
        B = diff_get_epgf_gradient(self.data, z_list=z_list, order_list=order_list)

        # Calulate C
        order_list[i], order_list[k] = 0, 1
        C = diff_get_epgf_gradient(self.data, z_list=z_list, order_list=order_list)

        # Calulate D
        order_list[i], order_list[k] = 1, 1
        D = diff_get_epgf_gradient(self.data, z_list=z_list, order_list=order_list)

        # construct gradient matrix
        mat = np.array([[A, B],
                        [C, D]])

        from numpy.linalg import norm
        mat /= norm(mat, axis=1, keepdims=True)

        # Perform CCA rank test
        rank_tester = CCARankTester(data=self.data.T, alpha=0.05)
        res, stat_list = rank_tester.test_my(matrix=mat)

        # res = True means rank is not 1, so False implies rank equals 1, that is collider structure: i->j<-k
        if not res:
            if i_to_j:  # i->j is already directed
                self.dag[j][k] = 0
            else:
                self.dag[j][i] = 0  # i->j
                self.dag[j][k] = 0  # k->j

        # If it is a chain or fork structure, it can be oriented based on the direction of one of the edges
        elif i_to_j:
            self.dag[k][j] = 0  # i->j->k

        # Otherwise, the direction cannot be determined. It could be any of i->j->k, i<-j-<k, or i<-j->k (equivalence class)
        else:
            tag_for_orientated = False
            pass

        return tag_for_orientated


    def orientate_in_triangle(self, i, j, k):
        '''
        orientate the direction of triangle i-j-k.
        
        Parameters
        ----------
        i: the index of node i.
        j: the index of node i.
        k: the index of node i.
        '''

        z_list = [0.6] * self.n
        z_list[i], z_list[j], z_list[k] = 1, 1, 1

        edge_list = np.array([[i, j], [j, k], [i, k]])
        res_list = []

        # conduct d irectional testing on three potential edges
        for edge in edge_list:
            # calculate gradient products of gradients in two direction
            order_list = np.zeros(self.n).astype(int)
            # direction 1->2
            order_list[edge] = (1, 2)
            d3_PGF = diff_get_epgf_gradient_np(self.data, z_list, order_list=order_list)
            order_list[edge] = (1, 0)
            d1_PGF = diff_get_epgf_gradient_np(self.data, z_list, order_list=order_list)
            res1 = d3_PGF * d1_PGF

            # direction 2->1
            order_list[edge] = (2, 1)
            d3_PGF = diff_get_epgf_gradient_np(self.data, z_list, order_list=order_list)
            order_list[edge] = (0, 1)
            d1_PGF = diff_get_epgf_gradient_np(self.data, z_list, order_list=order_list)
            res2 = d3_PGF * d1_PGF

            res_list.append([res1, res2])

        orientate_res = []
        for ind in range(3):
            edge = edge_list[ind]
            res1, res2 = res_list[ind][0], res_list[ind][1]
            # compare gradient products to determine direction
            if res1 > res2:
                orientate_res.append([edge[0], edge[1]])
            else:
                orientate_res.append([edge[1], edge[0]])

        check_res = check_orientate(orientate_res)
        orientate_res = np.array(orientate_res)
        if check_res[0]:
            for edge in orientate_res[check_res[1]]:
                self.dag[edge[1]][edge[0]] = 0
                self.dag[edge[0]][edge[1]] = 1
                self.directed_edges.add(tuple(sorted(edge)))