import argparse
import torch
import pickle as pkl
import torch.nn as nn
import utils as utils
import numpy as np
from modules import GCN, NN, Predictor, Discriminator, Density_Estimator, Discriminator_simplified

class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis).cuda()
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class TR(nn.Module):

    def __init__(self, degree, knots):
        """
                This class is to initialize the perturbation regression model and adjust the bias correction term
                :param degree: order
                :param knots: node list
        """
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis
        self.weight = torch.nn.Parameter(torch.rand(self.d, device='cuda'), requires_grad=True)

    def forward(self, t):
        """
                This function is used for forward propagation,
                adjusting the base model prediction through target learning to eliminate the influence of model errors
                :param t: Input tensor
                :return: disturbance parameter
        """
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        """Initialize weights to avoid excessive initial perturbations"""
        # self.weight.data.normal_(0, 0.1)
        self.weight.data.zero_()


class TargetedModel_DoubleBSpline(nn.Module):

    def __init__(self, Xshape, hidden, dropout, num_grid=None, init_weight=True, tr_knots=0.25, cfg_density=None):
        """
            This class initializes the TNet model
            :param Xshape: Input feature dimension
            :param hidden: hidden layer dimension
            :param dropout: Dropout probability
            :param num_grid: Number of B-spline grids
            :param initw_weight: whether to initialize weights
            :param tr_knots: perturbed node density
        """
        super(TargetedModel_DoubleBSpline, self).__init__()
        if num_grid is None:
            num_grid = 20

        """
          1. Using GCN model to handle individual and neighbor features in network interference
        """
        self.encoder = GCN(nfeat=Xshape, nclass=hidden, dropout=dropout)

        """
         2.Integrating individual embeddings with original features to capture the interaction effects of X and X_N
        """
        self.X_XN = Predictor(input_size=hidden + Xshape, hidden_size1=hidden, hidden_size2=hidden, output_size=int(hidden/2))
        """
        3. Calculate the potential outcome Q (t, z | x, x_N)
        """
        self.Q1 = Predictor(input_size=int(hidden/2) + 1, hidden_size1=int(hidden*2), hidden_size2=hidden, output_size=1)
        self.Q0 = Predictor(input_size=int(hidden/2) + 1, hidden_size1=int(hidden*2), hidden_size2=hidden, output_size=1)
        """
        4.Calculate propensity score model through joint feature calculation (estimate p (t | x, x_N))
        """
        self.g_T = Discriminator_simplified(input_size=int(hidden/2), hidden_size1=hidden, output_size=1)
        """
         5. Calculate neighbor exposure model through joint feature calculation (estimate p (z | x, x_N))
        """
        self.g_Z = Density_Estimator(input_size=int(hidden/2), num_grid=num_grid)

        """
         6. Disturbance term model (corrected base model bias)
        """
        tr_knots = list(np.arange(tr_knots, 1, tr_knots))
        tr_degree = 2
        self.tr_reg_t1 = TR(tr_degree, tr_knots)
        self.tr_reg_t0 = TR(tr_degree, tr_knots)


        if init_weight:
            self.encoder._initialize_weights()
            self.X_XN._initialize_weights()
            self.Q1._initialize_weights()
            self.Q0._initialize_weights()
            self.g_Z._initialize_weights()
            self.g_T._initialize_weights()
            self.tr_reg_t1._initialize_weights()
            self.tr_reg_t0._initialize_weights()

    def parameter_base(self):
        """
            This function is used for optimizers (such as Adam) to update the core component parameters responsible for feature encoding, propensity score modeling, and result prediction in the model.
            These modules are the foundation of causal inference models and directly affect the accuracy of feature representation learning, potential outcome prediction, and propensity score estimation.

            Returns:
                List [Parameter]: A list of basic module parameters, including trainable parameters for the following submodules:
                Encoder: Node feature encoder (extracts high-order representations of input features, such as GNN layers)
                X_XN: Feature interaction module (modeling the interaction relationship between the node's own features and neighboring features)
                Q1: Potential outcome predictor for processing group (predicting the potential outcome Y ₁ when processing group T=1)
                Q0: Control group potential outcome predictor (predicting the potential outcome Y ₀ of the control group when T=0)
                GT: Treatment propensity score model (estimating the probability of nodes receiving treatment P (T=1 | X))
                GZ: Peer Effect Propensity Score Model (Estimating the Probability of Peer Effect Impact on Nodes)
              """
        return list(self.encoder.parameters()) +\
            list(self.X_XN.parameters()) +\
            list(self.Q1.parameters())+list(self.Q0.parameters())+\
            list(self.g_T.parameters())+\
            list(self.g_Z.parameters())

    def parameter_trageted(self):
        """
            The adjustment module parameters used for optimizer to update causal effect estimates are mainly responsible for bias calibration and robustness optimization of causal effects,
            Usually associated with bi robust estimation or regularization of processing effects.

            Returns:
                List [Parameter]: A list of target module parameters, including:
                    Tr_cd_t0: Control group (T=0) regularization term adjuster (calibrating the causal effect estimation bias of the control group)
-                   Tr_ded_t1: Processing group (T=1) regularization term adjuster (calibrating the causal effect estimation bias of the processing group)
                """
        return list(self.tr_reg_t0.parameters()) + list(self.tr_reg_t1.parameters())

    def tr_reg(self, T, neighborAverageT):
        """
            Dynamically generate regularization coefficients based on the processing state (T) of the node and the neighborhood average processing value (neighborAverageT),
            Used to constrain the bias of causal effect estimation and enhance the robustness of the model in complex causal relationships.
            :param T (Tensor): Processing variables (binary value, shape [N], N is the number of nodes; 1 indicates acceptance of processing, 0 indicates non acceptance of processing)
            :param neighborAverageT (Tensor): Neighborhood average processing value (the average processing value of each adjacent node, shape [N],
            Reflecting the processing intensity of the neighborhood where the node is located

            Returns:
                Tensor: dynamically adjusted regularization term (shape [N], each node corresponds to a regularization coefficient related to its processing state and neighborhood)
        """
        # Calculate the regularization coefficients for the processing group (T=1) and the control group (T=0)
        # 处理组正则化系数（由tr_reg_t1网络生成）
        tr_reg_t1 = self.tr_reg_t1(neighborAverageT)
        # 对照组正则化系数（由tr_reg_t0网络生成）
        tr_reg_t0 = self.tr_reg_t0(neighborAverageT)
        regur = torch.where(T==1, tr_reg_t1, tr_reg_t0)
        return regur



    def forward(self, A, X, T, Z=None):

        """
              This function is used to integrate network feature encoding, propensity score estimation,
              neighbor exposure modeling, result prediction, and perturbation term calculation,
              providing key intermediate quantities for dual robust causal effect estimation under network interference.
              :param A: torch.tensor, Adjacency matrix（shape：batch_size×batch_size）
              :param X: torch.tensor, Individual characteristics（shape：batch_size×Xshape）
              :param T: torch.tensor, Process variables（shape：batch_size）
              :param Z: torch.tensor, Neighbor exposure variables（Optional, shape：batch_size）
              :return: (g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT)
        """

        """
        Feature encoding: Integrating individual and neighbor features
        """
        embeddings = self.encoder(X, A)  # X_i,X_N
        embeddings = self.X_XN(torch.cat((embeddings,X), dim=1))
        """Tendency score estimation：p(t|x,x_N)p(t∣x,X_N)"""
        g_T_hat = self.g_T(embeddings)  # X_i,X_N -> T_i
        """Neighbor Exposure Calculation:Z=agg(T_N )"""
        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)  # treated_neighbors / all_neighbors
        else:
            neighborAverageT = Z

        """Estimation of neighbor exposure probability:p(z∣x,x_n )"""
        g_Z_hat = self.g_Z(embeddings, neighborAverageT)  # X_i,X_N -> Z
        g_Z_hat = g_Z_hat.unsqueeze(1)

        """result prediction Q(t,z∣x,x_n)"""
        embed_avgT = torch.cat((embeddings, neighborAverageT.reshape(-1, 1)), 1)

        Q_hat = T.reshape(-1, 1) * self.Q1(embed_avgT) + (1-T.reshape(-1, 1)) * self.Q0(embed_avgT)

        epsilon = self.tr_reg(T, neighborAverageT)  # epsilon(T,Z)


        return g_T_hat, g_Z_hat, Q_hat, epsilon, embeddings, neighborAverageT

    def infer_potential_outcome(self, A, X, T, Z=None):
        """
                This function uses dual robust correction to output estimated potential outcomes of individuals under network interference
                param A: Adjacency Matrix，Representing the connectivity relationships between nodes (individuals) in a network。
                param X: Individual Features，Contains covariate information for each node.
                param T: Treatment，Indicate whether each node accepts processing。
                param Z: Neighbor Exposure，Representing the proportion of neighbors receiving processing for each node
                :return: Revised Potential Results（Q_hat + ε/(p(t|x,x_N)p(z|x,x_N))）
        """

        """ Forward propagation to obtain base model predictions and perturbation terms"""
        embeddings = self.encoder(X, A)  # X_i,X_N
        embeddings = self.X_XN(torch.cat((embeddings,X), dim=1))
        # Propensity score p(t|x,x_N)（batch_size）
        g_T_hat = self.g_T(embeddings)  # X_i,X_N -> T_i
        g_T_hat = g_T_hat.squeeze(1)

        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)  # treated_neighbors / all_neighbors
        else:
            neighborAverageT = Z

        # Neighbor exposure probability p(z|x,x_N)（batch_size）
        g_Z_hat = self.g_Z(embeddings, neighborAverageT)  # X_i,X_N -> Z



        embed_avgT = torch.cat((embeddings, neighborAverageT.reshape(-1, 1)), 1)

        Q_hat = T.reshape(-1, 1) * self.Q1(embed_avgT) + (1-T.reshape(-1, 1)) * self.Q0(embed_avgT)

        epsilon = self.tr_reg(T, neighborAverageT)  # epsilon(T,Z)
        # epsilon = epsilon.squeeze(1)

        # Double robust correction（Q_hat + ε/(p(t|x,x_N)p(z|x,x_N))）
        return Q_hat.reshape(-1) + (epsilon.reshape(-1) * 1/(g_Z_hat.reshape(-1)*g_T_hat.reshape(-1) + 1e-6))

