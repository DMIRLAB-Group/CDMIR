import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    """
       Simple graph convolutional layer (GCN layer) for graph neural networks.
       
       Typically used as a building block in graph neural networks for tasks like node classification, 
       link prediction, and graph representation learning.
       
       :param in_features (int): Dimension of input node features
       :param out_features (int): Dimension of output node features after convolution
       :param weight (torch.nn.Parameter): Learnable weight matrix of shape (in_features, out_features)
       :param bias (torch.nn.Parameter or None): Optional learnable bias vector of shape (out_features,)
       """

    def __init__(self, in_features, out_features, bias=True):
        """
               Initialize graph convolutional layer

               :param in_features (int): Number of input features per node
               :param out_features (int): Number of output features per node after convolution
               :param bias (bool, optional): Whether to include a learnable bias term. Defaults to True.
        """

        # Call parent Module constructor
        super(GraphConvolution, self).__init__()
        # Store dimension parameters
        self.in_features = in_features
        self.out_features = out_features
        # Initialize learnable weight matrix
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.register_parameter("weight", self.weight)
        # Initialize optional bias term
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.register_parameter('bias', self.bias)
        else:
            self.register_parameter('bias', None)
        # Initialize parameters with proper initialization scheme
        self.reset_parameters()

    def reset_parameters(self):
        """
                Initialize learnable parameters using uniform distribution

                Follows the initialization strategy recommended in the original GCN paper:
                - Weights are initialized uniformly from [-1/sqrt(out_features), 1/sqrt(out_features)]
                - Bias terms (if present) use the same initialization range

                This initialization helps stabilize the variance of activations during training.
        """
        # Calculate standard deviation range for initialization
        stdv = 1. / math.sqrt(self.weight.size(1))
        # Initialize weight matrix
        self.weight.data.uniform_(-stdv, stdv)
        # Initialize bias term if present
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
                Implements the core GCN operation: H^{(l+1)} = σ(Ã H^{(l)} W^{(l)} + b)
                :param input (torch.Tensor): Input node feature matrix of shape (N, in_features),
                                          where N is the number of nodes
                :param adj (torch.Tensor): Sparse adjacency matrix of the graph of shape (N, N),
                                        representing node connectivity
                Returns:
                    torch.Tensor: Output feature matrix after graph convolution of shape (N, out_features)
        """
        # Step 1: Feature transformation (H * W)
        support = torch.mm(input, self.weight)
        # Step 2: Neighbor aggregation (A * (H * W))
        output = torch.spmm(adj, support)
        # Step 3: Add bias term (if present)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """
                String representation of the layer for debugging/printing
                Returns:
                    str: Layer description showing input and output feature dimensions
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'