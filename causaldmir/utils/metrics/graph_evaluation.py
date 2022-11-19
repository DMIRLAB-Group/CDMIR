from itertools import combinations, permutations
from sklearn.metrics import precision_score, recall_score, f1_score

from causaldmir.graph import Graph, PDAG


def _miss_match_graph_type(true_graph: Graph, est_graph: Graph):
    assert type(true_graph) == type(est_graph), 'The graph type cannot be matched.'


def _miss_match_graph_nodes(true_graph: Graph, est_graph: Graph):
    assert tuple(true_graph.nodes) == tuple(est_graph.nodes), 'The graph nodes cannot be matched.'


def graph_equal(true_graph: Graph, est_graph: Graph):
    _miss_match_graph_type(true_graph, est_graph)
    _miss_match_graph_nodes(true_graph, est_graph)
    return (true_graph.to_numpy() == est_graph.to_numpy()).all()


def skeleton_evaluation(true_graph: Graph, est_graph: Graph):
    _miss_match_graph_type(true_graph, est_graph)
    _miss_match_graph_nodes(true_graph, est_graph)

    true_skeleton = [true_graph.is_connected(node_u, node_v) for node_u, node_v in combinations(true_graph.nodes, 2)]
    est_skeleton = [est_graph.is_connected(node_u, node_v) for node_u, node_v in combinations(est_graph.nodes, 2)]

    return {
        'precision': precision_score(true_skeleton, est_skeleton),
        'recall': recall_score(true_skeleton, est_skeleton),
        'f1': f1_score(true_skeleton, est_skeleton)
    }


def arrow_evaluation(true_graph: Graph, est_graph: Graph):
    _miss_match_graph_type(true_graph, est_graph)
    _miss_match_graph_nodes(true_graph, est_graph)

    true_arrow = [true_graph.is_arrow(node_u, node_v) for node_u, node_v in permutations(true_graph.nodes, 2)]
    est_arrow = [est_graph.is_arrow(node_u, node_v) for node_u, node_v in permutations(est_graph.nodes, 2)]

    return {
        'precision': precision_score(true_arrow, est_arrow),
        'recall': recall_score(true_arrow, est_arrow),
        'f1': f1_score(true_arrow, est_arrow)
    }


def directed_edge_evaluation(true_graph: Graph, est_graph: Graph):
    _miss_match_graph_type(true_graph, est_graph)
    _miss_match_graph_nodes(true_graph, est_graph)

    true_directed_edge = [true_graph.is_fully_directed(node_u, node_v) for node_u, node_v in
                          permutations(true_graph.nodes, 2)]
    est_directed_edge = [est_graph.is_fully_directed(node_u, node_v) for node_u, node_v in
                         permutations(est_graph.nodes, 2)]

    return {
        'precision': precision_score(true_directed_edge, est_directed_edge),
        'recall': recall_score(true_directed_edge, est_directed_edge),
        'f1': f1_score(true_directed_edge, est_directed_edge)
    }


def shd(true_pdag: PDAG, est_pdag: PDAG):
    # Tsamardinos, Ioannis, Laura E. Brown, and Constantin F. Aliferis.
    # "The max-min hill-climbing Bayesian network structure learning algorithm."
    # Machine learning 65.1 (2006): 31-78.
    _miss_match_graph_type(true_pdag, est_pdag)
    _miss_match_graph_nodes(true_pdag, est_pdag)

    return sum(0 if true_pdag.get_edge(node_u, node_v) == est_pdag.get_edge(node_u, node_v) else 1
               for node_u, node_v in combinations(true_pdag.nodes, 2))
