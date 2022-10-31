from causaldmir.graph import Graph


def graph_equal(true_graph: Graph, est_graph: Graph):
    assert type(true_graph) == type(est_graph)
    return (true_graph.to_numpy() == est_graph.to_numpy()).all()
