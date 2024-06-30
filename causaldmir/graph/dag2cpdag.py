from itertools import combinations, permutations

from causaldmir.graph import DiGraph, PDAG


def dag2cpdag(dag: DiGraph):
    n = dag.number_of_nodes()
    pdag = PDAG(list(dag.nodes))
    pdag.create_complete_undirected_graph()
    sep_set = {(node_u, node_v): set() for node_u, node_v in permutations(pdag.nodes, 2)}

    for node_u, node_v in combinations(pdag.nodes, 2):
        for condition_size in range(0, n - 1):
            for nodes_z in combinations(pdag.node_set - {node_u, node_v}, condition_size):
                if dag.is_d_separate(node_u, node_v, nodes_z):
                    sep_set[(node_u, node_v)] |= set(nodes_z)
                    sep_set[(node_v, node_u)] |= set(nodes_z)
                    if pdag.is_connected(node_u, node_v):
                        pdag.remove_edge(node_u, node_v)

    pdag.rule0(sep_set=sep_set, verbose=True)

    pdag.orient_by_meek_rules(verbose=True)

    return pdag
