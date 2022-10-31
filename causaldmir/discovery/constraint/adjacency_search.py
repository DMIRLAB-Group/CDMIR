import logging
from itertools import combinations, permutations

from causaldmir.graph import PDAG


def adjacency_search(indep_test, var_names, alpha=0.05, verbose=False):
    g = PDAG(var_names)
    g.create_complete_undirected_graph()

    sep_set = {(node_u, node_v): set() for node_u, node_v in permutations(g.nodes, 2)}
    n = len(var_names)
    for condition_size in range(0, n - 1):
        remove_edges = []
        for node_u, node_v in permutations(var_names, 2):
            if not g.is_connected(node_u, node_v):
                continue
            adj_u = list(g.get_neighbours(node_u))
            if node_v in adj_u:
                adj_u.remove(node_v)
            for nodes_z in combinations(adj_u, condition_size):
                pval = indep_test.test(node_u, node_v, nodes_z)[0]
                if pval >= alpha:
                    remove_edges.append((node_u, node_v))
                    sep_set[(node_u, node_v)] |= set(nodes_z)
                    sep_set[(node_v, node_u)] |= set(nodes_z)

        for node_u, node_v in remove_edges:
            if g.is_connected(node_u, node_v):
                g.remove_edge(node_u, node_v)
                if verbose:
                    logging.info(f'Adjacency Search: Remove edge between {node_u} and {node_v}.')

    return g, sep_set


def fast_adjacency_search(independence, var_names, alpha=0.05, verbose=False):
    g = PDAG(var_names)
    g.create_complete_undirected_graph()
    sep_set = {(node_u, node_v): set() for node_u, node_v in permutations(g.nodes, 2)}
    n = len(var_names)

    for condition_size in range(0, n - 1):
        def get_sep_result(node_u, node_v):
            sep_result = None

            def sep_search(sep_result, adj):
                for nodes_z in combinations(adj, condition_size):
                    pval = independence.test(node_u, node_v, nodes_z)[0]
                    if pval >= alpha:
                        if sep_result is None:
                            sep_result = node_u, node_v, set(nodes_z)
                        else:
                            sep_result = node_u, node_v, sep_result[2] | set(nodes_z)
                        sep_set[(node_u, node_v)] |= set(nodes_z)
                        sep_set[(node_v, node_u)] |= set(nodes_z)
                return sep_result

            sep_result = sep_search(sep_result, (node for node in g.get_neighbours(node_u) if node != node_v))
            sep_result = sep_search(sep_result, (node for node in g.get_neighbours(node_v) if node != node_u))

            return sep_result

        nodes_u, nodes_v = (edge[0] for edge in g.edges), (edge[1] for edge in g.edges)
        for result in map(get_sep_result, nodes_u, nodes_v):
            if result is None:
                continue

            g.remove_edge(result[0], result[1])
            sep_set[result[0], result[1]] |= result[2]
    return g, sep_set
